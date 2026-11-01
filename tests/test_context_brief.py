"""Pure mock-clock contract tests; no devices, network, models or private data."""

import json
import unittest
from copy import deepcopy

from interaction_context.brief import SemanticBrief, sanitize_change


class SemanticBriefTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.boundary = ("core-1", "bus-1", 0, 0, 0, "session-a")
        self.brief = SemanticBrief(clock=lambda: self.now, interval=1)

    def item(self, kind="posture", source="camera.front", observed=None, ttl=120, **data):
        default = {"presence": {"count": 1, "track_ids": ["anonymous-1"]},
                   "posture": {"state": "healthy", "reasons": [], "enabled": True,
                               "calibrated": True, "generation": 1},
                   "objects": {"counts": {"cup": 1}}}[kind]
        default.update(data)
        observed = self.now if observed is None else observed
        return dict(kind=kind, source=source, subject=None, session_id=None, state="present",
                    observed_at=observed, expires_at=observed + ttl, data=default)

    def snapshot(self, *items, epoch="core-1", version=1):
        return dict(epoch=epoch, version=f"{epoch}:{version}", items=list(items))

    def update(self, *items):
        snapshot = self.snapshot(*items)
        self.brief.update(snapshot, self.boundary)
        return snapshot

    def change(self, **kwargs):
        self.update(self.item(**kwargs))
        self.now += 1
        return self.update(self.item(state="unhealthy", reasons=["torso_lean"], **kwargs))

    def changes(self, snapshot, boundary=None):
        return self.brief.read(snapshot, boundary or self.boundary)["changes"]

    def test_records_real_change_not_revision_or_refresh(self):
        first = self.item()
        self.update(first)
        self.now += 1
        duplicate = self.snapshot(first, version=50)
        self.brief.update(duplicate, self.boundary)
        self.assertEqual([], self.changes(duplicate))
        self.now += 1
        refreshed = self.update(self.item())
        self.assertEqual([], self.changes(refreshed))
        self.now += 1
        changed = self.update(self.item(state="unhealthy"))
        records = self.changes(changed)
        self.assertEqual(1, len(records))
        self.assertEqual("healthy", records[0]["before"]["state"])
        self.assertEqual("unhealthy", records[0]["after"]["state"])
        self.assertEqual(self.now, records[0]["observed_at"])

    def test_out_of_order_and_equal_time_conflicts_do_not_replace_baseline(self):
        snapshot = self.change()
        count = self.brief.stats()["changes"]
        for timestamp in (self.now - 1, self.now):
            self.brief.update(self.snapshot(self.item(observed=timestamp)), self.boundary)
        self.assertEqual(count, self.brief.stats()["changes"])
        self.now += 1
        refreshed = self.update(self.item(state="unhealthy", reasons=["torso_lean"]))
        self.assertEqual(count, self.brief.stats()["changes"])
        self.assertEqual(self.changes(snapshot), self.changes(refreshed))
        self.assertEqual(2, self.brief.stats()["out_of_order"])

    def test_read_never_prepares_or_updates_history(self):
        snapshot = self.change()
        stats = self.brief.stats()
        self.now += 10
        self.assertTrue(self.changes(snapshot))
        fresh_change = self.snapshot(self.item(state="healthy"))
        self.assertEqual([], self.changes(fresh_change))
        self.assertEqual(stats, self.brief.stats())

    def test_expired_source_is_not_available_even_without_tick(self):
        snapshot = self.change(ttl=3)
        self.now += 3
        self.assertEqual([], self.changes(snapshot))
        self.brief.update(snapshot, self.boundary)
        self.assertEqual(0, self.brief.stats()["records"])

    def test_new_snapshot_cannot_revive_history_after_host_tick_stops(self):
        snapshot = self.change(ttl=3)
        self.assertTrue(self.changes(snapshot))
        stats = self.brief.stats()
        self.now += 4
        refreshed = self.snapshot(self.item(state="unhealthy", reasons=["torso_lean"], ttl=3))
        self.assertEqual([], self.changes(refreshed))
        self.assertEqual(stats, self.brief.stats())
        self.brief.update(refreshed, self.boundary)
        self.assertEqual([], self.changes(refreshed))
        self.assertEqual(0, self.brief.stats()["records"])

    def test_observation_gap_cannot_bridge_same_track(self):
        self.update(self.item(ttl=2))
        self.now += 5
        fresh = self.update(self.item(state="unhealthy", ttl=2))
        self.assertEqual([], self.changes(fresh))

    def test_history_timestamps_are_not_refreshed_and_retention_expires(self):
        snapshot = self.change(ttl=1000)
        observed = self.now
        self.now += 200
        refreshed = self.update(self.item(state="unhealthy", reasons=["torso_lean"], ttl=1000))
        self.assertEqual(observed, self.changes(refreshed)[0]["observed_at"])
        self.assertEqual(observed + 300, self.changes(refreshed)[0]["retain_until"])
        self.now += 100
        self.assertEqual([], self.changes(refreshed))
        self.brief.update(refreshed, self.boundary)
        self.assertEqual(0, self.brief.stats()["records"])

    def test_gap_restart_failure_and_session_boundaries_clear(self):
        for index, value in ((1, "bus-2"), (2, 1), (3, 1), (4, 1), (5, "session-b")):
            with self.subTest(index=index):
                self.brief.invalidate()
                snapshot = self.change()
                new_boundary = list(self.boundary)
                new_boundary[index] = value
                new_boundary = tuple(new_boundary)
                self.assertEqual([], self.changes(snapshot, new_boundary))
                self.brief.update(snapshot, new_boundary)
                self.assertEqual(0, self.brief.stats()["records"])

    def test_core_epoch_reset_cannot_reuse_history(self):
        snapshot = self.change()
        snapshot["epoch"] = "new-core"
        self.assertEqual([], self.changes(snapshot))
        self.brief.update(snapshot, self.boundary)
        self.assertEqual(0, self.brief.stats()["records"])

    def test_track_switch_multiple_people_or_generation_change_clear_source(self):
        for change in ("track", "multiple", "generation"):
            with self.subTest(change=change):
                self.brief.invalidate()
                self.update(self.item("presence"), self.item())
                self.now += 1
                snapshot = self.update(self.item("presence"), self.item(state="unhealthy"))
                self.assertTrue(self.changes(snapshot))
                self.now += 1
                presence = self.item("presence")
                posture = self.item(state="unhealthy")
                if change == "track":
                    presence["data"]["track_ids"] = ["anonymous-2"]
                elif change == "multiple":
                    presence["data"].update(count=2, track_ids=["anonymous-1", "anonymous-2"])
                else:
                    posture["data"]["generation"] = 2
                snapshot = self.update(presence, posture)
                self.assertEqual([], self.changes(snapshot))
                self.assertEqual(0, self.brief.stats()["records"])

    def test_bad_source_does_not_drop_other_camera(self):
        front = self.item()
        environment = self.item("objects", source="camera.environment")
        self.update(front, environment)
        self.now += 1
        front = self.item(state="unhealthy")
        environment = self.item("objects", source="camera.environment", counts={"cup": 2})
        self.update(front, environment)
        self.now += 1
        front["state"] = "unavailable"
        snapshot = self.update(front, environment)
        changes = self.changes(snapshot)
        self.assertEqual(1, len(changes))
        self.assertEqual("camera.environment", changes[0]["source"])

    def test_unknown_stale_unavailable_disabled_conflict_clear_not_absence(self):
        for state in ("unknown", "stale", "unavailable", "disabled", "conflict"):
            with self.subTest(state=state):
                self.brief.invalidate()
                snapshot = self.change()
                bad = deepcopy(snapshot["items"][0])
                if state == "conflict":
                    bad["conflict"] = True
                else:
                    bad["state"] = state
                self.now += 1
                snapshot = self.update(bad)
                self.assertEqual([], self.changes(snapshot))
                self.assertEqual(0, self.brief.stats()["records"])
                self.now += 1
                snapshot = self.update(self.item())
                self.assertEqual([], self.changes(snapshot))

    def test_missing_evidence_is_not_observed_absence(self):
        snapshot = self.change()
        self.assertEqual([], self.changes(self.snapshot()))
        self.brief.update(self.snapshot(), self.boundary)
        self.now += 1
        snapshot = self.update(self.item())
        self.assertEqual([], self.changes(snapshot))

    def test_presence_enter_leave_only_invalidates_never_links_track_history(self):
        self.update(self.item("presence"), self.item())
        self.now += 1
        snapshot = self.update(self.item("presence"), self.item(state="unhealthy"))
        self.assertTrue(self.changes(snapshot))
        self.now += 1
        absent = self.item("presence", count=0, track_ids=[])
        absent["state"] = "observed_absent"
        snapshot = self.update(absent, self.item(state="unhealthy"))
        self.assertEqual([], self.changes(snapshot))
        self.now += 1
        snapshot = self.update(self.item("presence"), self.item())
        self.assertEqual([], self.changes(snapshot))
        self.assertEqual(0, self.brief.stats()["records"])

    def test_known_track_ids_and_generation_never_appear_in_output(self):
        self.update(self.item("presence"), self.item())
        self.now += 1
        snapshot = self.update(self.item("presence"), self.item(state="unhealthy"))
        serialized = json.dumps(self.brief.read(snapshot, self.boundary))
        self.assertNotIn("anonymous-1", serialized)
        self.assertNotIn("track_ids", serialized)
        self.assertNotIn("generation", serialized)
        self.assertNotIn("session-a", serialized)

    def test_identity_session_subject_affect_and_raw_are_not_cached(self):
        sentinel = "DO_NOT_KEEP_THIS_PRIVATE_VALUE"
        private = self.item()
        private.update(kind="identity", data={"label": sentinel})
        scoped = self.item()
        scoped.update(session_id=sentinel)
        subject = self.item()
        subject.update(subject=sentinel)
        initial = self.item(reasons=[sentinel], raw=sentinel, embedding=[1.0, 2.0])
        self.update(private, scoped, subject, initial)
        self.now += 1
        current = self.item(state="unhealthy", reasons=["torso_lean", sentinel], raw=sentinel)
        snapshot = self.update(private, scoped, subject, current)
        serialized = json.dumps(self.brief.read(snapshot, self.boundary))
        self.assertNotIn(sentinel, serialized)
        self.assertNotIn(sentinel, repr(self.brief._previous))
        self.assertEqual(["torso_lean"], self.changes(snapshot)[0]["after"]["reasons"])

    def test_object_counts_only_allow_known_labels_and_bounded_numbers(self):
        self.update(self.item("objects"))
        self.now += 1
        snapshot = self.update(self.item("objects", counts={"cup": 2, "private text": 1,
            "laptop": True, "bottle": 1001, "chair": -1}))
        self.assertEqual({"counts": {"cup": 2}}, self.changes(snapshot)[0]["after"])
        self.now += 1
        snapshot = self.update(self.item("objects", counts={"raw": 1}))
        self.assertEqual([], self.changes(snapshot))

    def test_publishes_at_thirty_seconds_not_each_read_or_poll(self):
        self.brief = SemanticBrief(clock=lambda: self.now)
        start = self.now
        self.assertTrue(self.brief.update(self.snapshot(self.item()), self.boundary))
        self.now += 1
        snapshot = self.snapshot(self.item(state="unhealthy"))
        self.assertFalse(self.brief.update(snapshot, self.boundary))
        self.assertEqual([], self.changes(snapshot))
        self.now = start + 29
        self.assertFalse(self.brief.update(snapshot, self.boundary))
        self.assertEqual([], self.changes(snapshot))
        self.now += 1
        self.assertTrue(self.brief.update(snapshot, self.boundary))
        self.assertEqual(1, len(self.changes(snapshot)))
        self.assertEqual(2, self.brief.stats()["preparations"])

    def test_unchanged_semantics_skip_republication_after_interval(self):
        self.brief = SemanticBrief(clock=lambda: self.now)
        start = self.now
        self.assertTrue(self.brief.update(self.snapshot(self.item()), self.boundary))
        for second in range(1, 66):
            self.now = start + second
            snapshot = self.snapshot(self.item(ttl=3), version=second)
            self.assertFalse(self.brief.update(snapshot, self.boundary))
        result = self.brief.read(snapshot, self.boundary)
        self.assertEqual(start, result["prepared_at"])
        self.assertEqual(1, result["stats"]["preparations"])
        self.assertGreater(result["stats"]["skipped_unchanged"], 0)

    def test_published_changes_are_not_recopied_or_redated_on_refresh(self):
        snapshot = self.change()
        published = self.brief._published
        prepared = self.brief.read(snapshot, self.boundary)["prepared_at"]
        preparations = self.brief.stats()["preparations"]
        for _ in range(10):
            self.now += 1
            snapshot = self.snapshot(self.item(state="unhealthy", reasons=["torso_lean"]))
            self.assertFalse(self.brief.update(snapshot, self.boundary))
        self.assertIs(published, self.brief._published)
        self.assertEqual(prepared, self.brief.read(snapshot, self.boundary)["prepared_at"])
        self.assertEqual(preparations, self.brief.stats()["preparations"])

    def test_new_unprepared_state_hides_old_summary(self):
        snapshot = self.change()
        self.assertTrue(self.changes(snapshot))
        new = self.snapshot(self.item(state="suspect", observed=self.now + .5))
        self.assertEqual([], self.changes(new))

    def test_capacity_and_public_result_bounded(self):
        self.update(self.item("objects"))
        for count in range(2, 80):
            self.now += 1
            snapshot = self.update(self.item("objects", counts={"cup": count}))
        self.assertEqual(64, self.brief.stats()["records"])
        self.assertEqual(6, len(self.changes(snapshot)))
        self.assertGreater(self.brief.stats()["evicted"], 0)

    def test_disabled_invalidate_close_do_not_retain(self):
        snapshot = self.change()
        self.brief.enabled = False
        self.assertEqual([], self.changes(snapshot))
        self.brief.update(snapshot, self.boundary)
        self.assertEqual(0, self.brief.stats()["records"])
        self.brief.enabled = True
        self.brief.invalidate()
        snapshot = self.change()
        self.assertTrue(self.changes(snapshot))
        self.brief.close()
        self.assertFalse(self.brief.update(snapshot, self.boundary))
        self.assertEqual([], self.changes(snapshot))
        self.assertEqual(0, self.brief.stats()["records"])
        self.assertFalse(self.brief.stats()["enabled"])

    def test_invalid_bounds_rejected_and_limits_capped(self):
        for kwargs in ({"interval": 0}, {"retention": float("nan")}, {"maximum": 0}):
            with self.assertRaises(ValueError):
                SemanticBrief(**kwargs)
        cache = SemanticBrief(maximum=1000, retention=1000)
        self.assertEqual(64, cache.maximum)
        self.assertEqual(300, cache.retention)

    def test_public_change_sanitizer_uses_same_whitelist(self):
        snapshot = self.change()
        change = self.changes(snapshot)[0]
        change["profile"] = "private"
        change["after"].update(raw="private", reasons=["torso_lean", "private"])
        clean = sanitize_change(change)
        self.assertNotIn("private", json.dumps(clean))
        self.assertEqual(["torso_lean"], clean["after"]["reasons"])
        clean["after"]["reasons"].append("changed-copy")
        self.assertNotIn("changed-copy", change["after"]["reasons"])
        for mutation in ({"kind": "identity"}, {"source": "other-camera"},
                         {"retain_until": self.now + 301}, {"observed_at": float("nan")},
                         {"source_expires_at": self.now}):
            self.assertIsNone(sanitize_change(dict(change, **mutation)))

    def test_public_sanitizer_rejects_unchanged_and_private_or_multipeople(self):
        base = dict(source="camera.front", kind="presence", observed_at=100,
                    source_expires_at=110, retain_until=400,
                    before={"count": 0}, after={"count": 1, "track_ids": ["private"]})
        self.assertEqual({"count": 1}, sanitize_change(base)["after"])
        self.assertIsNone(sanitize_change(dict(base, after={"count": 0})))
        self.assertIsNone(sanitize_change(dict(base, after={"count": 2})))
        self.assertIsNone(sanitize_change(dict(base, after={"count": True})))

    def test_malformed_types_and_huge_numbers_fail_closed_without_raising(self):
        base = self.item()
        malformed = [dict(base, **mutation) for mutation in (
            {"source": []}, {"kind": []}, {"state": []},
            {"observed_at": 10 ** 1000}, {"expires_at": 10 ** 1000},
            {"data": dict(base["data"], state=[])})]
        for item in malformed:
            with self.subTest(item_fields=list(item)):
                snapshot = self.snapshot(item)
                self.brief.update(snapshot, self.boundary)
                self.assertEqual([], self.changes(snapshot))
        for snapshot in (None, [], {"epoch": []}, {"epoch": "core-1", "items": None}):
            self.brief.update(snapshot, self.boundary)
            self.assertEqual([], self.changes(snapshot))
        snapshot = self.change()
        record = self.changes(snapshot)[0]
        for field in ("observed_at", "source_expires_at", "retain_until"):
            self.assertIsNone(sanitize_change(dict(record, **{field: 10 ** 1000})))
        record["after"]["state"] = []
        self.assertIsNone(sanitize_change(record))


if __name__ == "__main__":
    unittest.main()

"""Small deterministic IoU tracker; track IDs are ephemeral, not identities."""

from __future__ import annotations

import secrets


def _iou(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


class FaceTracker:
    def __init__(self, *, confirm_hits=3, iou_threshold=0.25, lost_after=1.5, prefix=None):
        self.confirm_hits = confirm_hits
        self.iou_threshold = iou_threshold
        self.lost_after = lost_after
        self.prefix = prefix or secrets.token_hex(3)
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, now):
        pairs = []
        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                overlap = _iou(track["bbox"], detection["bbox"])
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, track_id, index))
        matched_tracks, matched_detections = set(), set()
        for _overlap, track_id, index in sorted(pairs, reverse=True):
            if track_id in matched_tracks or index in matched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[index]
            track.update(bbox=detection["bbox"], score=detection["score"],
                         last_seen=now, hits=track["hits"] + 1)
            matched_tracks.add(track_id)
            matched_detections.add(index)

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = f"face_{self.prefix}_{self.next_id}"
            self.next_id += 1
            self.tracks[track_id] = {
                "track_id": track_id, "bbox": detection["bbox"],
                "score": detection["score"], "first_seen": now,
                "last_seen": now, "hits": 1, "confirmed": False,
            }

        events = []
        for track_id, track in list(self.tracks.items()):
            if not track["confirmed"] and track["hits"] >= self.confirm_hits:
                track["confirmed"] = True
                events.append({"type": "entered", **self._public(track, now)})
            if now - track["last_seen"] >= self.lost_after:
                if track["confirmed"]:
                    events.append({"type": "left", **self._public(track, now)})
                del self.tracks[track_id]
        return events

    def snapshot(self, now):
        tracks = [self._public(track, now) for track in self.tracks.values()]
        tracks.sort(key=lambda value: value["track_id"])
        return {
            "face_count": sum(item["state"] == "confirmed" for item in tracks),
            "candidate_count": sum(item["state"] == "candidate" for item in tracks),
            "tracks": tracks,
        }

    @staticmethod
    def _public(track, now):
        return {
            "track_id": track["track_id"],
            "state": "confirmed" if track["confirmed"] else "candidate",
            "bbox": [round(float(value), 5) for value in track["bbox"]],
            "score": round(float(track["score"]), 4),
            "present_seconds": round(max(0.0, now - track["first_seen"]), 2),
            "last_seen_seconds_ago": round(max(0.0, now - track["last_seen"]), 2),
            "identity_status": "unassigned",
        }

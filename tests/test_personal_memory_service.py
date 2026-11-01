import tempfile
import unittest
from pathlib import Path

from personal_memory_service.store import PersonalMemoryError, PersonalMemoryStore


class PersonalMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = PersonalMemoryStore(Path(self.temp.name) / "memory.sqlite3", max_memories=2)

    def tearDown(self): self.temp.cleanup()

    def test_profile_requires_consent_and_is_allowlisted(self):
        with self.assertRaises(PersonalMemoryError):
            self.store.profile_set("local-user", "preferred_name", "小明", consent="implicit")
        value = self.store.profile_set(
            "local-user", "preferred_name", "小明", consent="user_confirmed"
        )
        self.assertEqual(value["fields"]["preferred_name"]["value"], "小明")
        with self.assertRaises(PersonalMemoryError):
            self.store.profile_set("local-user", "phone", "123", consent="user_confirmed")

    def test_memory_recall_update_and_forget(self):
        first = self.store.memory_put(
            "local-user", "preference", "favorite_topic", "机器人",
            source="explicit-dialogue", consent="user_confirmed", tags=["聊天"],
        )
        updated = self.store.memory_put(
            "local-user", "preference", "favorite_topic", "星星和机器人",
            source="explicit-dialogue", consent="user_confirmed", tags=["聊天"],
        )
        self.assertEqual(first["memory_id"], updated["memory_id"])
        recalled = self.store.memory_recall("local-user", query="机器人")
        self.assertEqual(recalled["items"][0]["value"], "星星和机器人")
        self.assertTrue(self.store.memory_forget("local-user", first["memory_id"])["forgotten"])

    def test_secrets_are_rejected_and_reset_is_explicit(self):
        with self.assertRaises(PersonalMemoryError):
            self.store.memory_put(
                "local-user", "stable_fact", "credential", "sk-secretvalue123456",
                source="dialogue", consent="user_confirmed",
            )
        self.store.profile_set("local-user", "language", "zh-CN", consent="user_confirmed")
        result = self.store.reset("local-user", consent="guardian_confirmed")
        self.assertEqual(result["removed_fields"], 1)


if __name__ == "__main__": unittest.main()


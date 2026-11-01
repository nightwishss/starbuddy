import unittest

from orchestration.turn_manager import TurnManager


class TurnManagerTests(unittest.TestCase):
    def test_cancel_only_current_output_and_reset_state(self):
        cancelled=[]
        manager=TurnManager(cancelled.append)
        first=manager.begin()
        self.assertTrue(manager.attach_output(first,"output_one"))
        self.assertTrue(manager.cancel())
        self.assertEqual(cancelled,["output_one"])
        self.assertEqual(manager.snapshot()["phase"],"idle")
        self.assertFalse(manager.cancel())

    def test_new_turn_invalidates_old_output_and_late_completion(self):
        cancelled=[]
        manager=TurnManager(cancelled.append)
        first=manager.begin()
        manager.attach_output(first,"output_one")
        second=manager.begin("listening")
        self.assertEqual(cancelled,["output_one"])
        self.assertFalse(manager.current(first))
        self.assertFalse(manager.attach_output(first,"output_late"))
        self.assertEqual(cancelled,["output_one","output_late"])
        self.assertFalse(manager.output_finished("output_one"))
        self.assertTrue(manager.transition(second,"thinking"))
        manager.attach_output(second,"output_two")
        self.assertTrue(manager.output_finished("output_two"))
        self.assertEqual(manager.snapshot()["phase"],"idle")


if __name__ == "__main__": unittest.main()

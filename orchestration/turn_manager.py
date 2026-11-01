"""Adapter-neutral owner for a single interruptible interaction turn.

This manager holds only transient control state. Conversation history remains in
ConversationService and audible playback remains owned by OutputService.
"""

from __future__ import annotations

import threading
import uuid


class TurnManager:
    def __init__(self, cancel_output):
        self.cancel_output = cancel_output
        self.lock = threading.RLock()
        self.turn_id = None
        self.phase = "idle"
        self.output_id = None
        self.control_epoch = uuid.uuid4().hex
        self.control_revision = 0

    def begin(self, phase="thinking") -> str:
        if phase not in {"listening", "thinking"}:
            raise ValueError("invalid turn phase")
        with self.lock:
            self.control_revision += 1
            previous = self.output_id
            self.turn_id = "interaction_" + uuid.uuid4().hex
            self.phase = phase
            self.output_id = None
            current = self.turn_id
        if previous is not None:
            self.cancel_output(previous)
        return current

    def current(self, turn_id: str) -> bool:
        with self.lock:
            return self.turn_id == turn_id

    def revoke_pending_inputs(self) -> None:
        """Invalidate delayed input admissions without cancelling another owner."""
        with self.lock:
            self.control_revision += 1

    def transition(self, turn_id: str, phase: str) -> bool:
        if phase not in {"thinking", "speaking", "error"}:
            raise ValueError("invalid turn phase")
        with self.lock:
            if self.turn_id != turn_id:
                return False
            self.phase = phase
            return True

    def attach_output(self, turn_id: str, output_id: str) -> bool:
        with self.lock:
            current = self.turn_id == turn_id
            if current:
                self.phase = "speaking"
                self.output_id = output_id
        if not current:
            self.cancel_output(output_id)
        return current

    def output_finished(self, output_id: str) -> bool:
        with self.lock:
            if self.output_id != output_id:
                return False
            self.output_id = None
            self.turn_id = None
            self.phase = "idle"
            return True

    def cancel(self) -> bool:
        with self.lock:
            # Explicit stop must revoke ambient work even when a very short
            # turn begins and ends between Fusion's bounded status polls.
            self.control_revision += 1
            active = self.turn_id is not None or self.output_id is not None
            previous = self.output_id
            self.turn_id = None
            self.output_id = None
            self.phase = "idle"
        if previous is not None:
            self.cancel_output(previous)
        return active

    def snapshot(self) -> dict:
        with self.lock:
            return {"turn_id": self.turn_id, "phase": self.phase,
                    "output_id": self.output_id, "control_epoch": self.control_epoch,
                    "control_revision": self.control_revision}

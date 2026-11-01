"""跨请求Resource句柄的调用方归属与关闭信息。"""

from dataclasses import dataclass

from runtime.errors import PermissionDeniedError


@dataclass(frozen=True)
class ResourceRecord:
    owner_uid: int
    close_capability: str
    argument_name: str


class ResourceTracker:
    HANDLE_KEYS = ("display_id", "stream_id", "playback_id")
    OPEN_ACTIONS = {
        "display.open": ("display_id", "display.close"),
        "audio.open_playback": ("stream_id", "audio.close_playback"),
        "audio.open_capture": ("stream_id", "audio.close_capture"),
        "audio.start_playback": ("playback_id", "audio.stop_playback"),
        "audio.start_stream_playback": ("playback_id", "audio.stop_playback"),
    }
    CLOSE_ACTIONS = {
        "display.close": "display_id",
        "audio.close_playback": "stream_id",
        "audio.close_capture": "stream_id",
        "audio.stop_playback": "playback_id",
    }
    TERMINAL_STATUS_ACTIONS = {
        "audio.playback_status": ("playback_id", "state", {"completed", "failed", "stopped"}),
        "audio.append_stream_playback": ("playback_id", "state", {"completed", "failed", "stopped"}),
        "audio.finish_stream_playback": ("playback_id", "state", {"completed", "failed", "stopped"}),
    }

    def __init__(self):
        self._records = {}

    def authorize(self, peer_uid, args):
        for key in self.HANDLE_KEYS:
            handle = args.get(key)
            if handle is None:
                continue
            record = self._records.get((key, handle))
            if record is None:
                raise PermissionDeniedError(
                    f"资源不属于当前Runtime或已经关闭: {handle}",
                    details={key: handle})
            if record.owner_uid != peer_uid:
                raise PermissionDeniedError(
                    f"调用方无权访问资源: {handle}",
                    details={key: handle, "owner_uid": record.owner_uid})

    def record_result(self, peer_uid, capability, args, result):
        new_resource = None
        opened = self.OPEN_ACTIONS.get(capability)
        if opened and isinstance(result, dict) and not (
                capability == "audio.start_stream_playback"
                and result.get("state") in {"completed", "failed", "stopped"}):
            key, close_capability = opened
            handle = result.get(key)
            if handle:
                record = ResourceRecord(
                    peer_uid, close_capability, key)
                self._records[(key, handle)] = record
                new_resource = (key, handle, record)

        closed_key = self.CLOSE_ACTIONS.get(capability)
        if closed_key:
            self._records.pop((closed_key, args.get(closed_key)), None)

        terminal = self.TERMINAL_STATUS_ACTIONS.get(capability)
        if terminal and isinstance(result, dict):
            handle_key, status_key, terminal_states = terminal
            if result.get(status_key) in terminal_states:
                self._records.pop((handle_key, args.get(handle_key)), None)
        return new_resource

    def remove(self, key, handle):
        self._records.pop((key, handle), None)

    def records(self):
        return list(self._records.items())

    def clear(self):
        self._records.clear()

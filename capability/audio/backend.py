"""Audio Backend 接口 + File (WAV) backend (v0.1).

backend-neutral 原则: 上层 Audio Capability 只依赖 stream 抽象
(PlaybackStream / CaptureStream), 不知道 file / loopback / ALSA / PipeWire 细节。
未来加 loopback (snd-aloop) 或 physical (USB/BT) backend 时, 上层 contract 不变。

v0.1 实现 File backend: 用 WAV 文件模拟 playback(读源) 与 capture(写汇)。
snd-aloop 已确认 CONFIG_SND_ALOOP=m 且模块文件存在, 但需 sudo modprobe, 留作可选 backend。
"""
import struct
import wave

from runtime.errors import HardwareError, SchemaValidationError


class PlaybackStream:
    """读模式流: 从 WAV 源读 PCM (模拟 playback)。"""

    kind = "playback"

    def __init__(self, path):
        try:
            self.wf = wave.open(path, "rb")
        except FileNotFoundError:
            raise HardwareError(
                f"音频文件不存在: {path}", capability="audio", details={"path": path})
        except wave.Error as e:
            raise SchemaValidationError(
                f"非法 WAV: {path} ({e!r})", capability="audio", details={"path": path}) from e
        self.path = path
        self.sample_rate = self.wf.getframerate()
        self.channels = self.wf.getnchannels()
        self.sample_width = self.wf.getsampwidth()
        self.frames = self.wf.getnframes()
        self.position = 0

    def info(self):
        return {
            "path": self.path, "sample_rate": self.sample_rate,
            "channels": self.channels, "sample_width": self.sample_width,
            "frames": self.frames,
            "duration_seconds": round(self.frames / self.sample_rate, 4)
            if self.sample_rate else 0,
            "position": self.position,
        }

    def read(self, nframes):
        data = self.wf.readframes(nframes)
        n = len(data) // (self.sample_width * self.channels)
        if n == 0:
            return {"samples": [], "frames_read": 0,
                    "position": self.position, "eof": self.position >= self.frames}
        samples = list(struct.unpack("<" + "h" * (n * self.channels), data))
        self.position += n
        return {"samples": samples, "frames_read": n,
                "position": self.position, "eof": self.position >= self.frames}

    def close(self):
        self.wf.close()


class CaptureStream:
    """写模式流: 写 PCM 到 WAV 汇 (模拟 capture)。"""

    kind = "capture"

    def __init__(self, path, sample_rate, channels, sample_width):
        self.path = path
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.frames_written = 0
        try:
            self.wf = wave.open(path, "wb")
            self.wf.setframerate(sample_rate)
            self.wf.setnchannels(channels)
            self.wf.setsampwidth(sample_width)
        except OSError as e:
            raise HardwareError(
                f"无法创建 WAV: {path} ({e!r})", capability="audio", details={"path": path}) from e

    def write(self, samples):
        if not samples:
            return {"frames_written": self.frames_written}
        data = struct.pack("<" + "h" * len(samples), *samples)
        self.wf.writeframes(data)
        self.frames_written += len(samples) // self.channels
        return {"frames_written": self.frames_written}

    def close(self):
        self.wf.close()
        return {"path": self.path, "frames_written": self.frames_written}

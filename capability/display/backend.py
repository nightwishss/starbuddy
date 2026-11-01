"""Display Backend接口、Pillow framebuffer与X11窗口backend。

backend-neutral 原则:
  上层 Display Capability 只依赖 DisplayBackend 接口,
  不知道 framebuffer / X11 / VNC / HDMI 等实现细节。
  未来加 X11/Xvnc backend 或物理 HDMI backend 时, 上层 contract 不变。

X11Backend复用FramebufferBackend进行确定性渲染，再通过一个受控GTK子进程把
同一PNG呈现在指定DISPLAY中。这样capture与屏幕内容来自同一帧，同时不把GTK/X11
细节泄露到上层Capability API。
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from runtime.errors import HardwareError, SchemaValidationError


class DisplayBackend:
    """display 后端的抽象接口 (Resource/Stateful)。"""

    kind = "abstract"

    def show_text(self, text, position=None, color="white"):
        raise NotImplementedError

    def show_image(self, path):
        raise NotImplementedError

    def clear(self, color="black"):
        raise NotImplementedError

    def capture(self, path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class FramebufferBackend(DisplayBackend):
    """Pillow 软件 framebuffer (内存 RGB 图像) — v0.1 的 Virtual/Software backend。"""

    kind = "framebuffer"

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), "black")
        self.draw = ImageDraw.Draw(self.image)

    def show_text(self, text, position=None, color="white"):
        if position is None:
            position = (20, 20)
        self.draw.text(position, text, fill=color)
        return {"width": self.width, "height": self.height}

    def show_image(self, path):
        try:
            img = Image.open(path)
            img = img.convert("RGB")
            img.load()
        except FileNotFoundError:
            raise HardwareError(
                f"图片不存在: {path}", capability="display", details={"path": path})
        except Exception as e:
            raise SchemaValidationError(
                f"不支持的图片格式: {path} ({e!r})",
                capability="display", details={"path": path}) from e
        img.thumbnail((self.width, self.height))
        self.image.paste(img, (0, 0))
        return {"size": list(img.size)}

    def clear(self, color="black"):
        self.draw.rectangle((0, 0, self.width, self.height), fill=color)
        return {}

    def capture(self, path):
        try:
            self.image.save(path)
        except Exception as e:
            raise HardwareError(
                f"截图失败 {path}: {e!r}", capability="display", details={"path": path}) from e
        return {"path": path}

    def close(self):
        self.image = None
        self.draw = None


class X11Backend(FramebufferBackend):
    """Pillow渲染 + GTK3 X11窗口镜像backend。"""

    kind = "x11"

    @staticmethod
    def is_available():
        return importlib.util.find_spec("gi") is not None

    def __init__(self, width, height, display, title="VQ Display"):
        if not display:
            raise HardwareError(
                "X11 backend需要display参数或DISPLAY环境变量",
                capability="display")
        if not self.is_available():
            raise HardwareError(
                "X11 backend需要PyGObject/GTK3",
                capability="display")

        super().__init__(width, height)
        self.display = display
        self.title = title
        self._tmp = tempfile.TemporaryDirectory(prefix="vq-display-")
        base = Path(self._tmp.name)
        self._frame_path = base / "frame.png"
        self._ready_path = base / "ready"
        self._publish()

        helper = Path(__file__).with_name("x11_viewer.py")
        env = dict(os.environ)
        env["DISPLAY"] = display
        self.process = subprocess.Popen(
            [sys.executable, str(helper), "--image", str(self._frame_path),
             "--ready", str(self._ready_path), "--title", title,
             "--width", str(width), "--height", str(height)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self._wait_ready()
        except Exception:
            self._tmp.cleanup()
            raise

    def _wait_ready(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready_path.exists():
                return
            if self.process.poll() is not None:
                error = self.process.stderr.read().strip()
                self._cleanup_process()
                raise HardwareError(
                    f"X11窗口启动失败: {error or 'viewer提前退出'}",
                    capability="display",
                    details={"display": self.display})
            time.sleep(0.05)
        self._cleanup_process()
        raise HardwareError(
            "X11窗口启动超时", capability="display",
            details={"display": self.display})

    def _publish(self):
        next_path = self._frame_path.with_suffix(".next.png")
        self.image.save(next_path)
        os.replace(next_path, self._frame_path)

    def show_text(self, text, position=None, color="white"):
        result = super().show_text(text, position=position, color=color)
        self._publish()
        return result

    def show_image(self, path):
        result = super().show_image(path)
        self._publish()
        return result

    def clear(self, color="black"):
        result = super().clear(color=color)
        self._publish()
        return result

    def _cleanup_process(self):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def close(self):
        self._cleanup_process()
        super().close()
        self._tmp.cleanup()

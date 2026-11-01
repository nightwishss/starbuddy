#!/usr/bin/env python3
"""X11Backend的内部GTK3 viewer；不是公共Capability API。"""

import argparse
import os
import signal

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GdkX11", "3.0")
from gi.repository import GdkX11, GLib, Gtk  # noqa: E402,F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    window = Gtk.Window(title=args.title)
    window.set_default_size(args.width, args.height)
    window.connect("destroy", Gtk.main_quit)
    image = Gtk.Image.new_from_file(args.image)
    window.add(image)
    window.show_all()

    state = {"mtime": 0}

    def refresh():
        try:
            mtime = os.stat(args.image).st_mtime_ns
            if mtime != state["mtime"]:
                image.set_from_file(args.image)
                state["mtime"] = mtime
        except OSError:
            pass
        return True

    GLib.timeout_add(100, refresh)
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(Gtk.main_quit))
    with open(args.ready, "w", encoding="utf-8") as ready:
        ready.write(str(window.get_window().get_xid()))
    Gtk.main()


if __name__ == "__main__":
    main()

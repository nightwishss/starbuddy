#!/usr/bin/env python3
"""Minimal local-only WebKit shell for the HDMI StarBuddy display."""

from __future__ import annotations

import logging
import os
import threading
import time

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.1")
from gi.repository import Gio, GLib, Gtk, WebKit2  # noqa: E402

from .kiosk_watchdog import (
    KioskRecoveryPolicy,
    RecoveryDecision,
    EXPECTED_TITLE_PREFIX,
    health_url_for,
    notify_systemd,
    probe_url,
    process_memory_snapshot,
    wait_for_urls,
)


LOG = logging.getLogger("vq.display.kiosk")
KIOSK_URI = os.environ.get("VQ_KIOSK_URI", "http://127.0.0.1:8765/")
HEARTBEAT_SCRIPT = (
    "document.readyState === 'complete' && Boolean(document.documentElement) "
    f"&& document.title.startsWith('{EXPECTED_TITLE_PREFIX}')"
)
# A real, opaque viewport pixel requests damage, without a continuous animation
# or queue_draw() (which could acknowledge a repaint of the old cached frame).
# Native Gtk::draw is render *activity*, not proof of HDMI scanout/pixel freshness.
RENDER_PROBE_SCRIPT = """(() => {
  if (document.hidden) return 'hidden';
  if (document.readyState !== 'complete' || !document.body ||
      !document.title.startsWith('StarBuddy')) return 'not_ready';
  let pixel = document.getElementById('__vq_render_probe');
  if (!pixel) {
    pixel = document.createElement('div');
    pixel.id = '__vq_render_probe';
    pixel.setAttribute('aria-hidden', 'true');
    pixel.style.cssText = 'position:fixed;right:0;bottom:0;width:1px;height:1px;'
      + 'pointer-events:none;z-index:2147483647;contain:strict;';
    document.body.appendChild(pixel);
  }
  pixel.dataset.phase = pixel.dataset.phase === '1' ? '0' : '1';
  pixel.style.backgroundColor = pixel.dataset.phase === '1' ? '#131415' : '#171819';
  return 'visible';
})()"""


def _number(name, default, minimum, maximum, value_type=float):
    value = value_type(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


class Kiosk:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.interval = _number("VQ_KIOSK_WATCHDOG_INTERVAL_SECONDS", 10, 5, 60)
        watchdog_usec = int(os.environ.get("WATCHDOG_USEC", "0"))
        if watchdog_usec > 0:
            self.interval = min(self.interval, max(0.1, watchdog_usec / 3_000_000))
        self.js_timeout = _number("VQ_KIOSK_HEARTBEAT_TIMEOUT_SECONDS", 8, 2, 30)
        self.load_timeout = _number("VQ_KIOSK_LOAD_TIMEOUT_SECONDS", 30, 10, 120)
        self.log_interval = _number("VQ_KIOSK_STATUS_LOG_SECONDS", 60, 30, 600)
        self.render_mode = os.environ.get("VQ_KIOSK_RENDER_WATCHDOG", "shadow")
        if self.render_mode not in {"off", "shadow", "enforce"}:
            raise ValueError("VQ_KIOSK_RENDER_WATCHDOG must be off, shadow or enforce")
        if self.render_mode != "off":
            try:
                gi.require_foreign("cairo")
            except ImportError:
                if self.render_mode == "enforce":
                    raise RuntimeError("render watchdog requires python3-gi-cairo") from None
                LOG.warning("kiosk_render_probe_unavailable dependency=python3-gi-cairo mode=off")
                self.render_mode = "off"
        self.render_interval = _number("VQ_KIOSK_RENDER_INTERVAL_SECONDS", 20, 10, 120)
        self.render_timeout = _number("VQ_KIOSK_RENDER_TIMEOUT_SECONDS", 8, 2, 30)
        self.policy = KioskRecoveryPolicy(
            max_reloads=_number("VQ_KIOSK_MAX_RELOADS", 2, 1, 5, int),
            min_reload_interval=_number("VQ_KIOSK_MIN_RELOAD_SECONDS", 10, 5, 120),
            heartbeat_failures=_number("VQ_KIOSK_HEARTBEAT_FAILURES", 3, 2, 10, int),
            backend_failures=_number("VQ_KIOSK_BACKEND_FAILURES", 3, 2, 10, int),
            memory_soft_mb=_number("VQ_KIOSK_MEMORY_SOFT_MB", 1200, 128, 8192),
            memory_breaches=_number("VQ_KIOSK_MEMORY_BREACHES", 6, 2, 30, int),
            render_required=self.render_mode == "enforce",
            render_failures=_number("VQ_KIOSK_RENDER_FAILURES", 3, 2, 10, int),
            healthy_reset_seconds=_number("VQ_KIOSK_HEALTHY_RESET_SECONDS", 60, 30, 600),
        )
        self.exit_code = 0
        self.load_started_at = None
        self.last_load_success_at = None
        self.last_heartbeat_at = None
        self._heartbeat_token = 0
        self._heartbeat_pending = None
        self._heartbeat_cancel = None
        self._backend_pending = False
        self._recovery_source = None
        self._last_status_log = float("-inf")
        self._draw_serial = 0
        self._render_token = 0
        self._render_pending = None
        self._render_cancel = None
        self._last_render_probe = float("-inf")
        self.last_render_success_at = None
        self._replacing_web_process = False

        self.window = Gtk.Window(title="StarBuddy")
        self.window.set_decorated(False)
        self.window.set_keep_above(True)
        self.window.connect("destroy", Gtk.main_quit)
        settings = WebKit2.Settings()
        settings.set_enable_javascript(True)
        settings.set_enable_page_cache(False)
        self.render_profile = os.environ.get("VQ_KIOSK_RENDER_PROFILE", "auto")
        if self.render_profile not in {"auto", "software"}:
            raise ValueError("VQ_KIOSK_RENDER_PROFILE must be auto or software")
        if self.render_profile == "software":
            # GTK 3 / WebKit 2.52: disable this WebView's accelerated compositor
            # as well as canvas/WebGL. CPU Skia alone retains the EGL compositor
            # and did not bypass the previously observed stuck worker path.
            settings.set_hardware_acceleration_policy(WebKit2.HardwareAccelerationPolicy.NEVER)
            settings.set_enable_2d_canvas_acceleration(False)
            settings.set_enable_webgl(False)
        LOG.info("kiosk_graphics_settings profile=%s acceleration=%s canvas_2d=%s webgl=%s",
                 self.render_profile, settings.get_hardware_acceleration_policy().value_nick,
                 settings.get_enable_2d_canvas_acceleration(), settings.get_enable_webgl())
        self.view = WebKit2.WebView.new_with_settings(settings)
        self.view.connect("context-menu", lambda *_args: True)
        self.view.connect("load-changed", self._load_changed)
        self.view.connect("load-failed", self._load_failed)
        self.view.connect("web-process-terminated", self._web_process_terminated)
        if self.render_mode != "off":
            self.view.connect_after("draw", self._drawn)
        self.view.connect("unmap", self._unmapped)
        self.window.add(self.view)

    def run(self):
        self.window.fullscreen()
        self.window.show_all()
        self.view.load_uri(KIOSK_URI)
        GLib.timeout_add(round(self.interval * 1000), self._watchdog_tick)
        notify_systemd("READY=1\nWATCHDOG=1")
        LOG.info("kiosk_render_profile skia_cpu=%s gpu_painting_threads=%s render_mode=%s",
                 os.environ.get("WEBKIT_SKIA_ENABLE_CPU_RENDERING", "default"),
                 os.environ.get("WEBKIT_SKIA_GPU_PAINTING_THREADS", "default"), self.render_mode)
        Gtk.main()
        return self.exit_code

    def _load_changed(self, _view, event):
        if event == WebKit2.LoadEvent.STARTED:
            self._cancel_heartbeat()
            self._cancel_render()
            self.policy.navigation()
            self.load_started_at = self.clock()
        elif event == WebKit2.LoadEvent.FINISHED:
            self.load_started_at = None
            self.last_load_success_at = self.clock()
            LOG.info("kiosk_load_ok uri=%s", self.view.get_uri())

    def _load_failed(self, _view, _event, uri, error):
        try:
            if error.matches(WebKit2.NetworkError.quark(), WebKit2.NetworkError.CANCELLED):
                return False
        except (AttributeError, TypeError):
            pass
        LOG.warning("kiosk_load_failed uri=%s error=%s", uri, error.message)
        if self._recovery_source is None:
            self._request_recovery(self.policy.failure("load_failed", self.clock()))
        return False

    def _web_process_terminated(self, _view, reason):
        reason_name = getattr(reason, "value_nick", str(reason))
        self._cancel_render()
        if self._replacing_web_process and reason == WebKit2.WebProcessTerminationReason.TERMINATED_BY_API:
            self._replacing_web_process = False
            LOG.info("kiosk_web_process_replaced")
            return
        LOG.error("kiosk_web_process_terminated reason=%s", reason_name)
        if self._recovery_source is None:
            self._request_recovery(self.policy.failure("web_process_terminated", self.clock()))

    def _request_recovery(self, decision: RecoveryDecision | None):
        if decision is None or self.exit_code:
            return
        if decision.action == "exit":
            if self._recovery_source is not None:
                GLib.source_remove(self._recovery_source)
                self._recovery_source = None
            self._exit(decision.reason)
            return
        if self._recovery_source is not None:
            return
        self._cancel_render()
        LOG.warning(
            "kiosk_recovery_scheduled reason=%s delay_seconds=%.1f reload_count=%d",
            decision.reason, decision.delay_seconds, self.policy.reload_count,
        )
        self._recovery_source = GLib.timeout_add(
            max(1, round(decision.delay_seconds * 1000)), self._reload, decision.reason,
        )

    def _reload(self, reason):
        self._recovery_source = None
        self._cancel_heartbeat()
        self._cancel_render()
        self.policy.navigation()
        self.load_started_at = self.clock()
        LOG.warning("kiosk_reload reason=%s reload_count=%d", reason, self.policy.reload_count)
        if reason == "render_stall":
            # A normal reload may reuse the wedged EGL contexts. Replace only
            # this WebProcess, preserving the current URI/window and services.
            self._replacing_web_process = True
            self.view.terminate_web_process()
        self.view.reload()
        return GLib.SOURCE_REMOVE

    def _exit(self, reason):
        self.exit_code = 1
        self._cancel_heartbeat()
        self._cancel_render()
        LOG.error("kiosk_exit_for_systemd_restart reason=%s status=%s", reason, self.policy.status())
        Gtk.main_quit()

    def _watchdog_tick(self):
        notify_systemd("WATCHDOG=1")
        now = self.clock()
        if self._recovery_source is None:
            if self.load_started_at is not None and now - self.load_started_at >= self.load_timeout:
                self.load_started_at = None
                self._request_recovery(self.policy.failure("load_timeout", now))
            if self._recovery_source is None and not self.exit_code:
                # Loading already has its own deadline; don't spend a second
                # recovery budget on readyState=false during a legitimate load.
                if self.load_started_at is None:
                    self._check_javascript(now)
                    self._check_render(now)
                self._check_backend()
        memory = process_memory_snapshot()
        self._request_recovery(self.policy.memory(memory["total_rss_mb"], now))
        if now - self._last_status_log >= self.log_interval:
            self._last_status_log = now
            load_age = None if self.last_load_success_at is None else round(
                now - self.last_load_success_at, 1)
            heartbeat_age = None if self.last_heartbeat_at is None else round(
                now - self.last_heartbeat_at, 1)
            render_age = None if self.last_render_success_at is None else round(
                now - self.last_render_success_at, 1)
            LOG.info(
                "kiosk_health main_rss_mb=%.2f web_rss_mb=%.2f total_rss_mb=%.2f "
                "web_processes=%d last_load_age_seconds=%s "
                "last_heartbeat_age_seconds=%s render_mode=%s "
                "last_render_age_seconds=%s draw_serial=%d status=%s",
                memory["main_rss_mb"], memory["web_process_rss_mb"],
                memory["total_rss_mb"], memory["web_process_count"], load_age,
                heartbeat_age, self.render_mode, render_age, self._draw_serial, self.policy.status(),
            )
        return GLib.SOURCE_CONTINUE

    def _unmapped(self, _view):
        self._cancel_render()
        self.policy.render_unavailable()

    def _drawn(self, _view, _context):
        self._draw_serial += 1
        self._finish_render_if_drawn()
        return False

    def _check_render(self, now):
        if (self.render_mode == "off" or not self.view.get_mapped()
                or not self.view.get_realized()):
            self._unmapped(self.view)
            return
        if self.load_started_at is not None or self.last_load_success_at is None:
            return
        try:
            health_url_for(self.view.get_uri() or "")
        except ValueError:
            self._cancel_render()
            return
        if self._render_pending is not None:
            if now - self._render_pending[1] >= self.render_timeout:
                self._render_expired(now)
            return
        if now - self._last_render_probe < self.render_interval:
            return
        self._last_render_probe = now
        self._render_token += 1
        self._render_pending = [self._render_token, now, self._draw_serial, False]
        self._render_cancel = Gio.Cancellable()
        self.view.run_javascript(RENDER_PROBE_SCRIPT, self._render_cancel,
                                 self._render_script_finished, self._render_token)

    def _render_script_finished(self, view, result, token):
        if self._render_pending is None or token != self._render_pending[0]:
            return
        if self.clock() - self._render_pending[1] >= self.render_timeout:
            self._render_expired(self.clock())
            return
        try:
            state = view.run_javascript_finish(result).get_js_value().to_string()
        except GLib.Error:
            state = "error"
        if state == "hidden":
            self._unmapped(view)
            return
        if state != "visible":
            self._cancel_render()
            self._request_recovery(self.policy.render(False, self.clock()))
            return
        self._render_pending[3] = True
        self._finish_render_if_drawn()

    def _finish_render_if_drawn(self):
        pending = self._render_pending
        if pending is None or not pending[3] or self._draw_serial <= pending[2]:
            return
        if self.clock() - pending[1] >= self.render_timeout:
            self._render_expired(self.clock())
            return
        self.last_render_success_at = self.clock()
        self._cancel_render()
        self.policy.render(True, self.last_render_success_at)

    def _render_expired(self, now):
        self._cancel_render()
        LOG.warning("kiosk_render_no_draw mode=%s", self.render_mode)
        self._request_recovery(self.policy.render(False, now))

    def _cancel_render(self):
        self._render_pending = None
        self._render_token += 1
        if self._render_cancel is not None:
            self._render_cancel.cancel()
            self._render_cancel = None

    def _check_javascript(self, now):
        if self._heartbeat_pending is not None:
            token, started = self._heartbeat_pending
            if now - started < self.js_timeout:
                return
            self._cancel_heartbeat()
            LOG.warning("kiosk_heartbeat_timeout token=%d", token)
            self._request_recovery(self.policy.heartbeat(False, now))
            return
        self._heartbeat_token += 1
        token = self._heartbeat_token
        self._heartbeat_pending = (token, now)
        self._heartbeat_cancel = Gio.Cancellable()
        self.view.run_javascript(HEARTBEAT_SCRIPT, self._heartbeat_cancel,
                                 self._javascript_finished, token)

    def _cancel_heartbeat(self):
        self._heartbeat_pending = None
        self._heartbeat_token += 1
        if self._heartbeat_cancel is not None:
            self._heartbeat_cancel.cancel()
            self._heartbeat_cancel = None

    def _javascript_finished(self, view, result, token):
        if self._heartbeat_pending is None or token != self._heartbeat_pending[0]:
            return
        self._heartbeat_pending = None
        self._heartbeat_cancel = None
        success = False
        try:
            value = view.run_javascript_finish(result).get_js_value()
            success = bool(value.to_boolean())
        except GLib.Error:
            pass
        now = self.clock()
        if self._recovery_source is not None:
            return
        if success:
            self.last_heartbeat_at = now
        self._request_recovery(self.policy.heartbeat(success, now))

    def _check_backend(self):
        if self._backend_pending:
            return
        try:
            health_url = health_url_for(self.view.get_uri() or KIOSK_URI)
        except ValueError:
            # WebKit can briefly show about:blank/an error page. Its JS/load
            # watchdog owns recovery; don't let this stop the GLib timer.
            return
        self._backend_pending = True
        threading.Thread(
            target=self._probe_backend, args=(health_url,),
            name="vq-kiosk-health", daemon=True,
        ).start()

    def _probe_backend(self, health_url):
        success = probe_url(health_url, timeout=min(3.0, self.interval / 2))
        GLib.idle_add(self._backend_finished, success, health_url)

    def _backend_finished(self, success, health_url):
        self._backend_pending = False
        try:
            current_url = health_url_for(self.view.get_uri() or KIOSK_URI)
        except ValueError:
            return GLib.SOURCE_REMOVE
        if current_url != health_url or self.exit_code:
            return GLib.SOURCE_REMOVE  # Late result from a page we already left.
        if not success:
            LOG.warning("kiosk_backend_unhealthy url=%s", health_url)
        if self._recovery_source is None:
            self._request_recovery(self.policy.backend(success, self.clock()))
        return GLib.SOURCE_REMOVE


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    health_url = health_url_for(KIOSK_URI)
    startup_timeout = _number("VQ_KIOSK_STARTUP_TIMEOUT_SECONDS", 30, 5, 180)
    if not wait_for_urls((KIOSK_URI, health_url), startup_timeout):
        LOG.error("kiosk_startup_readiness_timeout")
        raise SystemExit(1)
    raise SystemExit(Kiosk().run())


if __name__ == "__main__":
    main()

/* Bounded, opt-in turn-taking. No browser microphone, history or retry loop. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.StarBuddyContinuousVoice = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const readyPhases = new Set(['idle', 'engaged', 'attending']);
  const terminal = new Set(['completed', 'failed', 'cancelled', 'interrupted', 'expired']);
  const actionBlocked = (state) => ['draft', 'proposed', 'awaiting_confirmation', 'submitting', 'unknown', 'cancelling'].includes((state.action || {}).status);
  function controlFor(state) {
    return state && typeof state.control_epoch === 'string' && /^[a-f0-9]{32}$/.test(state.control_epoch) &&
      Number.isSafeInteger(state.control_revision) && state.control_revision >= 0
      ? { control_epoch: state.control_epoch, control_revision: state.control_revision } : null;
  }
  function quiescent(state) {
    // Missing metadata from an older server is UNKNOWN, never microphone-off.
    return Boolean(state && state.input_active === false && state.voice_ready === true &&
      state.output_pending === false && state.turn_pending !== true &&
      !(state.scheduler && state.scheduler.busy === true));
  }
  function startReady(state) {
    if (!quiescent(state) || !controlFor(state) || !readyPhases.has(state.phase) || actionBlocked(state)) return false;
    const delivery = state.delivery;
    return Boolean(delivery && (delivery.state === 'idle' ||
      terminal.has(delivery.state) && delivery.confirmed_terminal === true));
  }
  class Session {
    constructor(options) {
      this.requestVoice = options.requestVoice;
      this.cancelTurn = options.cancelTurn;
      this.onUpdate = options.onUpdate || (() => {});
      this.onAnswer = options.onAnswer || (() => {});
      this.now = options.now || (() => performance.now());
      // Browser timer methods require Window as their receiver. Never store
      // then invoke the native method as `session.setTimer(...)`.
      this.setTimer = options.setTimer || ((callback, delay) => setTimeout(callback, delay));
      this.clearTimer = options.clearTimer || ((identifier) => clearTimeout(identifier));
      this.maxSessionMs = options.maxSessionMs || 180000;
      this.maxTurns = options.maxTurns || 20;
      this.deliveryTimeoutMs = options.deliveryTimeoutMs || 60000;
      this.freshnessMs = options.freshnessMs || 4500;
      this.cooldownMs = options.cooldownMs || 400;
      this.connected = false;
      this.state = null;
      this.observedAt = -Infinity;
      this.observations = 0;
      this.stopObservation = -1;
      this.enabled = false;
      this.stopping = false;
      this.inFlight = false;
      this.cancelInFlight = false;
      this.cancelFailed = false;
      this.cancelAcknowledged = false;
      this.generation = 0;
      this.turns = 0;
      this.expectedOutput = null;
      this.reason = 'off';
      this.timer = null;
      this.lastView = '';
    }
    get engaged() { return this.enabled || this.stopping || this.inFlight; }
    fresh() { return this.connected && this.now() - this.observedAt <= this.freshnessMs; }
    canStart() { return !this.engaged && this.fresh() && startReady(this.state); }
    observe(state, connected = true) {
      const epochChanged = this.state && state && this.state.control_epoch !== state.control_epoch;
      this.connected = connected === true;
      if (connected) { this.state = state; this.observedAt = this.now(); ++this.observations; }
      if (!connected && this.enabled) this.stop('disconnected');
      else if (epochChanged && this.enabled) this.stop('disconnected');
      if (this.stopping && this.stoppedConfirmed()) {
        this.stopping = false;
        this.cancelFailed = false;
      }
      this.notify();
      this.pump();
    }
    start() {
      if (!this.canStart()) return false;
      this.enabled = true;
      this.reason = 'running';
      this.startedAt = this.now();
      this.turns = 0;
      this.expectedOutput = null;
      this.nextAt = this.now();
      ++this.generation;
      this.notify();
      this.pump();
      return true;
    }
    stop(reason = 'user') {
      const wasEngaged = this.engaged;
      this.enabled = false;
      this.reason = reason;
      ++this.generation;
      this.clearTimer(this.timer); this.timer = null;
      if (wasEngaged || !this.fresh() || !quiescent(this.state)) this.stopping = true;
      this.stopObservation = this.observations;
      this.cancelAcknowledged = false;
      this.notify();
      // Exactly one cancel attempt per explicit stop; no autonomous POST retry.
      if (this.stopping && !this.cancelInFlight) {
        this.cancelInFlight = true;
        Promise.resolve().then(() => this.cancelTurn()).then(() => {
          this.cancelAcknowledged = true;
        }).catch(() => {
          this.cancelFailed = true;
        }).finally(() => {
          this.cancelInFlight = false;
          if (this.stoppedConfirmed()) this.stopping = false;
          this.notify();
        });
      }
      return !this.stopping;
    }
    stopObserved() { return this.observations > this.stopObservation; }
    stoppedConfirmed() {
      return this.cancelAcknowledged && !this.inFlight && this.stopObserved() && this.fresh() && quiescent(this.state);
    }
    tick() {
      this.timer = null;
      if (this.enabled && !this.fresh()) this.stop('disconnected');
      else this.pump();
    }
    schedule(delay = 250) {
      this.clearTimer(this.timer);
      this.timer = this.setTimer(() => this.tick(), delay);
    }
    pump() {
      if (!this.enabled) return;
      if (this.now() - this.startedAt >= this.maxSessionMs) {
        this.stop('limit'); return;
      }
      if (!this.fresh()) { this.schedule(); return; }
      if (actionBlocked(this.state)) { this.stop('confirmation'); return; }
      const delivery = this.state.delivery || {};
      if (this.expectedOutput) {
        if (delivery.job_id !== this.expectedOutput || delivery.state === 'unknown' ||
            terminal.has(delivery.state) && delivery.confirmed_terminal !== true) {
          this.stop('delivery_unknown'); return;
        }
        if (terminal.has(delivery.state) && delivery.state !== 'completed') {
          this.stop('delivery_failed'); return;
        }
        if (delivery.state !== 'completed') {
          if (this.now() - this.outputWaitAt >= this.deliveryTimeoutMs) this.stop('delivery_timeout');
          else { this.notify(); this.schedule(); }
          return;
        }
      }
      if (this.inFlight || !startReady(this.state) || this.now() < this.nextAt) {
        this.notify(); this.schedule(); return;
      }
      if (this.turns >= this.maxTurns) { this.stop('limit'); return; }
      this.expectedOutput = null;
      this.runTurn();
    }
    async runTurn() {
      if (!this.enabled || this.inFlight) return;
      const generation = this.generation;
      this.inFlight = true;
      ++this.turns;
      this.notify(); this.schedule();
      try {
        // This exact server-issued generation fences delayed admission after
        // Stop (including a cancel POST which overtakes the voice POST).
        const control = controlFor(this.state);
        if (!control) { this.stop('request_failed'); return; }
        const data = await this.requestVoice(control);
        if (!this.enabled || generation !== this.generation) return;
        const result = data && data.result && typeof data.result === 'object' ? data.result : data;
        if (!result || result.accepted !== true) {
          this.stop(result && result.reason === 'no_speech' ? 'silence' : 'request_failed'); return;
        }
        if (result.input_cleanup_confirmed !== true) {
          this.stop('input_cleanup_unknown'); return;
        }
        if (typeof result.playback_id !== 'string' || !result.playback_id) {
          this.stop('delivery_unknown'); return;
        }
        this.expectedOutput = result.playback_id;
        this.outputWaitAt = this.now();
        this.nextAt = this.now() + this.cooldownMs;
        this.onAnswer(data);
      } catch (_) {
        if (this.enabled && generation === this.generation) this.stop('request_failed');
      } finally {
        this.inFlight = false;
        if (this.stopping && this.stoppedConfirmed()) this.stopping = false;
        this.notify();
        if (this.enabled) this.schedule();
      }
    }
    view() {
      let stage = 'off';
      if (this.stopping) stage = this.stopObserved() && this.state && this.state.input_active === false ? 'draining' : 'stopping';
      else if (this.enabled) {
        if (!this.fresh()) stage = 'connecting';
        else if (this.state.phase === 'listening' && this.state.input_active === true) stage = 'listening';
        else if (this.state.output_pending === true || this.expectedOutput) stage = 'speaking';
        else if (this.inFlight) stage = this.state.phase === 'thinking' ? 'thinking' : 'starting';
        else stage = 'waiting';
      }
      return { enabled: this.enabled, engaged: this.engaged, stage, reason: this.reason,
        cancelFailed: this.cancelFailed, canStart: this.canStart(), turns: this.turns };
    }
    notify() {
      const view = this.view(), signature = JSON.stringify(view);
      if (signature !== this.lastView) { this.lastView = signature; this.onUpdate(view); }
    }
    dispose() { this.clearTimer(this.timer); this.timer = null; this.enabled = false; ++this.generation; }
  }
  return { Session, quiescent, startReady, controlFor };
});

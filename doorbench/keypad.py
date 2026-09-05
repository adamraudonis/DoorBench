"""Code locks: the state machine of a keypad, and its binding to a MuJoCo model.

The keypad of a DoorBench door is real hardware, not a texture: every button is a body on a slide joint with a
return spring (``geometry/common.py: add_keypad_buttons``), and the lock is released by *pressing those buttons*.
``model.json -> meta.keypad`` (see ``geometry/common.py: keypad_meta_block``) describes the unit; this module turns
button motion into presses and presses into a release.

Two mechanisms, both real hardware (``hardware.KEYPADS``):

``code_kind = "sequence"`` - electronic keypad (Schlage FE595 / BE365, Yale Assure)
    The digits must arrive in order.  A partial entry is cleared after ``code_timeout_s`` of no press, and after
    ``max_attempts`` consecutive wrong codes the keypad ignores every press for ``lockout_s`` (Schlage: 3 -> 30 s).
    A correct code either engages the clutch of the outside lever (``release = "clutch"``) or runs the motor that
    retracts the deadbolt (``release = "motor_bolt"``).

``code_kind = "set"`` - mechanical pushbutton lock (Kaba Simplex 1000)
    There is no electronics and no clock: the buttons of the combination are pressed in ANY order (each button is
    in the combination at most once), and the *lever* is what checks them.  Turning the outside lever with the
    right set in the chamber engages the clutch; turning it with a wrong set clears the chamber and counts as a
    wrong attempt.  No timeout, no lockout.

A press is debounced exactly the way a real keypad does it: the button must travel past ``press_depth_frac`` of
its stroke and stay there for ``debounce_s`` to register, and must come back above ``release_depth_frac`` before
the same button can register again.

Simplification (documented in docs/BENCHMARK.md): once released, the lock stays released for the rest of the
episode.  Real locks re-lock a few seconds later (electronic) or when the lever springs back (Simplex); an
episode is one traversal, so the re-lock timer is not modelled.
"""
from __future__ import annotations


class CodeLock:
    """The lock's logic, with no simulator in it (mirrored by ``viewer/src/keypad.ts``).

    Feed it ``press(label, t)`` / ``lever(t)`` events and ``tick(t)`` every step; read ``unlocked``,
    ``code_entered``, ``wrong_attempts``, ``locked_out``, ``entered`` and ``events``.
    """

    def __init__(self, cfg: dict):
        self.code = str(cfg.get("code") or "")
        self.code_kind = cfg.get("code_kind") or "sequence"
        self.timeout_s = cfg.get("code_timeout_s")
        self.lockout_s = cfg.get("lockout_s")
        self.max_attempts = cfg.get("max_attempts")
        self.reset()

    # -- state ---------------------------------------------------------
    def reset(self):
        self.entered: list[str] = []          # digits so far (sequence) / buttons held in the chamber (set)
        self.unlocked = False
        self.code_entered = False             # the right code was entered at least once this episode
        self.wrong_attempts = 0
        self.events: list[dict] = []
        self._streak = 0                      # consecutive wrong codes (electronic lockout)
        self._locked_out_until = None
        self._last_press_t = None

    @property
    def locked_out(self) -> bool:
        return self._locked_out_until is not None

    @property
    def lockout_remaining(self) -> float:
        return 0.0 if self._locked_out_until is None else float(self._locked_out_until)

    def _event(self, t, kind, **kw):
        self.events.append({"t": round(float(t), 4), "event": kind, **kw})

    # -- inputs --------------------------------------------------------
    def tick(self, t: float):
        """Advance the clock: expire a lockout, clear a stale partial entry."""
        if self._locked_out_until is not None and t >= self._locked_out_until:
            self._locked_out_until = None
            self._event(t, "lockout_expired")
        if self.timeout_s and self.entered and self._last_press_t is not None and t - self._last_press_t > self.timeout_s:
            self._event(t, "timeout", entered=len(self.entered))
            self.entered = []

    def press(self, label: str, t: float):
        """One debounced button press."""
        if self._locked_out_until is not None:
            self._event(t, "ignored_locked_out", key=label)
            return
        self._last_press_t = t
        if self.code_kind == "set":
            if label not in self.entered:
                self.entered.append(label)
            self._event(t, "press", key=label)
            return
        self.entered.append(label)
        self._event(t, "press", key=label)
        if self.code and len(self.entered) >= len(self.code):
            self._evaluate(t)

    def lever(self, t: float):
        """The outside lever was turned (mechanical locks check the chamber on the turn; a wrong set is cleared)."""
        if self.code_kind != "set":
            return
        if not self.entered:
            return
        self._evaluate(t)

    # -- evaluation ----------------------------------------------------
    def _evaluate(self, t: float):
        if self.code_kind == "set":
            ok = sorted(self.entered) == sorted(self.code)
        else:
            ok = "".join(self.entered) == self.code
        entered = "".join(self.entered)
        self.entered = []
        if ok:
            self.code_entered = True
            self._streak = 0
            if not self.unlocked:
                self.unlocked = True
                self._event(t, "unlocked", code=entered)
            return
        self.wrong_attempts += 1
        self._streak += 1
        self._event(t, "wrong_code", entered=entered)
        if self.max_attempts and self.lockout_s and self._streak >= self.max_attempts:
            self._streak = 0
            self._locked_out_until = float(t) + float(self.lockout_s)
            self._event(t, "lockout", until=round(self._locked_out_until, 3), seconds=self.lockout_s)


class Keypad:
    """``CodeLock`` bound to a compiled MuJoCo model: reads the button joints, applies the release.

    ``step(d)`` must be called once per simulation step (it detects presses and the lever turn) and returns True
    on the step the lock is released.  ``apply(d)`` holds the release (clutch range / bolt motor) and is called
    from the same place.
    """

    def __init__(self, mujoco, model, meta: dict, cfg: dict | None = None):
        self.mj = mujoco
        self.m = model
        self.cfg = cfg or (meta or {}).get("keypad") or {}
        self.lock = CodeLock(self.cfg)
        self.travel = float(self.cfg.get("travel_m", 0.0015))
        self.press_thr = self.travel * float(self.cfg.get("press_depth_frac", 0.6))
        self.release_thr = self.travel * float(self.cfg.get("release_depth_frac", 0.3))
        self.debounce = float(self.cfg.get("debounce_s", 0.02))
        self.buttons = []
        for b in self.cfg.get("buttons", []):
            j = self._jid(b["joint"])
            if j >= 0:
                self.buttons.append({"label": b["label"], "jid": j, "qadr": int(model.jnt_qposadr[j]), "dof": int(model.jnt_dofadr[j]), "site": b.get("site")})
        self.by_label = {b["label"]: b for b in self.buttons}
        self.clutch = self._jid(self.cfg.get("clutch_joint"))
        self.bolt = self._jid(self.cfg.get("bolt_joint"))
        self._clutch_range0 = tuple(float(x) for x in model.jnt_range[self.clutch]) if self.clutch >= 0 else None
        self.clutch_open = float(self.cfg.get("clutch_open_rad") or 0.0)
        self.motor_force = float(self.cfg.get("motor_force_N") or 0.0)
        self.bolt_throw = float(self.cfg.get("bolt_throw_m") or 0.0)
        self.release_mode = self.cfg.get("release", "none")
        self._down = {}
        self._latched = {}
        self._lever_high = False
        self.reset()

    # -- helpers -------------------------------------------------------
    def _jid(self, name):
        if not name:
            return -1
        return self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, name)

    @property
    def present(self) -> bool:
        return bool(self.buttons) and bool(self.cfg.get("code"))

    @property
    def unlocked(self) -> bool:
        return self.lock.unlocked

    def reset(self, d=None):
        """New episode: forget the entry and put the clutch back in its locked state."""
        self.lock.reset()
        self._down = {b["label"]: None for b in self.buttons}
        self._latched = {b["label"]: False for b in self.buttons}
        self._lever_high = False
        if self.clutch >= 0 and self._clutch_range0 is not None:
            self.m.jnt_range[self.clutch] = self._clutch_range0

    # -- per-step ------------------------------------------------------
    def step(self, d) -> bool:
        """Detect presses / the lever turn, evaluate the code.  True on the step the lock is released.

        Call once per step, AFTER ``mj_step``; ``apply(d)`` goes before it (its forces must be in ``qfrc_applied``
        when the step is taken)."""
        if not self.buttons:
            return False
        t = float(d.time)
        was = self.lock.unlocked
        self.lock.tick(t)
        for b in self.buttons:
            q = float(d.qpos[b["qadr"]])
            lab = b["label"]
            if q >= self.press_thr:
                if self._down[lab] is None:
                    self._down[lab] = t
                elif not self._latched[lab] and t - self._down[lab] >= self.debounce:
                    self._latched[lab] = True
                    self.lock.press(lab, t)
            elif q <= self.release_thr:
                self._down[lab] = None
                self._latched[lab] = False
        # the outside lever: a mechanical lock checks its chamber when the lever is turned (and clears it either way)
        if self.clutch >= 0 and self.lock.code_kind == "set" and not self.lock.unlocked:
            lo, hi = (float(x) for x in self.m.jnt_range[self.clutch])
            q = float(d.qpos[self.m.jnt_qposadr[self.clutch]])
            span = max(hi - lo, 1e-9)
            if not self._lever_high and (q - lo) >= 0.5 * span:
                self._lever_high = True
                self.lock.lever(t)
            elif self._lever_high and (q - lo) <= 0.2 * span:
                self._lever_high = False
        if self.release_mode == "clutch":
            self.apply(d)          # a joint range costs nothing to set; the bolt motor's force must precede mj_step
        return bool(self.lock.unlocked and not was)

    def apply(self, d):
        """Hold the release: free the outside trim (clutch) or run the bolt motor."""
        if not self.lock.unlocked:
            return
        if self.release_mode == "clutch" and self.clutch >= 0 and self.clutch_open > 0:
            if self.m.jnt_range[self.clutch][1] < self.clutch_open - 1e-9:
                self.m.jnt_range[self.clutch] = [self._clutch_range0[0] if self._clutch_range0 else 0.0, self.clutch_open]
        elif self.release_mode == "motor_bolt" and self.bolt >= 0 and self.motor_force > 0:
            q = float(d.qpos[self.m.jnt_qposadr[self.bolt]])
            if q < self.bolt_throw - 1e-4:
                d.qfrc_applied[self.m.jnt_dofadr[self.bolt]] += self.motor_force

    # -- programmatic finger -------------------------------------------
    def press_force(self) -> float:
        """Fingertip force that bottoms a button out with margin (N)."""
        return 1.6 * float(self.cfg.get("press_force_N", 3.0))

    def hold(self, d, label: str, force: float | None = None):
        """Push one button for this step (a fingertip on its face).  Call before ``mj_step``."""
        b = self.by_label.get(label)
        if b is None:
            raise KeyError(f"{label}: not a button of this keypad ({sorted(self.by_label)})")
        d.qfrc_applied[b["dof"]] += self.press_force() if force is None else float(force)

    def press_sequence(self, d, code: str | None = None, step=None, hold_s: float = 0.08, gap_s: float = 0.06, lever_s: float = 0.25, lever_torque: float = 4.0):
        """Physically enter a code: press each button in turn (and, on a mechanical lock, turn the outside lever).

        ``step`` advances the simulation by one timestep (default: ``mj_step`` + ``self.step``); every press goes
        through the same button joints, debounce and state machine a robot finger would.  Returns ``unlocked``.
        """
        mujoco = self.mj
        dt = float(self.m.opt.timestep)

        def default_step():
            self.apply(d)
            mujoco.mj_step(self.m, d)
            d.qfrc_applied[:] = 0        # applied forces are per-step: a fingertip holds one step at a time
            self.step(d)

        adv = step or default_step
        seq = list(code if code is not None else (self.cfg.get("code") or ""))
        for label in seq:
            for _ in range(max(1, int(round(hold_s / dt)))):
                self.hold(d, label)
                adv()
            for _ in range(max(1, int(round(gap_s / dt)))):
                adv()
        if self.lock.code_kind == "set" and self.clutch >= 0:
            for _ in range(max(1, int(round(lever_s / dt)))):
                d.qfrc_applied[self.m.jnt_dofadr[self.clutch]] += lever_torque
                adv()
            for _ in range(max(1, int(round(gap_s / dt)))):
                adv()
        return self.lock.unlocked

    def wait(self, d, seconds: float, step=None):
        """Let time pass without touching the keypad (inactivity timeout / lockout)."""
        mujoco = self.mj
        dt = float(self.m.opt.timestep)

        def default_step():
            self.apply(d)
            mujoco.mj_step(self.m, d)
            d.qfrc_applied[:] = 0
            self.step(d)

        adv = step or default_step
        for _ in range(max(1, int(round(seconds / dt)))):
            adv()

    def skip_time(self, d, seconds: float):
        """Advance the keypad's clock without simulating (used by the QA gate for the 30 s lockout: nothing moves
        while a locked-out keypad is waiting, so stepping 15000 times would only cost time)."""
        d.time += float(seconds)
        self.lock.tick(float(d.time))

    def state(self) -> dict:
        L = self.lock
        return {"code": self.cfg.get("code"), "code_kind": L.code_kind, "entered": "".join(L.entered), "unlocked": L.unlocked,
                "code_entered": L.code_entered, "wrong_attempts": L.wrong_attempts, "locked_out": L.locked_out,
                "release": self.release_mode, "events": list(L.events)}


def keypad_for(mujoco, model, meta: dict, spec: dict | None = None) -> Keypad | None:
    """The door's keypad, or None.  ``meta`` is ``model.json['meta']``; ``spec`` only supplies the code when an
    older model.json has none."""
    cfg = (meta or {}).get("keypad")
    if not cfg:
        return None
    cfg = dict(cfg)
    if not cfg.get("code") and spec:
        cfg["code"] = (spec.get("lock") or {}).get("code")
    kp = Keypad(mujoco, model, meta, cfg)
    return kp if kp.buttons else None

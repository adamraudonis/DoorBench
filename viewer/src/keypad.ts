// Code locks in the viewer: the same state machine as `doorbench/keypad.py` (order / timeout / lockout for an
// electronic keypad, button *set* + lever for a mechanical Kaba Simplex), plus the helpers DoorView uses to press
// a button in the 3D scene and to apply what the code releases (the outside lever's clutch, the bolt motor).
//
// Kept in step with the Python by tests on both sides: viewer/src/keypad.test.ts mirrors tests/test_keypad_codes.py.
import type { ModelJ } from "./types";

export interface KeypadButtonJ {
  label: string;
  body: string;
  joint: string;
  site: string;
  pos: [number, number, number];
}

export interface KeypadJ {
  lock_model: string;
  keypad_model: string;
  code: string | null;
  code_kind: "sequence" | "set";
  engaged: boolean;
  face: number;
  center: number[];
  pad_size_m: number[];
  layout: string;
  buttons: KeypadButtonJ[];
  travel_m: number;
  press_force_N: number;
  preload_force_N: number;
  press_depth_frac: number;
  release_depth_frac: number;
  debounce_s: number;
  code_timeout_s: number | null;
  lockout_s: number | null;
  max_attempts: number | null;
  release: "clutch" | "motor_bolt" | "physical_catch" | "none";
  clutch_joint: string | null;
  bolt_joint: string | null;
  clutch_locked_rad: number | null;
  clutch_open_rad: number | null;
  bolt_throw_m: number | null;
  motor_force_N: number | null;
  source: string;
  note: string;
}

export interface KeypadEvent { t: number; event: string; key?: string; entered?: string; seconds?: number }

/** The lock's logic (no three.js in here): press / lever / tick, exactly like doorbench/keypad.py CodeLock. */
export class CodeLock {
  readonly code: string;
  readonly codeKind: "sequence" | "set";
  readonly timeout: number | null;
  readonly lockout: number | null;
  readonly maxAttempts: number | null;
  entered: string[] = [];
  unlocked = false;
  codeEntered = false;
  wrongAttempts = 0;
  events: KeypadEvent[] = [];
  private streak = 0;
  private lockedOutUntil: number | null = null;
  private lastPress: number | null = null;

  constructor(cfg: Pick<KeypadJ, "code" | "code_kind" | "code_timeout_s" | "lockout_s" | "max_attempts">) {
    this.code = cfg.code ?? "";
    this.codeKind = cfg.code_kind ?? "sequence";
    this.timeout = cfg.code_timeout_s ?? null;
    this.lockout = cfg.lockout_s ?? null;
    this.maxAttempts = cfg.max_attempts ?? null;
  }

  reset() {
    this.entered = [];
    this.unlocked = false;
    this.codeEntered = false;
    this.wrongAttempts = 0;
    this.events = [];
    this.streak = 0;
    this.lockedOutUntil = null;
    this.lastPress = null;
  }

  get lockedOut(): boolean { return this.lockedOutUntil !== null; }
  /** Seconds of lockout left at time `t` (0 when the keypad is live). */
  lockoutLeft(t: number): number { return this.lockedOutUntil === null ? 0 : Math.max(0, this.lockedOutUntil - t); }

  private event(t: number, event: string, extra: Partial<KeypadEvent> = {}) {
    this.events.push({ t: Math.round(t * 1e4) / 1e4, event, ...extra });
  }

  /** Advance the clock: expire a lockout, clear a stale partial entry. */
  tick(t: number) {
    if (this.lockedOutUntil !== null && t >= this.lockedOutUntil) {
      this.lockedOutUntil = null;
      this.event(t, "lockout_expired");
    }
    if (this.timeout && this.entered.length && this.lastPress !== null && t - this.lastPress > this.timeout) {
      this.event(t, "timeout");
      this.entered = [];
    }
  }

  press(label: string, t: number) {
    if (this.lockedOutUntil !== null) { this.event(t, "ignored_locked_out", { key: label }); return; }
    this.lastPress = t;
    if (this.codeKind === "set") {
      if (!this.entered.includes(label)) this.entered.push(label);
      this.event(t, "press", { key: label });
      return;
    }
    this.entered.push(label);
    this.event(t, "press", { key: label });
    if (this.code && this.entered.length >= this.code.length) this.evaluate(t);
  }

  /** The outside lever was turned: a mechanical lock checks its chamber (and clears it either way). */
  lever(t: number) {
    if (this.codeKind !== "set" || !this.entered.length) return;
    this.evaluate(t);
  }

  private evaluate(t: number) {
    const sortStr = (s: string[]) => [...s].sort().join("");
    const ok = this.codeKind === "set"
      ? sortStr(this.entered) === sortStr([...this.code])
      : this.entered.join("") === this.code;
    const entered = this.entered.join("");
    this.entered = [];
    if (ok) {
      this.codeEntered = true;
      this.streak = 0;
      if (!this.unlocked) { this.unlocked = true; this.event(t, "unlocked", { entered }); }
      return;
    }
    this.wrongAttempts += 1;
    this.streak += 1;
    this.event(t, "wrong_code", { entered });
    if (this.maxAttempts && this.lockout && this.streak >= this.maxAttempts) {
      this.streak = 0;
      this.lockedOutUntil = t + this.lockout;
      this.event(t, "lockout", { seconds: this.lockout });
    }
  }
}

/** The door's keypad block, or undefined. */
export function keypadOf(model: ModelJ | null | undefined): KeypadJ | undefined {
  const kp = (model?.meta as any)?.keypad as KeypadJ | undefined;
  return kp && kp.buttons?.length ? kp : undefined;
}

/** Button rows as they sit on the unit (top row first), for drawing the panel in the real layout. */
export function keypadRows(kp: KeypadJ): KeypadButtonJ[][] {
  const cols = kp.layout === "2x5" ? 2 : kp.layout === "3x4" ? 3 : 1;
  const rows: KeypadButtonJ[][] = [];
  for (let i = 0; i < kp.buttons.length; i += cols) rows.push(kp.buttons.slice(i, i + cols));
  return rows;
}

/** What the label of a wrong-code state should say. */
export function keypadStatus(kp: KeypadJ, lock: CodeLock, t: number): string {
  if (lock.lockedOut) return `keypad locked out for ${lock.lockoutLeft(t).toFixed(0)} s (${kp.max_attempts} wrong codes)`;
  if (lock.unlocked) return kp.release === "motor_bolt" ? "unlocked: the motor retracted the deadbolt" : kp.release === "clutch" ? "unlocked: the outside lever is clutched in" : "code accepted (this lock is not thrown)";
  if (!kp.engaged) return "the lock is not thrown; the code is still read";
  return kp.code_kind === "set" ? "press the buttons of the code, then turn the outside lever" : "enter the code";
}

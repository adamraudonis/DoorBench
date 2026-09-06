// bun test — the viewer's code-lock state machine must behave exactly like doorbench/keypad.py (tests mirrored
// from tests/test_keypad_codes.py), and read the real meta.keypad blocks from ../assets.
import { describe, expect, test } from "bun:test";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import type { ModelJ } from "./types";
import { CodeLock, keypadOf, keypadRows, type KeypadJ } from "./keypad";

const ASSETS = path.resolve(process.env.DOORBENCH_TEST_ASSETS || path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..", "assets"), "doors");
const have = existsSync(ASSETS);

const seq = (code = "4821") => new CodeLock({ code, code_kind: "sequence", code_timeout_s: 5, lockout_s: 30, max_attempts: 3 });
const set = (code = "234") => new CodeLock({ code, code_kind: "set", code_timeout_s: null, lockout_s: null, max_attempts: null });

describe("electronic keypad", () => {
  test("the right code unlocks", () => {
    const lk = seq();
    [..."4821"].forEach((c, i) => lk.press(c, 0.1 * i));
    expect(lk.unlocked).toBe(true);
    expect(lk.codeEntered).toBe(true);
    expect(lk.wrongAttempts).toBe(0);
  });

  test("a wrong code does not, and is counted", () => {
    const lk = seq();
    [..."4812"].forEach((c, i) => lk.press(c, 0.1 * i));
    expect(lk.unlocked).toBe(false);
    expect(lk.wrongAttempts).toBe(1);
  });

  test("order matters", () => {
    const lk = seq();
    [..."1248"].forEach((c, i) => lk.press(c, 0.1 * i));
    expect(lk.unlocked).toBe(false);
  });

  test("a partial entry times out", () => {
    const lk = seq();
    lk.press("4", 0);
    lk.press("8", 0.2);
    lk.tick(6);
    expect(lk.entered).toEqual([]);
    expect(lk.wrongAttempts).toBe(0);
    [..."4821"].forEach((c, i) => lk.press(c, 6.1 + 0.1 * i));
    expect(lk.unlocked).toBe(true);
  });

  test("three wrong codes lock the keypad out, and it comes back", () => {
    const lk = seq();
    for (let a = 0; a < 3; a++) [..."4822"].forEach((c, i) => lk.press(c, a + 0.1 * i));
    expect(lk.lockedOut).toBe(true);
    [..."4821"].forEach((c, i) => lk.press(c, 3 + 0.1 * i));
    expect(lk.unlocked).toBe(false);            // even the right code is ignored
    expect(lk.lockoutLeft(3)).toBeGreaterThan(25);
    lk.tick(3 + 30);
    expect(lk.lockedOut).toBe(false);
    [..."4821"].forEach((c, i) => lk.press(c, 34 + 0.1 * i));
    expect(lk.unlocked).toBe(true);
  });
});

describe("mechanical pushbutton lock", () => {
  test("any order, but the lever is what checks it", () => {
    const lk = set();
    [..."432"].forEach((c, i) => lk.press(c, 0.1 * i));
    expect(lk.unlocked).toBe(false);
    lk.lever(1);
    expect(lk.unlocked).toBe(true);
  });

  test("a wrong set is cleared by the lever", () => {
    const lk = set();
    lk.press("2", 0);
    lk.press("5", 0.1);
    lk.lever(0.5);
    expect(lk.unlocked).toBe(false);
    expect(lk.wrongAttempts).toBe(1);
    expect(lk.entered).toEqual([]);
    lk.press("3", 1);
    lk.press("4", 1.1);
    lk.lever(1.5);
    expect(lk.unlocked).toBe(false);
    [..."234"].forEach((c) => lk.press(c, 2));
    lk.lever(2.5);
    expect(lk.unlocked).toBe(true);
  });

  test("no timeout: the chamber holds the buttons", () => {
    const lk = set();
    lk.press("2", 0);
    lk.tick(600);
    expect(lk.entered).toEqual(["2"]);
  });
});

function load(id: string): ModelJ | null {
  const p = path.join(ASSETS, id, "model.json");
  return existsSync(p) ? (JSON.parse(readFileSync(p, "utf8")) as ModelJ) : null;
}

describe.skipIf(!have)("meta.keypad from the dataset", () => {
  test("an electronic keypad lever set: 10 keys in five rows of two, released by the clutch", () => {
    const model = load("db0526_swing_single");
    if (!model) return;
    const kp = keypadOf(model) as KeypadJ;
    expect(kp.code_kind).toBe("sequence");
    expect(kp.release).toBe("clutch");
    expect(kp.buttons.length).toBe(10);
    expect(keypadRows(kp).length).toBe(5);
    expect(kp.code!.length).toBe(4);
    // the code only uses buttons this keypad has, and every button has a joint to drive
    for (const c of kp.code!) expect(kp.buttons.some((b) => b.label === c)).toBe(true);
    for (const b of kp.buttons) expect(b.joint).toContain("keypad_key_");
    // entering it in the viewer unlocks exactly as in the simulator
    const lk = new CodeLock(kp);
    [...kp.code!].forEach((c, i) => lk.press(c, 0.3 * i));
    expect(lk.unlocked).toBe(true);
  });

  test("a mechanical pushbutton lock: five buttons, a set, no timer", () => {
    const model = load("db0166_swing_single");
    if (!model) return;
    const kp = keypadOf(model) as KeypadJ;
    expect(kp.code_kind).toBe("set");
    expect(kp.buttons.length).toBe(5);
    expect(kp.code_timeout_s).toBe(null);
    const lk = new CodeLock(kp);
    [...kp.code!].reverse().forEach((c, i) => lk.press(c, 0.3 * i));
    expect(lk.unlocked).toBe(false);
    lk.lever(3);
    expect(lk.unlocked).toBe(true);
  });

  test("a keypad deadbolt is released by its motor", () => {
    const model = load("db0086_swing_single");
    if (!model) return;
    const kp = keypadOf(model) as KeypadJ;
    expect(kp.release).toBe("motor_bolt");
    expect(kp.bolt_joint).toBe("leaf_deadbolt_slide");
    expect(kp.bolt_throw_m).toBeGreaterThan(0.02);
  });
});

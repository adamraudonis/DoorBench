import type { ModelJ } from "./types";

/** Filter meshes, never parent groups: handles must keep following the hidden leaf. */
export const isMechanism = (semantic: string) =>
  ["mechanism", "track", "operator", "lock", "hinge", "closer", "latch", "sensor"].includes(semantic);

/** Only expose an authored, interactive thumbturn with an actual bolt coupling. */
export function deadboltControls(model: ModelJ) {
  return model.bodies.flatMap(b => {
    const j = b.joint;
    if (!j || j.role !== "lock" || !j.robot_interactive || !/thumbturn/.test(j.name) || !j.range || j.range[1] - j.range[0] < .006) return [];
    const bolt = model.equalities.find(e => e.kind === "joint" && e.b === j.name && /deadbolt/.test(e.a));
    return bolt ? [{ joint: j.name, bolt: bolt.a, label: b.label || "Deadbolt thumbturn" }] : [];
  });
}

export function openingProcedure(model: ModelJ, spec: any): { steps: string[]; note: string } {
  const steps: string[] = [];
  const locked = !!spec?.lock?.engaged;
  const accessible = !!spec?.lock?.robot_side_release;
  const multipoint = Array.isArray(model.meta?.multipoint_locks) && model.meta.multipoint_locks.length > 0;
  const oldMultipoint = spec?.lock?.model === "multipoint" && !multipoint;
  const turns = deadboltControls(model);
  const note = "Model-derived inspection guide. Access depends on the approach side; this is not a validated human motion procedure.";
  if (locked && !accessible) return { steps: ["This task starts locked with no declared release from the approach side. Do not force the handle; use the task’s locked-door outcome."], note };
  if (oldMultipoint) return { steps: [
    "This loaded model has an older multipoint assembly. Its handle and deadbolt thumbturn are separate controls.",
    "Inspect the thumbturn and its coupled bolt separately from the handle. The handle button is not an unlock command.",
    "A complete opening sequence is unavailable for this version; load the rebuilt multipoint model before evaluating its operation.",
  ], note };
  if (locked) {
    if (turns.length) steps.push("From the side with the thumbturn, rotate each deadbolt thumbturn to withdraw its bolt before operating the handle.");
    else if (model.meta?.keypad) steps.push("Enter the accepted credential at the keypad and wait for the lock to release.");
    else return { steps: ["Release is declared, but an explicit unlock procedure has not yet been authored for this mechanism. Inspect its lock controls and native recording before attempting passage."], note };
  } else steps.push("The task starts unlocked. Confirm that all locking points are withdrawn before moving the door.");
  if (multipoint) steps.push("Depress the lever to withdraw the latch and auxiliary locking points. Lifting the lever extends those points; it is not the opening action.");
  else if (model.meta?.operator_coupling === "individual") steps.push("Release each independent latch in turn. Keep the leaf supported until every latch is clear.");
  else if (model.meta?.operator_joint) steps.push("Operate the handle or latch control. Watch the connected latch retract; an independent deadbolt does not retract with the handle.");
  else steps.push("There is no declared manual handle. Check the powered activation or free-passage behavior in the native recording.");
  steps.push("Move the door along its permitted travel only after all catches are clear. Verify clearance before passing through; use the native recording for contact-dependent mechanisms.");
  return { steps, note };
}

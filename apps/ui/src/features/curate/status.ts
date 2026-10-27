// Mechanism status semantics (design §9.3).
//   POST /curate actions: save -> predicted; validate -> validated, or corrected when a step was
//   corrected (the server decides); reject -> rejected.
//   PATCH transitions (anything else is 409 INVALID_TRANSITION):
//     predicted -> validated (no corrected step) | corrected (>= 1 corrected step) | rejected
//     validated -> rejected      corrected -> rejected      rejected -> predicted (reopen)

export type MechanismStatus = "predicted" | "validated" | "corrected" | "rejected";

export const MECHANISM_STATUSES: readonly MechanismStatus[] = ["predicted", "validated", "corrected", "rejected"];

export const STATUS_MEANING: Record<MechanismStatus, string> = {
  predicted: "stored, no verdict yet",
  validated: "every step accepted exactly as predicted",
  corrected: "accepted, with at least one user-corrected step",
  rejected: "wrong; kept as a negative record and never exported",
};

export function isStatus(s: string): s is MechanismStatus {
  return (MECHANISM_STATUSES as readonly string[]).includes(s);
}

/** The statuses a PATCH may move a mechanism to, given how many of its steps were corrected. */
export function allowedTransitions(status: string, nCorrected: number): MechanismStatus[] {
  switch (status) {
    case "predicted":
      return [nCorrected > 0 ? "corrected" : "validated", "rejected"];
    case "validated":
    case "corrected":
      return ["rejected"];
    case "rejected":
      return ["predicted"];
    default:
      return [];
  }
}

/** What "validate" stores for a pathway or mechanism with `nCorrected` corrected steps. */
export function validateTarget(nCorrected: number): MechanismStatus {
  return nCorrected > 0 ? "corrected" : "validated";
}

export function statusBadgeClass(status: string): string {
  if (status === "validated") return "badge badge-ok";
  if (status === "corrected") return "badge cu-badge-corrected";
  if (status === "rejected") return "badge badge-error";
  return "badge";
}

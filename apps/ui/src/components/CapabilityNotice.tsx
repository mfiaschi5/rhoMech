// CapabilityNotice: a feature the server switched off (/api/meta.capabilities, spec v8 port).
//
// Props:
//   name     string   the capability ("predict", "correct", "curate_edit", ...)
//   title    string?  the heading (default "<name> is disabled")
//   compact  boolean? one line instead of a panel (inside another page)
//
// Renders nothing while /api/meta loads or when the capability is on. DOM:
// [data-testid=capability-notice][data-capability=<name>], the reason in
// [data-testid=capability-reason].
import { useCapability } from "../state/store";

export interface CapabilityNoticeProps {
  name: string;
  title?: string;
  compact?: boolean;
}

export function CapabilityNotice({ name, title, compact = false }: CapabilityNoticeProps) {
  const cap = useCapability(name);
  if (!cap.known || cap.enabled) return null;
  const heading = title ?? `${name.replace(/_/g, " ")} is disabled`;
  if (compact) {
    return (
      <div className="capability-notice compact muted" data-testid="capability-notice" data-capability={name} role="note">
        <span className="badge badge-warn">disabled</span> <strong>{heading}:</strong>{" "}
        <span data-testid="capability-reason">{cap.reason}</span>
      </div>
    );
  }
  return (
    <section className="panel capability-notice" data-testid="capability-notice" data-capability={name} role="note">
      <h2>{heading}</h2>
      <p data-testid="capability-reason">{cap.reason}</p>
      <p className="muted">
        The app is being ported to spec v8, whose prediction target is the reaction matrix of each step. Browse, the
        curation database (read-only), tokens, geometry and orbitals keep working.
      </p>
    </section>
  );
}

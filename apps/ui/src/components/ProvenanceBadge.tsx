// ProvenanceBadge: who produced a prediction-bearing response (design §6.2 Provenance).
//
// Props:
//   provenance  Provenance | null | undefined   nothing renders for null
//   compact     boolean      only the predictor and version on one line (default false: with a
//                            <details> listing spec version, fingerprint, chemistry, KS params,
//                            temperature, scope, contact distance and notes)
//
// DOM: <span data-testid="provenance-badge" data-predictor data-spec-version data-chemistry>.
import type { Provenance } from "../api/types.gen";

export interface ProvenanceBadgeProps {
  provenance: Provenance | null | undefined;
  compact?: boolean;
}

export function ProvenanceBadge({ provenance: p, compact = false }: ProvenanceBadgeProps) {
  if (!p) return null;
  const version = p.predictor_version ?? "unversioned";
  const title = `predictor ${p.predictor} (${version}); spec ${p.spec_version}; chemistry ${p.chemistry}`;
  return (
    <span
      className="provenance-badge"
      data-testid="provenance-badge"
      data-predictor={p.predictor}
      data-spec-version={p.spec_version}
      data-chemistry={p.chemistry}
      title={title}
    >
      <span className="badge">{p.predictor}</span> <code className="code">{version}</code>
      {p.chemistry !== "real" && <span className="badge badge-warn">{p.chemistry}</span>}
      {!compact && (
        <details className="provenance-details">
          <summary>provenance</summary>
          <dl className="kv">
            <dt>spec</dt>
            <dd><code>{p.spec_version}</code></dd>
            <dt>physics fingerprint</dt>
            <dd><code>{p.physics_fingerprint}</code></dd>
            <dt>checkpoint</dt>
            <dd><code>{p.checkpoint_hash ?? "none"}</code></dd>
            <dt>KS params</dt>
            <dd>K {p.ks_params.K}, ε {p.ks_params.eps}, τ {p.ks_params.tau} eV</dd>
            <dt>temperature</dt>
            <dd>{p.temperature ?? "default"}</dd>
            <dt>scope</dt>
            <dd>{p.candidate_scope}</dd>
            <dt>contact distance</dt>
            <dd>{p.contact_distance_ang.toFixed(2)} Å</dd>
            <dt>artifacts</dt>
            <dd>{Object.entries(p.artifact_versions).map(([k, v]) => `${k} ${v}`).join(", ")}</dd>
            {p.notes.length > 0 && (
              <>
                <dt>notes</dt>
                <dd>{p.notes.join("; ")}</dd>
              </>
            )}
          </dl>
        </details>
      )}
    </span>
  );
}

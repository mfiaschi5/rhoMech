// PhysicsPanel: the physics of one (donor, donor site, acceptor, acceptor site) arrow, from
// GET /pool/{state}/physics (design §7.3-§7.5, §11.7): the KS terms, G and beta^2 as a bar chart,
// the dominant canonical MO pair with its energy-level diagram, the floor rank "r / N", the
// predictor's score and probability, the frozen head (weight, logit, tau and where they come
// from), the contact distance, the CLASH badge, the KS2 detail, the site-pair block when the
// requested sites are not the contact sites, the flags and the provenance.
//
// Props:
//   target       PhysicsTarget      the state and the tuple
//   predictor    string             predictor whose score / probability / rank is shown
//   temperature  number | null
//   title        string             heading (e.g. "O1→C7, arrow 1 of edge e…")
//   onOpen3D     () => void         the "open 3D" button
//   threedOpen   boolean
//   onClose      () => void
//
// DOM: physics-panel > physics-ks1, physics-ks2, physics-ks3, physics-g, physics-beta2 (each
// a number with data-value), ks-bar-chart, dominant-mo-pair, energy-levels, floor-rank
// ("r / N"), frozen-head, contact-distance, clash-badge[data-clash], provenance-badge, open-3d.
import { getPhysics } from "../../api/client";
import type { PhysicsOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { ProvenanceBadge } from "../../components/ProvenanceBadge";
import { EnergyLevels } from "./EnergyLevels";
import { exact, fmt, fmtProb, fmtSig, moLabel } from "../../components/format";
import { usePendingCached } from "../../api/hooks";
import { KsBarChart } from "./KsBarChart";

export interface PhysicsTarget {
  stateId: string;
  donor: number;
  donorSite: number;
  acceptor: number;
  acceptorSite: number;
}

export interface PhysicsPanelProps {
  target: PhysicsTarget;
  predictor: string;
  temperature: number | null;
  title: string;
  onOpen3D: () => void;
  threedOpen: boolean;
  onClose: () => void;
}

export const SITE_NOTE = "the predictor scores the token pair at its contact sites; the values at your sites are diagnostics";

function physicsKey(t: PhysicsTarget, predictor: string, temperature: number | null): string {
  return `/pool/${t.stateId}/physics?d=${t.donor}&ds=${t.donorSite}&a=${t.acceptor}&as=${t.acceptorSite}&p=${predictor}&T=${temperature ?? ""}`;
}

function Value({ testId, value, unit, defined = true }: { testId: string; value: number; unit: string; defined?: boolean }) {
  return (
    <>
      <span data-testid={testId} data-value={exact(value)} data-defined={defined ? "true" : "false"} className="num" title={`${exact(value)} ${unit}`}>
        {Number.isFinite(value) ? fmtSig(value, 6) : "–"}
      </span>{" "}
      <span className="muted">
        {unit}
        {defined ? "" : " (undefined)"}
      </span>
    </>
  );
}

function Terms({ p }: { p: PhysicsOut }) {
  const u = p.units;
  return (
    <dl className="kv physics-values">
      <dt>KS1</dt>
      <dd>
        <Value testId="physics-ks1" value={p.ks1} unit={u.ks1 ?? "eV"} defined={p.ks1_defined} />
      </dd>
      <dt>KS2</dt>
      <dd>
        <Value testId="physics-ks2" value={p.ks2} unit={u.ks2 ?? "eV"} defined={p.ks2_defined} />
      </dd>
      <dt>KS3</dt>
      <dd>
        <Value testId="physics-ks3" value={p.ks3} unit={u.ks3 ?? "eV"} />
      </dd>
      <dt>G</dt>
      <dd>
        <Value testId="physics-g" value={p.G} unit={u.G ?? "1/eV"} />
      </dd>
      <dt>β²</dt>
      <dd>
        <Value testId="physics-beta2" value={p.beta2} unit={u.beta2 ?? "eV^2"} />
        <span className="muted"> (β {fmtSig(p.beta, 4)} {u.beta ?? "eV"}, |S| {fmtSig(p.amplitude_S, 4)})</span>
      </dd>
      <dt>floor</dt>
      <dd>
        <span className="num">{fmtSig(p.floor, 6)}</span> <span className="muted">eV = -(KS3 + KS2), higher is better</span>
      </dd>
      <dt>floor rank</dt>
      <dd>
        <span data-testid="floor-rank" className="num">
          {p.floor_rank === null ? "-" : p.floor_rank} / {p.n_valid}
        </span>
        {p.floor_rank === null && <span className="muted"> (invalid pair: not ranked)</span>}
      </dd>
      <dt>{p.predictor}</dt>
      <dd>
        score {fmtSig(p.score, 5)}, p {fmtProb(p.probability)}, rank {p.rank ?? "–"}
      </dd>
      <dt>contact distance</dt>
      <dd>
        <span data-testid="contact-distance">{fmt(p.contact_distance_ang, 2)} Å</span>
        <span className="muted"> (site a to site b along the donor lobe; KS2 uses R_contact {fmt(p.ks2_detail.R_contact, 2)} Å)</span>
      </dd>
      <dt>clash</dt>
      <dd>
        <span data-testid="clash-badge" data-clash={p.clash ? "true" : "false"} className={`badge ${p.clash ? "badge-error" : "badge-ok"}`}>
          {p.clash ? "CLASH" : "no clash"}
        </span>
        {p.clash && <span className="muted"> every placement of the site pair clashed (1.5 Å)</span>}
      </dd>
    </dl>
  );
}

function DominantPair({ p }: { p: PhysicsOut }) {
  const d = p.dominant_pair;
  return (
    <div className="dominant-pair" data-testid="dominant-mo-pair" data-k={d.k_canonical} data-m={d.m_canonical} data-index-source={d.index_source}>
      <strong>
        Dominant MO pair: {moLabel(d.k_label)} (MO {d.k_canonical}) → {moLabel(d.m_label)} (MO {d.m_canonical})
      </strong>
      <div>
        E_k {fmt(d.E_k, 3)} eV ({d.donor_species}), E_m {fmt(d.E_m, 3)} eV ({d.acceptor_species}); gap E_k − E_m {fmt(d.gap, 3)} eV
      </div>
      <div>
        term T[k,m] {fmtSig(d.term, 4)} 1/eV = {fmt(100 * d.share, 1)}% of G; β²·term {fmtSig(d.ks3_part, 4)} eV; p_rk {fmt(d.p_rk, 3)}, p_sm {fmt(d.p_sm, 3)}
      </div>
      <div className="muted">
        index source {d.index_source}
        {d.same_molecule ? "; both MOs of one molecule" : "; intermolecular: k and m come from two separate QM runs"}
      </div>
    </div>
  );
}

function FrozenHead({ p }: { p: PhysicsOut }) {
  const f = p.frozen_head;
  return (
    <div className="frozen-head" data-testid="frozen-head" data-source={f.source}>
      <strong>Frozen KS head</strong>: weight {f.weight === null ? "n/a (invalid pair)" : fmt(f.weight, 4)}, logit {fmtSig(f.logit, 5)}, τ {fmt(f.tau, 3)} eV
      <div className="muted">
        source {f.source === "untrained_kernel" ? "untrained kernel (no model checkpoint exists)" : "checkpoint"}; {f.normalization}
      </div>
    </div>
  );
}

function Details({ p }: { p: PhysicsOut }) {
  const k = p.ks2_detail;
  const s = p.site_pair;
  return (
    <>
      <div className="ks2-detail" data-testid="ks2-detail" data-form={k.form}>
        <strong>KS2</strong> ({k.form} form): main centres {k.main_centres.join(", ")}, q {fmt(k.q_donor, 3)} / {fmt(k.q_acceptor, 3)}, R_contact {fmt(k.R_contact, 2)} Å
        {k.R_geometric !== null && `, R ${fmt(k.R_geometric, 2)} Å`}, ε {fmt(k.eps, 1)}
      </div>
      {(!p.requested_sites_are_contact || s) && (
        <div className="site-pair-block" data-testid="site-pair-block" data-contact={p.requested_sites_are_contact ? "true" : "false"}>
          <strong>Sites</strong>: requested {p.donor_site}/{p.acceptor_site}, contact {p.contact_sites.join("/")}
          <div className="muted">{SITE_NOTE}</div>
          {s && (
            <div>
              at your sites: S {fmtSig(s.S, 4)}, β {fmtSig(s.beta, 4)} eV, KS3 {fmtSig(s.ks3_site, 4)} eV{s.clash ? ", CLASH" : ""}
              {s.site_rule_excluded ? ", excluded by the site rule" : ""}
            </div>
          )}
        </div>
      )}
      {!p.pair_valid && <div className="banner banner-warn">pair excluded: {p.reason ?? "invalid"}</div>}
      {p.flags.length > 0 && (
        <div className="row" data-testid="physics-flags">
          {p.flags.map((f) => (
            <span key={f} className="badge badge-warn">
              {f}
            </span>
          ))}
        </div>
      )}
    </>
  );
}

export function PhysicsPanel({ target, predictor, temperature, title, onOpen3D, threedOpen, onClose }: PhysicsPanelProps) {
  const res = usePendingCached<PhysicsOut>(physicsKey(target, predictor, temperature), (o) =>
    getPhysics(target.stateId, { donor: target.donor, acceptor: target.acceptor, donorSite: target.donorSite, acceptorSite: target.acceptorSite, predictor, temperature }, o),
  );
  const p = res.data;
  return (
    <section className="panel physics-panel" data-testid="physics-panel" data-state-id={target.stateId} data-tuple={`${target.donor},${target.donorSite},${target.acceptor},${target.acceptorSite}`} data-loaded={p ? "true" : "false"}>
      <div className="row drawer-head">
        <h3>Physics: {title}</h3>
        <button type="button" className="btn btn-small" data-testid="physics-close" onClick={onClose}>
          close
        </button>
      </div>
      {res.loading && <PendingNote label={res.pending?.label ?? "loading physics"} worker={res.pending?.worker} />}
      <ErrorBanner error={res.error} title="Cannot compute the physics" />
      {res.error && (
        <button type="button" className="btn btn-small" onClick={res.reload}>
          retry
        </button>
      )}
      {p && (
        <>
          <ProvenanceBadge provenance={p.provenance} />
          <Terms p={p} />
          <KsBarChart physics={p} />
          <DominantPair p={p} />
          <EnergyLevels levels={p.energy_levels} dominant={p.dominant_pair} />
          <FrozenHead p={p} />
          <Details p={p} />
        </>
      )}
      <div className="row">
        <button type="button" className="btn btn-primary" data-testid="open-3d" aria-pressed={threedOpen} onClick={onOpen3D}>
          {threedOpen ? "3D open" : "open 3D"}
        </button>
        <span className="muted">the oriented contact placement the kernel evaluated</span>
      </div>
    </section>
  );
}

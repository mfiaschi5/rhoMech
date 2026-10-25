// PropertyTabs: the computed molecule's properties as sortable tables, under the sketch and the
// 3D view. Global (energies, frontier levels, conceptual DFT indices, provenance), Atoms (every
// atom property; diagnostics greyed), Bonds (Wiberg indices, lengths), Orbitals (the shown set)
// and Descriptors (RDKit.js on the sketch). Clicking an atom row selects that record atom (with
// Shift / Ctrl: adds it): the 3D view draws a halo and the sketch highlights the drawn atom.
//
// DOM: tab buttons tab-<name> (aria-selected), panels wb-tab-<name>; rows atom-row
// [data-record-index][data-element], bond-row [data-i][data-j], orbital-table-row [data-index],
// descriptor-row [data-name].
import { useMemo, useState, type ReactNode } from "react";
import type { ApiError } from "../../../api/client";
import type { PendingCached } from "../../../api/hooks";
import type { AtomPropsOut, BondPropsOut, MoleculeOut, MoleculePropertiesOut, OrbitalRowOut, OrbitalSetOut } from "../../../api/types.gen";
import { ErrorBanner } from "../../../components/ErrorBanner";
import { fmt, fmtSig, moLabel } from "../../../components/format";
import { PendingNote } from "../../../components/PendingNote";
import { atomPropDefs } from "../atomProps";
import { centresText } from "../orbitals/OrbitalPanel";
import { selectAtoms, setHover, useSelection } from "../store";
import { SortableTable, type Column } from "./SortableTable";

export const TABS = ["global", "atoms", "bonds", "orbitals", "descriptors"] as const;
export type TabName = (typeof TABS)[number];

export interface PropertyTabsProps {
  molecule: MoleculeOut | null;
  properties: PendingCached<MoleculePropertiesOut>;
  orbitals: PendingCached<OrbitalSetOut>;
  selectedOrbital: number | null;
  onPickOrbital(index: number): void;
  descriptors: Record<string, number> | null;
}

const TAB_LABELS: Record<TabName, string> = { global: "Global", atoms: "Atoms", bonds: "Bonds", orbitals: "Orbitals", descriptors: "RDKit descriptors" };

function Kv({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv wb-kv">
      {rows.map(([k, v]) => (
        <div key={k} className="kv-row" data-testid={`global-${k.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}`}>
          <dt>{k}</dt>
          <dd className="num">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function GlobalTab({ molecule, props }: { molecule: MoleculeOut | null; props: MoleculePropertiesOut }) {
  const g = props.global_props;
  const ev = (v: number | null) => (v === null ? "–" : `${fmt(v, 3)} eV`);
  const rows: [string, ReactNode][] = [
    ["energy", `${fmt(g.energy_ha, 6)} Eh (${fmt(g.energy_ev, 2)} eV)`],
    ["HOMO", ev(g.homo_ev)],
    ["LUMO", ev(g.lumo_ev)],
    ["gap", ev(g.gap_ev)],
    ["μ (chemical potential)", ev(g.mu_ev)],
    ["η (hardness)", ev(g.eta_ev)],
    ["ω (electrophilicity)", ev(g.omega_ev)],
    ["charge / multiplicity", `${g.charge} / ${g.multiplicity}`],
    ["atoms / AOs / MOs", `${g.n_atoms} / ${g.n_ao} / ${g.n_mo}`],
    ["occupied MOs / valence electrons", `${g.n_occupied} / ${fmt(g.n_valence_electrons, 1)}`],
    ["Fermi smeared", g.fermi_smeared ? "yes: outside the closed-shell domain" : "no"],
    ["model localization converged", g.pm_converged === null ? "–" : `${g.pm_converged ? "yes" : "NO"} (${g.pm_sweeps_occ ?? "–"} / ${g.pm_sweeps_virt ?? "–"} sweeps)`],
  ];
  const pv = props.provenance;
  return (
    <div className="wb-global">
      <Kv rows={rows} />
      {g.warnings.length > 0 && <div className="banner banner-warn">warnings: {g.warnings.join(", ")}</div>}
      <details className="wb-small">
        <summary>provenance</summary>
        <Kv
          rows={[
            ["key", <code key="k">{props.key}</code>],
            ["chemistry / method", `${pv.chemistry} / ${pv.method}`],
            ["QM digest", <code key="q">{pv.qm_digest}</code>],
            ["token digest", <code key="t">{pv.token_digest ?? "–"}</code>],
            ["versions", Object.entries(pv.versions).map(([k, v]) => `${k} ${v}`).join(", ")],
            ["mapping", molecule?.mapping ? `${molecule.mapping.n_sketch_atoms} drawn → ${molecule.mapping.n_record_atoms} record atoms${molecule.mapping.chirality_matched ? "" : " (chirality ignored)"}` : (molecule?.mapping_error ?? "none")],
          ]}
        />
        {pv.notes.map((n) => (
          <div key={n} className="muted">
            {n}
          </div>
        ))}
      </details>
    </div>
  );
}

function atomColumns(props: MoleculePropertiesOut): Column<AtomPropsOut>[] {
  const numeric: Column<AtomPropsOut>[] = atomPropDefs(props).map((d) => ({ key: d.key, label: d.label, title: d.title, value: d.get, digits: d.digits }));
  return [
    { key: "index", label: "#", value: (a) => a.index, digits: 0 },
    { key: "element", label: "el", value: (a) => a.element },
    ...numeric,
    { key: "hyb", label: "hybridization", value: (a) => a.diagnostic.hybridization, diagnostic: true },
    { key: "arom", label: "aromatic", value: (a) => a.diagnostic.aromatic, diagnostic: true },
    { key: "fg", label: "functional group", value: (a) => a.diagnostic.fg_label, diagnostic: true },
  ];
}

function AtomsTab({ props }: { props: MoleculePropertiesOut }) {
  const sel = useSelection();
  const cols = useMemo(() => atomColumns(props), [props]);
  return (
    <SortableTable
      testId="atom-table"
      rowTestId="atom-row"
      rows={props.atoms}
      columns={cols}
      rowKey={(a) => a.index}
      rowAttrs={(a) => ({ "data-record-index": String(a.index), "data-element": a.element })}
      isSelected={(a) => sel.atoms.includes(a.index)}
      onRowClick={(a, e) => selectAtoms([a.index], e.shiftKey || e.ctrlKey || e.metaKey)}
      onRowHover={(a) => setHover(a ? a.index : null)}
    />
  );
}

function BondsTab({ props }: { props: MoleculePropertiesOut }) {
  const sel = useSelection();
  const el = (i: number) => `${props.atoms[i]?.element ?? "?"}${i}`;
  const cols: Column<BondPropsOut>[] = [
    { key: "pair", label: "atoms", value: (b) => `${el(b.i)}–${el(b.j)}` },
    { key: "bonded", label: "Lewis bond", value: (b) => b.bonded },
    { key: "order", label: "order", value: (b) => b.order, digits: 0 },
    { key: "wiberg", label: "Wiberg", title: "Wiberg bond index", value: (b) => b.wiberg, digits: 3 },
    { key: "length", label: "length (Å)", value: (b) => b.length_ang, digits: 3 },
  ];
  return (
    <>
      <p className="muted wb-small">Lewis bonds of the record, plus non-bonded pairs with Wiberg index ≥ {fmt(props.wiberg_threshold, 2)}.</p>
      <SortableTable
        testId="bond-table"
        rowTestId="bond-row"
        rows={props.bonds}
        columns={cols}
        rowKey={(b) => `${b.i}-${b.j}`}
        rowAttrs={(b) => ({ "data-i": String(b.i), "data-j": String(b.j), "data-bonded": String(b.bonded) })}
        isSelected={(b) => sel.atoms.includes(b.i) && sel.atoms.includes(b.j)}
        onRowClick={(b, e) => selectAtoms([b.i, b.j], e.shiftKey || e.ctrlKey || e.metaKey)}
      />
    </>
  );
}

function OrbitalsTab({ data, selected, onPick }: { data: OrbitalSetOut; selected: number | null; onPick(i: number): void }) {
  const mi = (f: (m: NonNullable<OrbitalRowOut["model_input"]>) => number) => (r: OrbitalRowOut) => (r.model_input ? f(r.model_input) : null);
  const cols: Column<OrbitalRowOut>[] = [
    { key: "index", label: "#", value: (r) => r.index, digits: 0 },
    { key: "kind", label: "block", value: (r) => r.kind },
    { key: "label", label: "label", value: (r) => (r.label ? moLabel(r.label) : null) },
    { key: "occ", label: "occ", value: (r) => r.occupation, digits: 2 },
    { key: "energy", label: data.set === "canonical" ? "E (eV)" : "E_r (eV)", value: (r) => r.energy_ev, digits: 3 },
    { key: "ncent", label: "centres", value: (r) => r.centres.length, digits: 0 },
    { key: "centres", label: "centre weights", value: (r) => centresText(r, 4) },
    { key: "cos", label: "|cos| a", title: "model input: |cos| of the leading centre's p vector with the axis", value: mi((m) => m.cos_a), digits: 2 },
    { key: "ovl", label: "overlap pop.", title: "model input: signed overlap population of the two leading centres", value: mi((m) => m.overlap_population), digits: 3 },
    { key: "rg", label: "R_g", title: "model input: radius of gyration", value: mi((m) => m.radius_gyration), digits: 3 },
    { key: "retained", label: "retained", title: "spectral weight kept in the ≤ 8 entries", value: (r) => r.retained_fraction, digits: 3 },
    { key: "type", label: "type", value: (r) => r.diagnostic?.type ?? null, diagnostic: true },
    { key: "parents", label: "parents", value: (r) => r.diagnostic?.parents ?? null, diagnostic: true },
  ];
  return (
    <SortableTable
      testId="orbital-table"
      rowTestId="orbital-table-row"
      rows={data.orbitals}
      columns={cols}
      rowKey={(r) => r.index}
      rowAttrs={(r) => ({ "data-index": String(r.index), "data-set": data.set })}
      isSelected={(r) => r.index === selected}
      onRowClick={(r) => onPick(r.index)}
    />
  );
}

function DescriptorsTab({ descriptors }: { descriptors: Record<string, number> | null }) {
  const rows = useMemo(() => Object.entries(descriptors ?? {}).map(([name, value]) => ({ name, value })), [descriptors]);
  if (!descriptors) return <p className="muted">RDKit.js has no descriptors for the current sketch (empty, invalid or RDKit still loading).</p>;
  return (
    <>
      <p className="muted wb-small">From RDKit.js on the current sketch (not from the QM record).</p>
      <SortableTable
        testId="descriptor-table"
        rowTestId="descriptor-row"
        rows={rows}
        columns={[
          { key: "name", label: "descriptor", value: (r) => r.name },
          { key: "value", label: "value", value: (r) => r.value, render: (r) => fmtSig(r.value, 5), digits: 4 },
        ]}
        rowKey={(r) => r.name}
        rowAttrs={(r) => ({ "data-name": r.name })}
      />
    </>
  );
}

function Pending<T>({ res, what, children }: { res: PendingCached<T>; what: string; children(d: T): ReactNode }) {
  return (
    <>
      {res.loading && <PendingNote label={res.pending?.label ?? `loading the ${what}`} worker={res.pending?.worker} />}
      <ErrorBanner error={res.error as ApiError | null} title={`Cannot load the ${what}`} />
      {res.data !== undefined && children(res.data)}
    </>
  );
}

export function PropertyTabs(p: PropertyTabsProps) {
  const [tab, setTab] = useState<TabName>("global");
  const needsMolecule = tab !== "descriptors" && !p.molecule;
  return (
    <section className="panel wb-tabs" data-testid="wb-tabs" data-tab={tab}>
      <div className="wb-tab-bar" role="tablist">
        {TABS.map((t) => (
          <button key={t} type="button" role="tab" data-testid={`tab-${t}`} aria-selected={tab === t} className={`wb-tab${tab === t ? " active" : ""}`} onClick={() => setTab(t)}>
            {TAB_LABELS[t]}
          </button>
        ))}
      </div>
      <div className="wb-tab-body" data-testid={`wb-tab-${tab}`}>
        {needsMolecule && <p className="muted">Compute the molecule to see its properties.</p>}
        {!needsMolecule && tab === "global" && <Pending res={p.properties} what="properties">{(d) => <GlobalTab molecule={p.molecule} props={d} />}</Pending>}
        {!needsMolecule && tab === "atoms" && <Pending res={p.properties} what="properties">{(d) => <AtomsTab props={d} />}</Pending>}
        {!needsMolecule && tab === "bonds" && <Pending res={p.properties} what="properties">{(d) => <BondsTab props={d} />}</Pending>}
        {!needsMolecule && tab === "orbitals" && <Pending res={p.orbitals} what="orbitals">{(d) => <OrbitalsTab data={d} selected={p.selectedOrbital} onPick={p.onPickOrbital} />}</Pending>}
        {tab === "descriptors" && <DescriptorsTab descriptors={p.descriptors} />}
      </div>
    </section>
  );
}

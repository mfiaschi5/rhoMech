// OverrideBuilder: the ordered list of user arrows of a correction (design §4.6, §11.8). The
// arrows are applied in this order (order matters for apply_predicted_arrows); each can be moved
// or removed. Options: "allow invalid" (send arrows the candidate masks exclude), "continue
// prediction" and the number of steps to continue (empty: the server default).
//
// Props:
//   arrows            BuilderArrow[]
//   selected          number | null           the arrow whose sites SitePicker edits
//   onSelect          (i) => void
//   onChange          (arrows) => void        removal and reordering
//   options           OverrideOptions
//   onOptions         (options) => void
//   onSubmit          () => void
//   busy              boolean                 an override job is being created or running
//
// DOM: override-arrow-{i}, override-remove-{i}, override-allow-invalid (checkbox),
// override-continue (number input: the continue steps; empty = server default, which is
// max(1, max_steps - depth)), override-continue-prediction (checkbox, default on), override-submit.
import type { ArrowIn, CandidateOut, CentreOut } from "../../api/types.gen";
import type { TokenArrowLike } from "../../chem/arrowGeom";
import { arrowText } from "./network";
import { NumberField } from "./NumberField";

export interface BuilderArrow extends TokenArrowLike {
  donor_species: string;
  acceptor_species: string;
  donor_centres: CentreOut[];
  acceptor_centres: CentreOut[];
  contact_sites: [number, number];
  valid: boolean;
  reason: string | null;
  diagnostic: Record<string, string>;
}

export interface OverrideOptions {
  allowInvalid: boolean;
  continuePrediction: boolean;
  continueSteps: number | null;
}

export const DEFAULT_OPTIONS: OverrideOptions = { allowInvalid: false, continuePrediction: true, continueSteps: null };

export function builderArrowFrom(c: CandidateOut): BuilderArrow {
  return {
    donor: c.donor,
    donor_site: c.donor_site,
    acceptor: c.acceptor,
    acceptor_site: c.acceptor_site,
    donor_site_map: c.donor_site_map,
    acceptor_site_map: c.acceptor_site_map,
    donor_element: c.donor_site_element,
    acceptor_element: c.acceptor_site_element,
    source_partner_map: null,
    sink_partner_map: null,
    electrons: 2,
    donor_species: c.donor_species,
    acceptor_species: c.acceptor_species,
    donor_centres: c.donor_desc.centres,
    acceptor_centres: c.acceptor_desc.centres,
    contact_sites: c.contact_sites,
    valid: c.valid,
    reason: c.reason,
    diagnostic: c.diagnostic,
  };
}

export const toArrowIn = (a: BuilderArrow): ArrowIn => ({ donor: a.donor, donor_site: a.donor_site, acceptor: a.acceptor, acceptor_site: a.acceptor_site });

export interface OverrideBuilderProps {
  arrows: BuilderArrow[];
  selected: number | null;
  onSelect: (i: number) => void;
  onChange: (arrows: BuilderArrow[]) => void;
  options: OverrideOptions;
  onOptions: (o: OverrideOptions) => void;
  onSubmit: () => void;
  busy: boolean;
  maxSteps?: number;
}

function move<T>(list: T[], i: number, by: -1 | 1): T[] {
  const j = i + by;
  if (j < 0 || j >= list.length) return list;
  const out = [...list];
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}

function ArrowItem({ a, i, n, props }: { a: BuilderArrow; i: number; n: number; props: OverrideBuilderProps }) {
  const selected = props.selected === i;
  return (
    <li className={`override-arrow${selected ? " selected" : ""}${a.valid ? "" : " invalid"}`} data-testid={`override-arrow-${i}`} data-donor={a.donor} data-donor-site={a.donor_site} data-acceptor={a.acceptor} data-acceptor-site={a.acceptor_site}>
      <button type="button" className="legend-button" aria-pressed={selected} onClick={() => props.onSelect(i)} title="edit this arrow's sites">
        <span className="legend-kind">{i + 1}.</span>
        <span className="legend-main">
          <code className="code">{arrowText(a)}</code> #{a.donor} → #{a.acceptor}
          <span className="muted">
            {" "}
            ({a.donor_species} → {a.acceptor_species})
          </span>
          {!a.valid && <span className="badge badge-warn">{a.reason ?? "excluded"}</span>}
        </span>
      </button>
      <button type="button" className="btn btn-small" disabled={i === 0} onClick={() => props.onChange(move(props.arrows, i, -1))} aria-label={`move arrow ${i + 1} up`}>
        ↑
      </button>
      <button type="button" className="btn btn-small" disabled={i === n - 1} onClick={() => props.onChange(move(props.arrows, i, 1))} aria-label={`move arrow ${i + 1} down`}>
        ↓
      </button>
      <button type="button" className="btn btn-small" data-testid={`override-remove-${i}`} onClick={() => props.onChange(props.arrows.filter((_, k) => k !== i))}>
        remove
      </button>
    </li>
  );
}

export function OverrideBuilder(props: OverrideBuilderProps) {
  const { arrows, options, onOptions } = props;
  const invalid = arrows.filter((a) => !a.valid).length;
  return (
    <section className="override-builder stack" data-testid="override-builder">
      <strong>Override arrows (applied in order)</strong>
      {arrows.length === 0 && <div className="muted">No arrow yet: use a candidate row's "use" button.</div>}
      <ol className="override-list">
        {arrows.map((a, i) => (
          <ArrowItem key={`${i}-${a.donor}-${a.donor_site}-${a.acceptor}-${a.acceptor_site}`} a={a} i={i} n={arrows.length} props={props} />
        ))}
      </ol>
      <div className="row">
        <label className="field-inline">
          <input type="checkbox" data-testid="override-allow-invalid" checked={options.allowInvalid} onChange={(e) => onOptions({ ...options, allowInvalid: e.target.checked })} />
          allow invalid
        </label>
        <label className="field-inline">
          <input type="checkbox" data-testid="override-continue-prediction" checked={options.continuePrediction} onChange={(e) => onOptions({ ...options, continuePrediction: e.target.checked })} />
          continue prediction
        </label>
        <NumberField
          label="continue steps"
          testId="override-continue"
          value={options.continueSteps}
          min={0}
          max={props.maxSteps ?? 12}
          allowEmpty
          placeholder="default"
          onChange={(continueSteps) => onOptions({ ...options, continueSteps })}
        />
      </div>
      {invalid > 0 && !options.allowInvalid && <div className="banner banner-warn">{invalid} arrow(s) are excluded by the candidate masks: the server rejects them unless "allow invalid" is on.</div>}
      <div className="row">
        <button type="button" className="btn btn-primary" data-testid="override-submit" disabled={arrows.length === 0 || props.busy} onClick={props.onSubmit}>
          {props.busy ? "applying…" : `apply ${arrows.length} arrow${arrows.length === 1 ? "" : "s"} and ${options.continuePrediction ? "continue" : "stop"}`}
        </button>
      </div>
    </section>
  );
}

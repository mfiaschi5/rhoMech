// SitePicker: the centre atoms of the donor and acceptor tokens of one override arrow, with their
// centre weights; the arrow's current sites are pressed and the contact sites (the argmax-|S|
// pair the predictor scores) are marked (design §7.4, §11.8).
//
// Props:
//   arrow      BuilderArrow | null    the selected override arrow
//   index      number                 its position in the override list
//   onChange   (arrow) => void        a new donor or acceptor site
//
// DOM: <div data-testid="site-picker"> > button[data-testid=site-option-donor-{site}] and
// button[data-testid=site-option-acceptor-{site}] (site = global atom index), aria-pressed.
import type { CentreOut } from "../../api/types.gen";
import { fmt } from "./format";
import type { BuilderArrow } from "./OverrideBuilder";
import { SITE_NOTE } from "./PhysicsPanel";

export interface SitePickerProps {
  arrow: BuilderArrow | null;
  index: number;
  onChange: (arrow: BuilderArrow) => void;
}

function withSite(a: BuilderArrow, side: "donor" | "acceptor", c: CentreOut): BuilderArrow {
  return side === "donor"
    ? { ...a, donor_site: c.site, donor_site_map: c.map, donor_element: c.element }
    : { ...a, acceptor_site: c.site, acceptor_site_map: c.map, acceptor_element: c.element };
}

function Centres({ a, side, onChange }: { a: BuilderArrow; side: "donor" | "acceptor"; onChange: (a: BuilderArrow) => void }) {
  const centres = side === "donor" ? a.donor_centres : a.acceptor_centres;
  const current = side === "donor" ? a.donor_site : a.acceptor_site;
  const contact = side === "donor" ? a.contact_sites[0] : a.contact_sites[1];
  const token = side === "donor" ? a.donor : a.acceptor;
  return (
    <div className="site-centres">
      <span className="field-label">
        {side} #{token} centres
      </span>
      <div className="row">
        {centres.length === 0 && <span className="muted">no centre list for this token</span>}
        {centres.map((c) => (
          <button
            key={c.site}
            type="button"
            className={`btn btn-small site-option${c.site === current ? " btn-primary" : ""}`}
            data-testid={`site-option-${side}-${c.site}`}
            data-map={c.map}
            data-element={c.element}
            data-contact={c.site === contact ? "true" : "false"}
            aria-pressed={c.site === current}
            onClick={() => onChange(withSite(a, side, c))}
            title={`site ${c.site} (map ${c.map}), centre weight ${c.weight}`}
          >
            {c.element}
            {c.map} {fmt(c.weight, 2)}
            {c.site === contact && <span className="badge badge-ok">contact</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

export function SitePicker({ arrow, index, onChange }: SitePickerProps) {
  if (!arrow) {
    return (
      <div className="site-picker muted" data-testid="site-picker">
        Add an arrow from the candidates to choose its sites.
      </div>
    );
  }
  const atContact = arrow.donor_site === arrow.contact_sites[0] && arrow.acceptor_site === arrow.contact_sites[1];
  const contactCentres = {
    donor: arrow.donor_centres.find((c) => c.site === arrow.contact_sites[0]),
    acceptor: arrow.acceptor_centres.find((c) => c.site === arrow.contact_sites[1]),
  };
  const toContact = () => {
    let next = arrow;
    if (contactCentres.donor) next = withSite(next, "donor", contactCentres.donor);
    if (contactCentres.acceptor) next = withSite(next, "acceptor", contactCentres.acceptor);
    onChange(next);
  };
  return (
    <div className="site-picker stack" data-testid="site-picker" data-index={index}>
      <strong>Sites of arrow {index + 1}</strong>
      <Centres a={arrow} side="donor" onChange={onChange} />
      <Centres a={arrow} side="acceptor" onChange={onChange} />
      <div className="row">
        <button type="button" className="btn btn-small" disabled={atContact || !contactCentres.donor || !contactCentres.acceptor} onClick={toContact}>
          use the contact sites
        </button>
        {!atContact && <span className="muted">{SITE_NOTE}</span>}
      </div>
    </div>
  );
}

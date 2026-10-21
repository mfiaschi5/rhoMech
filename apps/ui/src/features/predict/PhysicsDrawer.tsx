// PhysicsDrawer: the side drawer of the Predict job view (design §11.8): the PhysicsPanel of the
// selected arrow (arrow i of the selected edge, or the previewed candidate tuple of the selected
// node) and, after "open 3D", the ThreeDPanel of the same tuple. It sits in its own grid column,
// so it never covers the network, the list or the panels.
//
// Props:
//   net       NetworkOut
//   sel       JobSelection
//   select    (updates) => void
//
// Renders nothing when no arrow is selected. DOM: <aside data-testid="physics-drawer">.
import type { NetworkOut } from "../../api/types.gen";
import { shortId } from "./format";
import { arrowOrdinal, arrowText } from "./network";
import { PhysicsPanel, type PhysicsTarget } from "./PhysicsPanel";
import type { JobSelection, SelectionUpdate } from "./selection";
import { ThreeDPanel } from "./ThreeDPanel";

export interface DrawerTarget {
  target: PhysicsTarget;
  title: string;
}

/** What the drawer shows for this selection, or null (drawer closed). */
export function drawerTarget(net: NetworkOut, sel: JobSelection): DrawerTarget | null {
  if (sel.edge && sel.arrow !== null) {
    const e = net.edges.find((x) => x.id === sel.edge);
    const a = e?.arrows[sel.arrow];
    if (e && a) {
      const target = { stateId: e.source_state_id, donor: a.donor, donorSite: a.donor_site, acceptor: a.acceptor, acceptorSite: a.acceptor_site };
      return { target, title: `${arrowText(a)}, ${arrowOrdinal(e.arrows, sel.arrow)} of step ${shortId(e.id, 10)}` };
    }
  }
  if (sel.node && sel.cand && sel.candPhysics) {
    const n = net.nodes.find((x) => x.id === sel.node);
    if (n) {
      const [donor, donorSite, acceptor, acceptorSite] = sel.cand;
      return { target: { stateId: n.state_id, donor, donorSite, acceptor, acceptorSite }, title: `candidate #${donor} (site ${donorSite}) → #${acceptor} (site ${acceptorSite})` };
    }
  }
  return null;
}

export interface PhysicsDrawerProps {
  net: NetworkOut;
  sel: JobSelection;
  select: (updates: SelectionUpdate) => void;
}

export function PhysicsDrawer({ net, sel, select }: PhysicsDrawerProps) {
  const d = drawerTarget(net, sel);
  if (!d) return null;
  const t = d.target;
  const key = `${t.stateId}/${t.donor}/${t.donorSite}/${t.acceptor}/${t.acceptorSite}`;
  const predictor = net.request.predictor ?? net.provenance.predictor;
  const close = () => select(sel.candPhysics ? { phys: null, threed: null } : { arrow: null, threed: null });
  return (
    <aside className="predict-drawer" data-testid="physics-drawer" data-target={key}>
      <PhysicsPanel
        key={key}
        target={t}
        predictor={predictor}
        temperature={net.params.temperature}
        title={d.title}
        threedOpen={sel.threed}
        onOpen3D={() => select({ threed: 1 })}
        onClose={close}
      />
      {sel.threed && <ThreeDPanel key={`3d:${key}`} target={t} onClose={() => select({ threed: null })} />}
    </aside>
  );
}

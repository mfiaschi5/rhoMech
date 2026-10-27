// Tier 3 of the depiction (chem/depict.ts of @ksmech/viz): the server's own SVG and exact
// coordinates of a pool state, GET /pool/{ref}/depiction?server_svg=1 (hidden H kept by map).
import { getDepiction } from "../api/pool";
import type { DepictionOut } from "../api/types.gen";

const STATE_REF = /^s[0-9a-f]{20}$/;

/** The tier-3 fetcher of a depiction whose ref is a pool state id (undefined otherwise). */
export function serverDepictionFetcher(dep: DepictionOut | null): (() => Promise<DepictionOut>) | undefined {
  if (!dep || !STATE_REF.test(dep.ref)) return undefined;
  const keepMaps = dep.atoms.filter((a) => a.element === "H").map((a) => a.map);
  return () => getDepiction(dep.ref, { keepMaps, serverSvg: true });
}

// MolDepiction of @ksmech/viz with the app's tier-3 fetcher: a pool state the client cannot fit
// is drawn from the server's own SVG (GET /pool/{state}/depiction?server_svg=1). Props and DOM:
// packages/viz/src/components/MolDepiction.tsx.
import { MolDepiction as VizMolDepiction, type MolDepictionProps } from "@ksmech/viz/components/MolDepiction";
import { serverDepictionFetcher } from "../chem/serverDepiction";

export function MolDepiction(props: MolDepictionProps) {
  return <VizMolDepiction {...props} fetchServer={props.fetchServer ?? serverDepictionFetcher(props.depiction)} />;
}

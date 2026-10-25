// Predict-local React hooks: pool depictions keyed by the map numbers they must keep, and the
// pool tokens. The generic hooks (usePendingCached, useDebounced, useLatest) are in api/hooks.ts.
import { getDepiction, getTokens } from "../../api/client";
import { useCached, type CachedState } from "../../api/cache";
import { usePendingCached, type PendingCached } from "../../api/hooks";
import type { DepictionOut, TokensOut } from "../../api/types.gen";

/** The depiction of a pool state that keeps the explicit H atoms among `maps`. */
export function usePoolDepiction(stateId: string | null, maps: readonly number[] = []): CachedState<DepictionOut> {
  const keep = [...new Set(maps)].sort((a, b) => a - b);
  const key = stateId ? `/pool/${stateId}/depiction?keep_maps=${keep.join(",")}` : null;
  return useCached<DepictionOut>(key, () => getDepiction(stateId ?? "", { keepMaps: keep }));
}

/** The tokens of a pool state (pending while the featurization runs). */
export function usePoolTokens(stateId: string | null): PendingCached<TokensOut> {
  const key = stateId ? `/pool/${stateId}/tokens` : null;
  return usePendingCached<TokensOut>(key, (o) => getTokens(stateId ?? "", o));
}

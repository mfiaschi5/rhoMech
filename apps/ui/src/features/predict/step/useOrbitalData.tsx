// The 3D data of the orbital step view (StepOrbital's OrbitalData) from the API: the pool's
// geometry (GET /pool/{id}/geometry, with its pending note and error), its depiction and the
// orbital cubes (GET /pool/{id}/orbital/{token}), cached by URL like every other read.
import { useCallback } from "react";
import { cached } from "../../../api/cache";
import { usePendingCached } from "../../../api/hooks";
import { getGeometry, getOrbitalCube } from "../../../api/pool";
import { ErrorBanner } from "../../../components/ErrorBanner";
import { PendingNote } from "../../../components/PendingNote";
import { usePool } from "../hooks";
import { StepOrbital, type OrbitalData } from "@ksmech/viz/step/StepOrbital";
import type { StepVariantProps } from "@ksmech/viz/step/StepTextbook";

export function useOrbitalData(stateId: string): OrbitalData {
  const geometry = usePendingCached(`/pool/${stateId}/geometry`, (o) => getGeometry(stateId, o));
  const pool = usePool(stateId);
  const loadCube = useCallback((token: number) => cached(`/pool/${stateId}/orbital/${token}`, () => getOrbitalCube(stateId, token, {})), [stateId]);
  const pending = geometry.pending && (
    <div className="ss-viewer-note">
      <PendingNote label={geometry.pending.label} worker={geometry.pending.worker} />
    </div>
  );
  return { key: stateId, geometry: geometry.data, depiction: pool.depiction, loadCube, pending, error: <ErrorBanner error={geometry.error} title="Cannot load the geometry" /> };
}

/** StepOrbital on a pool state of the server. */
export function LiveStepOrbital(props: StepVariantProps & { stateId: string }) {
  const { stateId, ...rest } = props;
  return <StepOrbital {...rest} orbital={useOrbitalData(stateId)} />;
}

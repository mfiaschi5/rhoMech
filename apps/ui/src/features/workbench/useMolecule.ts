// The compute flow of the workbench: POST /molecule with the sketch's molblock, then poll
// GET /molecule/{id} while the featurization runs. The POST answer is the only one with the
// sketch <-> record mapping, so it is kept with the drawing it describes (`model`, `molblock`):
// the page compares that drawing with the current sketch to decide whether links are live.
import { useCallback, useEffect, useRef, useState } from "react";
import { createMolecule, getMolecule, isAbort, toApiError, type ApiError } from "../../api/client";
import type { MoleculeOut, SketchMappingOut } from "../../api/types.gen";
import type { SketchModel } from "./sketch";

export type ComputeStatus = "idle" | "posting" | MoleculeOut["status"] | "error";

export interface ComputeState {
  status: ComputeStatus;
  molecule: MoleculeOut | null;
  mapping: SketchMappingOut | null;
  /** The drawing that was posted (null when the molecule came from a bare id). */
  model: SketchModel | null;
  molblock: string | null;
  error: ApiError | null;
}

export const IDLE: ComputeState = { status: "idle", molecule: null, mapping: null, model: null, molblock: null, error: null };

const POLL_MS = 1500;

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("aborted", "AbortError"));
    });
  });
}

/** GET the molecule until it leaves "running", reporting every state. */
async function pollUntilSettled(id: string, signal: AbortSignal, report: (m: MoleculeOut) => void): Promise<void> {
  for (;;) {
    await sleep(POLL_MS, signal);
    const m = await getMolecule(id, { signal });
    report(m);
    if (m.status !== "running") return;
  }
}

export interface MoleculeCompute {
  state: ComputeState;
  /** POST this drawing (retry: run a recorded featurize failure again). */
  compute(model: SketchModel, molblock: string, retry?: boolean): void;
  /** Load a molecule by id only (a deep link without the drawing): no mapping. */
  load(id: string): void;
  reset(): void;
}

export function useMolecule(): MoleculeCompute {
  const [state, setState] = useState<ComputeState>(IDLE);
  const ctl = useRef<AbortController | null>(null);
  useEffect(() => () => ctl.current?.abort(), []);

  const run = useCallback((start: Partial<ComputeState>, first: (signal: AbortSignal) => Promise<MoleculeOut>) => {
    ctl.current?.abort();
    const c = new AbortController();
    ctl.current = c;
    setState({ ...IDLE, ...start, status: "posting" });
    const report = (m: MoleculeOut) => {
      if (!c.signal.aborted) setState((s) => ({ ...s, molecule: m, mapping: m.mapping ?? s.mapping, status: m.status, error: null }));
    };
    (async () => {
      const m = await first(c.signal);
      report(m);
      if (m.status === "running") await pollUntilSettled(m.molecule_id, c.signal, report);
    })().catch((err: unknown) => {
      if (!isAbort(err) && !c.signal.aborted) setState((s) => ({ ...s, status: "error", error: toApiError(err) }));
    });
  }, []);

  const compute = useCallback(
    (model: SketchModel, molblock: string, retry = false) =>
      run({ model, molblock }, (signal) => createMolecule({ molblock, retry }, { signal })),
    [run],
  );
  const load = useCallback((id: string) => run({}, (signal) => getMolecule(id, { signal })), [run]);
  const reset = useCallback(() => {
    ctl.current?.abort();
    setState(IDLE);
  }, []);
  return { state, compute, load, reset };
}

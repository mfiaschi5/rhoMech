// RDKit.js (design §11.4). The wasm build is loaded by the classic script
// /vendor/RDKit_minimal.js (index.html); loadRDKit() is a memoized promise of
// window.initRDKitModule({locateFile: f => "/vendor/" + f}). When it resolves it sets
// document.body.dataset.rdkit = "ready" ("error" on failure), which the e2e test waits for.
// JSMol objects live in wasm memory: always release them (withMol does).
import { useEffect, useState } from "react";
import type { MainModule, Mol } from "@rdkit/rdkit";

export type RDKitModule = MainModule;
export type RDKitMol = Mol;

let loading: Promise<RDKitModule> | null = null;
let loaded: RDKitModule | null = null;

export function loadRDKit(): Promise<RDKitModule> {
  if (loading) return loading;
  loading = (async () => {
    const init = window.initRDKitModule;
    if (typeof init !== "function") {
      throw new Error(
        "RDKit.js is not loaded (/vendor/RDKit_minimal.js missing): run `npm --prefix apps/ui run vendor`",
      );
    }
    const mod = await init({ locateFile: (file: string) => `/vendor/${file}` });
    try {
      mod.prefer_coordgen(true);
    } catch {
      // older builds without the switch: RDKit's default depiction
    }
    loaded = mod;
    document.body.dataset.rdkit = "ready";
    return mod;
  })();
  loading.catch((err: unknown) => {
    document.body.dataset.rdkit = "error";
    console.error("RDKit.js failed to load", err);
    loading = null; // a later call retries
  });
  return loading;
}

/** The module once loaded, else null (no load is started). */
export function getRDKit(): RDKitModule | null {
  return loaded;
}

/** React hook: the RDKit module (null while loading) and the load error, if any. */
export function useRDKit(): { rdkit: RDKitModule | null; error: Error | null } {
  const [state, setState] = useState<{ rdkit: RDKitModule | null; error: Error | null }>({
    rdkit: loaded,
    error: null,
  });
  useEffect(() => {
    if (loaded) return;
    let live = true;
    loadRDKit().then(
      (rdkit) => live && setState({ rdkit, error: null }),
      (error: unknown) =>
        live && setState({ rdkit: null, error: error instanceof Error ? error : new Error(String(error)) }),
    );
    return () => {
      live = false;
    };
  }, []);
  return state;
}

/** Parse `input` (SMILES or molblock), run `fn`, release the mol. null when RDKit rejects it. */
export function withMol<T>(
  rdkit: RDKitModule,
  input: string,
  details: Record<string, unknown> | null,
  fn: (mol: RDKitMol) => T,
): T | null {
  let mol: RDKitMol | null = null;
  try {
    mol = details ? rdkit.get_mol(input, JSON.stringify(details)) : rdkit.get_mol(input);
  } catch {
    mol = null;
  }
  if (!mol) return null;
  try {
    if (!mol.is_valid()) return null;
    return fn(mol);
  } finally {
    mol.delete();
  }
}

export interface SvgOptions {
  width: number;
  height: number;
  /** Extra RDKit drawing options (MolDrawOptions JSON keys). */
  draw?: Record<string, unknown>;
}

/** The RDKit.js SVG of a SMILES, or null when it does not parse. */
export function smilesToSvg(rdkit: RDKitModule, smiles: string, opts: SvgOptions): string | null {
  if (!smiles.trim()) return null;
  return withMol(rdkit, smiles, null, (mol) =>
    mol.get_svg_with_highlights(
      JSON.stringify({ width: opts.width, height: opts.height, bondLineWidth: 1.5, ...opts.draw }),
    ),
  );
}

/** An SVG string as a data URI (cytoscape node background-image). */
export function svgDataUri(svg: string): string {
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

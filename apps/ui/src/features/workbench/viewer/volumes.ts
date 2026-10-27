// Cube volumes of the surface layers: fetched once per (molecule, kind, set, start, index,
// quality), parsed into 3Dmol VolumeData and kept in an LRU of 8 (a cube is megabytes of text,
// its VolumeData a Float32Array of the grid). Isovalue, opacity, style and colours are applied
// on the client and never refetch. Each load takes an AbortSignal, so a layer that changes or
// goes away cancels its request (and the 202 pending retry loop).
import type { VolumeData } from "3dmol";
import { type RequestOptions } from "../../../api/http";
import { getMoleculeCube, type CubeMeta, type CubeQuality } from "../../../api/workbench";
import { parseCube, threeDmol } from "../../../chem/threed";
import { normalizeSpec, specKey, type LayerSpec } from "../params";

export const VOLUME_LRU_SIZE = 8;

export interface LoadedVolume {
  vol: VolumeData | null;
  meta: CubeMeta;
  /** Why `vol` is null (3Dmol.js missing): the text was fetched but cannot be drawn. */
  parseError: string | null;
}

export class LRU<K, V> {
  private readonly map = new Map<K, V>();
  constructor(private readonly size: number) {}
  get(key: K): V | undefined {
    const v = this.map.get(key);
    if (v !== undefined) {
      this.map.delete(key);
      this.map.set(key, v);
    }
    return v;
  }
  set(key: K, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.size) {
      const oldest = this.map.keys().next().value as K;
      this.map.delete(oldest);
    }
  }
  get length(): number {
    return this.map.size;
  }
}

const volumes = new LRU<string, LoadedVolume>(VOLUME_LRU_SIZE);

export function volumeKey(mid: string, spec: LayerSpec, quality: CubeQuality): string {
  return `${mid}|${specKey(spec)}|${quality}`;
}

/** The cached volume of a cube, or undefined (no load). */
export function peekVolume(mid: string, spec: LayerSpec, quality: CubeQuality): LoadedVolume | undefined {
  return volumes.get(volumeKey(mid, spec, quality));
}

function parse(text: string): { vol: VolumeData | null; parseError: string | null } {
  const lib = threeDmol();
  if (!lib) return { vol: null, parseError: "3Dmol.js is not loaded (/vendor/3Dmol-min.js)" };
  try {
    return { vol: parseCube(lib, text), parseError: null };
  } catch (err) {
    return { vol: null, parseError: `cannot parse the cube: ${String(err)}` };
  }
}

/** Fetch (or reuse) one cube as a volume. */
export async function loadVolume(mid: string, spec: LayerSpec, quality: CubeQuality, o: RequestOptions): Promise<LoadedVolume> {
  const key = volumeKey(mid, spec, quality);
  const hit = volumes.get(key);
  if (hit) return hit;
  const n = normalizeSpec(spec);
  const res = await getMoleculeCube(mid, { kind: n.kind, set: n.set, start: n.start, index: n.index, quality }, o);
  const loaded: LoadedVolume = { ...parse(res.text), meta: res.meta };
  if (loaded.vol) volumes.set(key, loaded);
  return loaded;
}

/** A default ESP colour range: the 95th percentile of |ESP| on the grid points near the density
 *  isosurface (both cubes share the whole-molecule grid of one quality); 0.05 a.u. when the
 *  grids differ or nothing is near the surface. */
export function espRange(density: VolumeData, esp: VolumeData, iso: number): number {
  const rho = density.data;
  const phi = esp.data;
  if (!rho || !phi || rho.length !== phi.length) return 0.05;
  const near: number[] = [];
  const lo = iso * 0.7;
  const hi = iso * 1.4;
  for (let i = 0; i < rho.length; i++) {
    const r = rho[i];
    if (r >= lo && r <= hi) near.push(Math.abs(phi[i]));
  }
  if (near.length < 10) return 0.05;
  near.sort((a, b) => a - b);
  const p = near[Math.floor(0.95 * (near.length - 1))];
  return p > 0 ? Number(p.toPrecision(2)) : 0.05;
}

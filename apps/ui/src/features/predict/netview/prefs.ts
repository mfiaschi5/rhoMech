// The network viewport's preferences, kept in localStorage: the variant (Scheme or Pathways) and
// the height the splitter was dragged to.
import { useState } from "react";
import { VARIANTS, type Variant } from "@ksmech/viz/network/shared";

const KEY = "ksmech.predict.networkVariant";
const HEIGHT_KEY = "ksmech.predict.networkHeight";
export const MIN_NET_H = 320;
export const MAX_NET_H = 1200;

/** The network viewport's height (px): 60 % of the window at first, then what the user dragged
 *  the splitter to (localStorage). */
export function useNetHeight(): [number, (h: number) => void] {
  const [h, setH] = useState<number>(() => {
    let v = Number.NaN;
    try {
      v = Number(localStorage.getItem(HEIGHT_KEY));
    } catch {
      // no storage: the default
    }
    return Number.isFinite(v) && v >= MIN_NET_H ? Math.min(MAX_NET_H, v) : Math.max(MIN_NET_H, Math.round(window.innerHeight * 0.6));
  });
  const set = (next: number) => {
    const c = Math.round(Math.max(MIN_NET_H, Math.min(MAX_NET_H, next)));
    setH(c);
    try {
      localStorage.setItem(HEIGHT_KEY, String(c));
    } catch {
      // private window
    }
  };
  return [h, set];
}

export function useVariant(): [Variant, (v: Variant) => void] {
  const [v, setV] = useState<Variant>(() => {
    try {
      const s = localStorage.getItem(KEY);
      return VARIANTS.some((x) => x.id === s) ? (s as Variant) : "layered";
    } catch {
      return "layered";
    }
  });
  const set = (next: Variant) => {
    setV(next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      // private window: the choice lasts for this page only
    }
  };
  return [v, set];
}

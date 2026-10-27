// The scene URL of #/scene?src=: resolved against the page, kept only on the page's origin.

/** `src` as an absolute URL on the origin of `page`, or null (another origin, not a URL). */
export function sameOriginUrl(src: string, page: string): string | null {
  try {
    const url = new URL(src, page);
    return url.origin === new URL(page).origin ? url.href : null;
  } catch {
    return null;
  }
}

// Advance widths of Helvetica (identical to Arial's), in 1/1000 em, from Adobe's Helvetica AFM
// (the PDF base-14 font metrics), for laying out atom labels without a DOM: the canvas, the SVG
// export and the PDF writer all place text by these numbers, so labels sit where a PDF viewer
// draws them. Bold widths are close enough for label placement (within a few %).

const ASCII_WIDTHS: readonly number[] = [
  // 32..126: space ! " # $ % & ' ( ) * + , - . / 0-9 : ; < = > ? @ A-Z [ \ ] ^ _ ` a-z { | } ~
  278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
  556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
  1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
  667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
  333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
  556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
];

/** Width of one character (em). The minus sign U+2212 is drawn as an en dash (556). */
export function charWidthEm(ch: string): number {
  const c = ch.codePointAt(0) ?? 32;
  if (c >= 32 && c <= 126) return ASCII_WIDTHS[c - 32] / 1000;
  if (c === 0x2212 || c === 0x2013) return 0.556;
  return 0.6;
}

/** Width of a string in em. */
export const textWidthEm = (text: string): number => [...text].reduce((s, ch) => s + charWidthEm(ch), 0);

/** Cap height and the em-box centre above the baseline (em), Helvetica. */
export const CAP_HEIGHT_EM = 0.718;
/** Baseline offset (em) of text "centred" on a point the way SVG dominant-baseline=central puts
 *  a capital: the cap middle on the point. */
export const CENTRAL_BASELINE_EM = CAP_HEIGHT_EM / 2;

// The R editor's keyboard: b / l / a / e pick the tool, Ctrl+Z undoes, Ctrl+Y and Ctrl+Shift+Z
// redo, Delete / Backspace removes the selected entries, Enter applies. Keys typed into a text
// field, a select or a textarea are left alone (a checkbox is not one), and so is anything while
// a dialog is open.
import { useEffect, type Dispatch } from "react";
import { keyAction, type EditorAction } from "./editor";

const TEXT_INPUTS = /^(text|number|search|email|url|tel|password|)$/;

/** A text field, select or textarea has the key (a checkbox or a button does not). */
function typing(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el || !el.tagName) return false;
  if (el.tagName === "INPUT") return TEXT_INPUTS.test((el as HTMLInputElement).getAttribute("type") ?? "");
  return /^(SELECT|TEXTAREA)$/.test(el.tagName) || el.isContentEditable;
}

export function useEditorKeys(dispatch: Dispatch<EditorAction>, onApply: () => void): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e.target) || document.querySelector("[role=dialog]")) return;
      const a = keyAction(e);
      if (!a) return;
      // Enter on a focused button or link is that control's own click
      if (a === "apply" && /^(BUTTON|A)$/.test((e.target as HTMLElement | null)?.tagName ?? "")) return;
      e.preventDefault();
      if (a === "apply") onApply();
      else dispatch(a);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dispatch, onApply]);
}

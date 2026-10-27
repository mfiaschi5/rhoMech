// The molecule workbench feature: registers the "workbench" page (WorkbenchPage.tsx) and, at
// load (main.tsx imports every feature eagerly), applies the stored theme preference, so every
// page starts in the user's light / dark / auto choice without a flash.
import { registerPage } from "../../registry";
import { WorkbenchPage } from "./WorkbenchPage";
import { applyStoredTheme } from "./layout/useWorkbenchLayout";

registerPage("workbench", WorkbenchPage);
applyStoredTheme();

// The molecule workbench feature: registers the "workbench" page (WorkbenchPage.tsx).
import { registerPage } from "../../registry";
import { WorkbenchPage } from "./WorkbenchPage";

registerPage("workbench", WorkbenchPage);

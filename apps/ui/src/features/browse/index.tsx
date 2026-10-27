// Feature seam (design §11.1): registers the "browse" page. Imports no other feature.
import { registerPage } from "../../registry";
import { BrowsePage } from "./BrowsePage";
import "./browse.css";

registerPage("browse", BrowsePage);

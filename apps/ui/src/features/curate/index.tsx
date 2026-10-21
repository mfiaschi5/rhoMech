// Feature seam (design §11.1): registers the "curate" page. Imports no other feature.
import { registerPage } from "../../registry";
import { CuratePage } from "./CuratePage";
import "./curate.css";

registerPage("curate", CuratePage);

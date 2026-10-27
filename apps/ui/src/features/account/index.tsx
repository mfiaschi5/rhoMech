// Feature seam (design §11.1): registers the "login" (sign-in and the account) and "users" (the
// admins' user management) pages. Imports no other feature; the header's UserMenu lives in
// src/components.
import { registerPage } from "../../registry";
import { LoginPage } from "./LoginPage";
import { UsersPage } from "./UsersPage";
import "./account.css";

registerPage("login", LoginPage);
registerPage("users", UsersPage);

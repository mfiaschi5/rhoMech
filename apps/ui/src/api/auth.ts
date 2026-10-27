// The accounts router (serve/accounts): sign-in, sign-out, the signed-in user and its quota,
// and the user management of the admins. The session is an HttpOnly cookie the browser sends
// by itself (same-origin fetch), so no call here handles a token.
import { request, type RequestOptions } from "./http";
import type * as T from "./types.gen";

export const getAuthMe = (o?: RequestOptions) => request<T.AuthStatusOut>({ path: "/auth/me" }, o);
export const signIn = (body: T.LoginIn, o?: RequestOptions) =>
  request<T.AuthStatusOut>({ method: "POST", path: "/auth/login", body }, o);
export const signOut = (o?: RequestOptions) => request<T.AuthStatusOut>({ method: "POST", path: "/auth/logout" }, o);
export const changePassword = (body: T.PasswordChangeIn, o?: RequestOptions) =>
  request<T.UserOut>({ method: "POST", path: "/auth/password", body }, o);

/** The provider's start URL, landing back on `returnTo` (a "#/..." route) after sign-in. */
export function providerStartUrl(provider: T.ProviderOut, returnTo: string): string {
  return `${provider.start_url}?${new URLSearchParams({ return_to: returnTo }).toString()}`;
}

export const listUsers = (o?: RequestOptions) => request<T.UsersOut>({ path: "/admin/users" }, o);
export const createUser = (body: T.UserCreateIn, o?: RequestOptions) =>
  request<T.UserOut>({ method: "POST", path: "/admin/users", body }, o);
export const updateUser = (id: number, body: T.UserPatchIn, o?: RequestOptions) =>
  request<T.UserOut>({ method: "PATCH", path: `/admin/users/${id}`, body }, o);

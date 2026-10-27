// The app router (serve/app): health and the server's meta (version, capabilities, limits).
import { request, type RequestOptions } from "./http";
import type * as T from "./types.gen";

export const getHealth = (o?: RequestOptions) => request<T.HealthOut>({ path: "/health" }, o);
export const getMeta = (o?: RequestOptions) => request<T.MetaOut>({ path: "/meta" }, o);

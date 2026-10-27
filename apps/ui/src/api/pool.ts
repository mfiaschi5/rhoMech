// The pool router (serve/pool): normalized input, the pool of a set of reactants, its tokens,
// geometry, orbital cubes and depiction.
import { request, type RequestOptions } from "./http";
import type * as T from "./types.gen";

export const normalize = (body: T.NormalizeIn, o?: RequestOptions) =>
  request<T.NormalizeOut>({ method: "POST", path: "/normalize", body }, o);

const pool = (stateId: string, rest = "") => `/pool/${encodeURIComponent(stateId)}${rest}`;

/** POST /pools: the pool of these reactants, once its analysis is ready (pending meanwhile). */
export const createPool = (body: T.NormalizeIn, o?: RequestOptions) =>
  request<T.PoolOut>({ method: "POST", path: "/pools", body }, o);

export const getPool = (stateId: string, o?: RequestOptions) =>
  request<T.PoolOut>({ path: pool(stateId) }, o);

export const getTokens = (stateId: string, o?: RequestOptions) =>
  request<T.TokensOut>({ path: pool(stateId, "/tokens") }, o);

// ---- geometry, depiction, 3D

export const getGeometry = (stateId: string, o?: RequestOptions) =>
  request<T.GeometryOut>({ path: pool(stateId, "/geometry") }, o);

export interface CubeQuery {
  spacing?: number;
  padding?: number;
}

export const orbitalCubePath = (stateId: string, tokenId: number) => pool(stateId, `/orbital/${tokenId}`);

/** The Gaussian cube text of one localized orbital (pending while the grid is computed). */
export const getOrbitalCube = (stateId: string, tokenId: number, q: CubeQuery = {}, o?: RequestOptions) =>
  request<string>({ path: orbitalCubePath(stateId, tokenId), query: { ...q }, as: "text" }, o);

export interface DepictionQuery {
  keepMaps?: readonly number[];
  serverSvg?: boolean;
}

export const getDepiction = (stateId: string, q: DepictionQuery = {}, o?: RequestOptions) =>
  request<T.DepictionOut>(
    {
      path: pool(stateId, "/depiction"),
      query: {
        keep_maps: q.keepMaps && q.keepMaps.length ? q.keepMaps.join(",") : undefined,
        server_svg: q.serverSvg ? 1 : undefined,
      },
    },
    o,
  );

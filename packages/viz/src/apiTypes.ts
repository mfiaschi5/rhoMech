// The HTTP schemas of ksmech-serve (pydantic, generated into apps/ui/src/api/types.gen.ts by
// `ksmech-serve schema-ts`), re-exported as types only. This is the package's one link to the
// app and it is erased at runtime (vizBoundary.test.ts: nothing else under packages/viz may
// import apps/ui).
export type * from "../../../apps/ui/src/api/types.gen";

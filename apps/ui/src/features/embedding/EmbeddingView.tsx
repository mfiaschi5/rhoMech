// EmbeddingView: one pool through a trained spec-v9 checkpoint (ksmech-eval embedding, run by
// the server's compute worker), its trace page embedded in an iframe.
//
// It lists the local checkpoints (GET /api/models; the choice is remembered in localStorage),
// starts the trace job for the target as soon as a checkpoint is chosen (POST /api/embedding:
// the server answers a stored trace at once), shows the job through JobProgress, then the page
// (GET /api/embedding/{id}/page) in an iframe that fills the view, in the app's theme when the
// app overrides the browser's. The page keeps its view (selected token, colouring, layer, head)
// in its URL hash; the view reads it from the iframe once a second, stores it per trace in
// sessionStorage (the trace reopens where it was left) and puts it on "open in new tab".
// Unavailable (the embedding capability is off): greyed, with the server's reason.
//
// Props: target {stateId} | {smiles}, testId (default "embedding-view"), title?.
// DOM: <section data-testid=<testId> data-status=idle|unavailable|running|ready|failed
// data-embedding-id>, <testId>-checkpoint (select), <testId>-rerun, <testId>-open (a link),
// <testId>-facts, <testId>-frame (the iframe, data-hash: the page's view), JobProgress inside
// (predict-job-*), the capability notice (capability-notice[data-capability=embedding]).
import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { toApiError, type ApiError } from "../../api/http";
import { useCached } from "../../api/cache";
import { useJobPoll } from "../../api/poll";
import type { CheckpointOut, EmbeddingResultOut, JobOut } from "../../api/types.gen";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import { fmtSig } from "@ksmech/viz/format";
import { useCapability } from "../../state/store";
import { JobProgress } from "../../components/JobProgress";
import { getModels } from "../../api/predict";
import { createEmbedding } from "../../api/embedding";
import { CHECKPOINT_KEY, checkpointKey, checkpointLabel, cleanViewHash, defaultCheckpoint, parseCheckpointKey, readStored, themeOf, tracePageUrl, viewKey, writeStored, type PageTheme } from "./embedding";
import "./embedding.css";

export interface EmbeddingTarget {
  stateId?: string | null;
  smiles?: string | null;
}

export interface EmbeddingViewProps {
  target: EmbeddingTarget;
  testId?: string;
}

const HASH_POLL_MS = 1000;
const local = () => (typeof window === "undefined" ? null : safe(() => window.localStorage));
const session = () => (typeof window === "undefined" ? null : safe(() => window.sessionStorage));

function safe<T>(get: () => T): T | null {
  try {
    return get();
  } catch {
    return null;
  }
}

/** The app's theme override (<html data-theme>), followed with a MutationObserver. */
function useDocTheme(): PageTheme | null {
  const read = () => themeOf(document.documentElement.dataset.theme);
  const [theme, setTheme] = useState<PageTheme | null>(read);
  useEffect(() => {
    const mo = new MutationObserver(() => setTheme(read()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => mo.disconnect();
  }, []);
  return theme;
}

/** The trace job of (target, checkpoint key): started whenever either changes or on rerun. */
function useTraceJob(target: EmbeddingTarget, key: string | null, on: boolean) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [round, setRound] = useState(0);
  const targetKey = target.stateId ? `s:${target.stateId}` : target.smiles ? `m:${target.smiles}` : "";
  useEffect(() => {
    const ref = parseCheckpointKey(key);
    setJobId(null);
    setError(null);
    if (!on || !ref || !targetKey) return;
    const ctl = new AbortController();
    const body = { ...ref, state_id: target.stateId ?? null, smiles: target.stateId ? null : (target.smiles ?? null) };
    createEmbedding(body, { signal: ctl.signal }).then(
      (created) => setJobId(created.job_id),
      (err) => {
        if (!ctl.signal.aborted) setError(toApiError(err));
      },
    );
    return () => ctl.abort();
  }, [targetKey, key, on, round]);
  const poll = useJobPoll(jobId, { intervalMs: 700 });
  return { jobId, poll, error, rerun: () => setRound((r) => r + 1) };
}

function resultOf(job: JobOut | null): EmbeddingResultOut | null {
  if (!job || job.status !== "done" || job.result_kind !== "embedding" || !job.result) return null;
  return job.result as EmbeddingResultOut;
}

/** The page's view hash, read from the (same-origin) iframe and stored per trace. */
function useFrameHash(frame: RefObject<HTMLIFrameElement | null>, eid: string | null): string {
  const [hash, setHash] = useState("");
  useEffect(() => {
    setHash(eid ? cleanViewHash(readStored(session(), viewKey(eid))) : "");
    if (!eid) return;
    const t = setInterval(() => {
      const h = cleanViewHash(safe(() => frame.current?.contentWindow?.location.hash ?? ""));
      if (!h) return;
      setHash((old) => (old === h ? old : h));
      writeStored(session(), viewKey(eid), h);
    }, HASH_POLL_MS);
    return () => clearInterval(t);
  }, [eid, frame]);
  return hash;
}

function Picker({ list, value, onChange, testId }: { list: readonly CheckpointOut[]; value: string | null; onChange: (k: string) => void; testId: string }) {
  return (
    <label className="row emb-picker">
      <span className="muted">checkpoint</span>
      <select data-testid={`${testId}-checkpoint`} value={value ?? ""} onChange={(e) => onChange(e.target.value)} disabled={!list.length}>
        {!list.length && <option value="">no local checkpoint</option>}
        {list.map((c) => (
          <option key={checkpointKey(c)} value={checkpointKey(c)} disabled={!c.usable} title={c.reason ?? c.path}>
            {checkpointLabel(c)}
          </option>
        ))}
      </select>
    </label>
  );
}

function Facts({ r, testId }: { r: EmbeddingResultOut; testId: string }) {
  return (
    <span className="muted emb-facts" data-testid={`${testId}-facts`} data-cached={r.cached ? "1" : "0"}>
      {r.encoder} encoder · {r.n_layers} layers × {r.heads} heads · {r.n_atoms} atoms, {r.n_orbitals} orbitals · rebuilt within {fmtSig(r.max_error, 2)}
      {r.cached ? " · stored trace" : ` · traced in ${r.seconds.toFixed(1)} s`}
    </span>
  );
}

function statusOf(on: boolean, job: JobOut | null, error: ApiError | null, result: EmbeddingResultOut | null) {
  if (!on) return "unavailable";
  if (result) return "ready";
  if (error || job?.status === "failed" || job?.status === "cancelled") return "failed";
  return job ? "running" : "idle";
}

export function EmbeddingView({ target, testId = "embedding-view" }: EmbeddingViewProps) {
  const cap = useCapability("embedding");
  const models = useCached(cap.enabled ? "/models#embedding" : null, () => getModels());
  const list = models.data?.checkpoints ?? [];
  const on = cap.enabled && (models.data ? models.data.embedding.enabled : true);
  const [choice, setChoice] = useState<string | null>(null);
  const picked = useMemo(() => (choice ? (list.find((c) => checkpointKey(c) === choice) ?? null) : defaultCheckpoint(list, readStored(local(), CHECKPOINT_KEY))), [list, choice]);
  const key = picked ? checkpointKey(picked) : null;
  const { poll, error, rerun } = useTraceJob(target, key, on && !!models.data);
  const result = resultOf(poll.job);
  const frame = useRef<HTMLIFrameElement | null>(null);
  const hash = useFrameHash(frame, result?.embedding_id ?? null);
  const theme = useDocTheme();
  // the src is fixed per trace and theme (the hash at that moment): the page moves its own hash
  const src = useMemo(() => (result ? tracePageUrl(result.page_url, theme, readStored(session(), viewKey(result.embedding_id))) : null), [result?.embedding_id, theme]);
  const choose = (k: string) => {
    setChoice(k);
    writeStored(local(), CHECKPOINT_KEY, k);
  };
  const status = statusOf(on, poll.job, error ?? models.error ?? null, result);
  return (
    <section className={`emb-view${on ? "" : " emb-off"}`} data-testid={testId} data-status={status} data-embedding-id={result?.embedding_id ?? ""}>
      <CapabilityNotice name="embedding" title="Embedding trace unavailable" compact />
      {cap.enabled && models.data && !models.data.embedding.enabled && (
        <div className="capability-notice compact muted" role="note" data-testid={`${testId}-reason`}>
          <span className="badge badge-warn">disabled</span> {models.data.embedding.reason}
        </div>
      )}
      <div className="row emb-bar">
        <Picker list={list} value={key} onChange={choose} testId={testId} />
        <button type="button" className="btn btn-small" data-testid={`${testId}-rerun`} disabled={!on || !key} onClick={rerun} title="ask the server again (a stored trace answers at once)">
          trace
        </button>
        {result && src && (
          <a className="btn btn-small" data-testid={`${testId}-open`} href={tracePageUrl(result.page_url, theme, hash)} target="_blank" rel="noopener" title="the self-contained trace page">
            open in new tab ↗
          </a>
        )}
        {result && <Facts r={result} testId={testId} />}
      </div>
      <ErrorBanner error={error ?? models.error ?? null} title="Cannot trace this pool" testId={`${testId}-error`} />
      {poll.job && !result && <JobProgress job={poll.job} pollError={poll.error} title="Embedding trace" />}
      {result && src && <iframe ref={frame} className="emb-frame" data-testid={`${testId}-frame`} data-hash={hash} src={src} title={`embedding trace ${result.run}/${result.file}`} />}
      {on && !result && !poll.job && !error && <div className="muted emb-placeholder">{models.data ? (key ? "starting the trace…" : "no usable checkpoint") : "listing the checkpoints…"}</div>}
    </section>
  );
}

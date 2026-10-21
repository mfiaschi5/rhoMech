// ErrorBanner: an error shown in place, with its reason code (design §6.3, §11.9).
//
// Props:
//   error      ApiError | Error | string | null | undefined   nothing is rendered for null
//   title      string           optional heading (default "Error")
//   onDismiss  () => void       shows a dismiss button when given
//   testId     string           default "error-banner"
//
// DOM: <div role="alert" data-testid="error-banner" data-code="<code>">: code, message, the
// upstream reason code and request id when present, and the detail JSON in a <details>.
import { ApiError } from "../api/client";

export interface ErrorBannerProps {
  error: ApiError | Error | string | null | undefined;
  title?: string;
  onDismiss?: () => void;
  testId?: string;
}

export function ErrorBanner({ error, title, onDismiss, testId = "error-banner" }: ErrorBannerProps) {
  if (!error) return null;
  const api = error instanceof ApiError ? error : null;
  const code = api?.code ?? "CLIENT_ERROR";
  const message = typeof error === "string" ? error : error.message;
  return (
    <div role="alert" className="banner banner-error" data-testid={testId} data-code={code}>
      <div className="banner-head">
        <strong>{title ?? "Error"}</strong> <code className="code">{code}</code>
        {api && api.status > 0 && <span className="muted"> HTTP {api.status}</span>}
        {onDismiss && (
          <button type="button" className="btn btn-small banner-dismiss" onClick={onDismiss} aria-label="dismiss">
            ×
          </button>
        )}
      </div>
      <div className="banner-message">{message}</div>
      {api?.upstream && (
        <div className="muted">
          upstream: <code>{api.upstream}</code>
        </div>
      )}
      {api?.requestId && <div className="muted">request id: {api.requestId}</div>}
      {api?.detail && (
        <details>
          <summary>detail</summary>
          <pre className="pre">{JSON.stringify(api.detail, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}

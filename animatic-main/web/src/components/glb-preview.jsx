import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Box, Loader2 } from "lucide-react";

const STATUS_LABELS = {
  loading: "Loading preview",
  ready: "Preview ready",
  error: "Preview unavailable",
};

const GlbPreview = ({
  src,
  filename = "generated.glb",
  helperText = "Use the preview to inspect the generated rigged GLB before downloading it again.",
  actions = null,
}) => {
  const viewerRef = useRef(null);
  const [viewerReady, setViewerReady] = useState(false);
  const [status, setStatus] = useState(src ? "loading" : "idle");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let cancelled = false;

    import("@google/model-viewer")
      .then(() => {
        if (!cancelled) {
          setViewerReady(true);
        }
      })
      .catch((error) => {
        console.error("Failed to load GLB preview renderer:", error);
        if (!cancelled) {
          setStatus("error");
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!src || !viewerRef.current || !viewerReady) {
      setStatus(src ? "loading" : "idle");
      setProgress(0);
      return;
    }

    const viewer = viewerRef.current;
    setStatus("loading");
    setProgress(0);

    const handleLoad = () => {
      setStatus("ready");
      setProgress(100);
    };
    const handleError = () => setStatus("error");
    const handleProgress = (event) => {
      const nextProgress = event.detail?.totalProgress;
      if (typeof nextProgress === "number") {
        setProgress(Math.round(nextProgress * 100));
      }
    };

    viewer.addEventListener("load", handleLoad);
    viewer.addEventListener("error", handleError);
    viewer.addEventListener("progress", handleProgress);

    return () => {
      viewer.removeEventListener("load", handleLoad);
      viewer.removeEventListener("error", handleError);
      viewer.removeEventListener("progress", handleProgress);
    };
  }, [src, viewerReady]);

  if (!src) return null;

  return (
    <section className="glass p-5 sm:p-6" aria-labelledby="glb-preview-title">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.1)]">
            <Box className="h-5 w-5 text-cyan-200" aria-hidden="true" />
          </div>
          <div>
            <h3
              id="glb-preview-title"
              className="text-lg sm:text-xl font-semibold text-white"
            >
              GLB Preview
            </h3>
            <p className="mt-1 text-sm text-white/60">{helperText}</p>
            <p className="mt-2 max-w-full truncate text-xs font-medium uppercase tracking-wide text-white/40">
              {filename}
            </p>
          </div>
        </div>

        {actions && <div className="shrink-0">{actions}</div>}
      </div>

      <div
        className="relative mt-5 h-[360px] overflow-hidden rounded-2xl border border-white/10 bg-neutral-950/80 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] sm:h-[440px]"
        aria-label={`Interactive preview of ${filename}`}
      >
        <model-viewer
          ref={viewerRef}
          src={src}
          alt={`Preview of ${filename}`}
          camera-controls
          auto-rotate
          interaction-prompt="auto"
          loading="eager"
          reveal="auto"
          shadow-intensity="0.75"
          exposure="1"
          environment-image="neutral"
          className="h-full w-full"
        />

        {status === "loading" && (
          <div
            className="absolute inset-0 flex flex-col items-center justify-center bg-neutral-950/70 p-6 text-center backdrop-blur-sm"
            role="status"
            aria-live="polite"
          >
            <Loader2
              className="h-8 w-8 animate-spin text-cyan-200"
              aria-hidden="true"
            />
            <p className="mt-3 text-sm font-medium text-white">
              {STATUS_LABELS.loading}
            </p>
            <p className="mt-1 text-xs text-white/55">
              {progress > 0 ? `${progress}% loaded` : "Preparing the model viewer."}
            </p>
          </div>
        )}

        {status === "error" && (
          <div
            className="absolute inset-0 flex flex-col items-center justify-center bg-neutral-950/80 p-6 text-center"
            role="alert"
          >
            <AlertTriangle
              className="h-8 w-8 text-amber-200"
              aria-hidden="true"
            />
            <p className="mt-3 text-sm font-medium text-white">
              {STATUS_LABELS.error}
            </p>
            <p className="mt-1 max-w-md text-xs text-white/60">
              The file can still be downloaded, but this browser could not render
              the preview.
            </p>
          </div>
        )}
      </div>

      <p className="mt-3 text-xs text-white/45">
        Drag to rotate, scroll or pinch to zoom. Preview quality depends on the
        generated GLB.
      </p>
    </section>
  );
};

export default GlbPreview;

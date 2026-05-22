import { useEffect, useRef, useState } from "react";
import { CheckCircle, Clock, Upload, User } from "lucide-react";
import { saveFile } from "../helper/db";
import getBackendURL from "../config";
import toast from "react-hot-toast";
import GlbPreview from "./glb-preview";
import {
  formatFileSize,
  MAX_UPLOAD_SIZE_BYTES,
  validateImageFile,
} from "../utils/file-validation";
import {
  getApiErrorMessage,
  isJsonContentType,
  validateGlbBlob,
} from "../utils/glb-response";

const GENERATION_TIMEOUT_MS = 120000; // Hugging Face Spaces can cold start, so keep this above typical browser timeouts.
const GENERATION_TIMEOUT_SECONDS = Math.round(GENERATION_TIMEOUT_MS / 1000);
const ACCEPTED_IMAGE_TYPES = "image/jpeg,image/png,image/webp";

const GENERATION_STATUS = {
  validating: {
    title: "Validating upload",
    detail: "Checking the selected image type and file size.",
  },
  uploading: {
    title: "Uploading image",
    detail: "Sending the character image to the hosted Flask API.",
  },
  waiting: {
    title: "Waiting for the ML backend",
    detail:
      "The backend is predicting 2D keypoints and building the GLB. Hugging Face Spaces may take longer after a cold start.",
  },
  preparing: {
    title: "Preparing download",
    detail: "Checking the returned file before saving it locally.",
  },
  saving: {
    title: "Saving to recent files",
    detail: "Storing the GLB in this browser before starting the download.",
  },
  ready: {
    title: "GLB ready",
    detail: "Saved to Recent Files and download started automatically.",
  },
  cancelling: {
    title: "Cancelling generation",
    detail: "Stopping the current request. You can retry with the same image.",
  },
};

const Home = () => {
  const heading = "Turn a 2D Character Image into a Rigged GLB";
  const subHeading =
    "Upload a front-facing character image. Animatic sends it to the ML backend, predicts 2D keypoints, builds a simple mesh and armature, then returns a GLB.";
  const uploadLimit = formatFileSize(MAX_UPLOAD_SIZE_BYTES);

  const [uploadedImage, setUploadedImage] = useState(null);
  const [file2d, setFile2d] = useState(null);

  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStatus, setGenerationStatus] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [isComplete, setIsComplete] = useState(false);
  const [generatedPreview, setGeneratedPreview] = useState(null);

  const fileInputRef = useRef(null);
  const activeRequestRef = useRef(null);
  const previewUrlRef = useRef(null);

  const revokeGeneratedPreviewUrl = () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
  };

  const clearGeneratedPreview = () => {
    revokeGeneratedPreviewUrl();
    setGeneratedPreview(null);
  };

  useEffect(() => revokeGeneratedPreviewUrl, []);

  const blobToBase64 = (blob) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });

  const handleImageUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const validation = validateImageFile(file);
    if (!validation.valid) {
      toast.error(validation.message);
      setFeedback({ type: "error", message: validation.message });
      event.target.value = "";
      setUploadedImage(null);
      setFile2d(null);
      setIsComplete(false);
      setGenerationStatus(null);
      return;
    }

    setFile2d(file);
    setFeedback({
      type: "success",
      message: `Selected ${file.name} (${formatFileSize(file.size)}).`,
    });

    const reader = new FileReader();
    reader.onload = (e) => setUploadedImage(e.target.result);
    reader.readAsDataURL(file);

    setIsComplete(false);
    setGenerationStatus(null);
  };

  const resetProcess = () => {
    if (activeRequestRef.current) {
      activeRequestRef.current.reason = "cancelled";
      activeRequestRef.current.controller.abort();
    }

    setUploadedImage(null);
    setFile2d(null);
    setIsGenerating(false);
    setGenerationStatus(null);
    setFeedback(null);
    setIsComplete(false);
    clearGeneratedPreview();
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const cancelGeneration = () => {
    if (!activeRequestRef.current) return;
    activeRequestRef.current.reason = "cancelled";
    activeRequestRef.current.controller.abort();
    setGenerationStatus(GENERATION_STATUS.cancelling);
  };

  const getGenerationErrorMessage = (err, reason) => {
    if (reason === "timeout") {
      return `The ML backend did not respond within ${GENERATION_TIMEOUT_SECONDS} seconds. Hugging Face Spaces can be slow after cold starts. Please try again.`;
    }

    if (reason === "cancelled" || err?.name === "AbortError") {
      return "Generation cancelled. You can retry with the selected image.";
    }

    if (err?.name === "TypeError") {
      return "The ML backend could not be reached. Check the API URL or try again after the Space wakes up.";
    }

    return (
      err?.message ||
      "The image could not be processed. Please try a different, front-facing character image."
    );
  };

  const startGeneration = async () => {
    let requestState = null;
    let timeoutId;

    try {
      if (!uploadedImage || !file2d) {
        const message = "Please upload a JPG, PNG, or WebP image first.";
        setFeedback({ type: "error", message });
        toast.error(message);
        return;
      }

      if (isGenerating || activeRequestRef.current) return;

      setIsGenerating(true);
      setIsComplete(false);
      setFeedback(null);
      setGenerationStatus(GENERATION_STATUS.validating);

      const validation = validateImageFile(file2d);
      if (!validation.valid) {
        throw new Error(validation.message);
      }

      const controller = new AbortController();
      requestState = { controller, reason: null };
      activeRequestRef.current = requestState;

      const href = `${getBackendURL()}/api/rig-character`;

      const formData = new FormData();
      formData.append("image", file2d);
      formData.append("pose", "t-pose");

      timeoutId = setTimeout(() => {
        requestState.reason = "timeout";
        controller.abort();
      }, GENERATION_TIMEOUT_MS);

      setGenerationStatus(GENERATION_STATUS.uploading);
      const responsePromise = fetch(href, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      setGenerationStatus(GENERATION_STATUS.waiting);

      const response = await responsePromise;
      setGenerationStatus(GENERATION_STATUS.preparing);

      const contentType = response.headers.get("Content-Type") || "";

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(getApiErrorMessage(errorData));
      }

      if (isJsonContentType(contentType)) {
        const errorData = await response.json().catch(() => null);
        throw new Error(getApiErrorMessage(errorData));
      }

      const disposition = response.headers.get("Content-Disposition");
      let filename = "download.glb";

      if (file2d && file2d.name) {
        filename = file2d.name.replace(/\.[^/.]+$/, "") + ".glb";
      } else if (disposition && disposition.includes("filename=")) {
        filename = disposition.split("filename=")[1].replace(/["']/g, "");
      }

      const blob = await response.blob();
      const glbValidation = await validateGlbBlob(blob);
      if (!glbValidation.valid) {
        throw new Error(glbValidation.message);
      }

      const base64Data = await blobToBase64(blob);
      const dataToStore = {
        fileData: base64Data,
        filename,
        image: file2d,
        createdAt: new Date().toISOString(),
      };

      setGenerationStatus(GENERATION_STATUS.saving);
      try {
        await saveFile(dataToStore);
      } catch (storageError) {
        console.error("Failed to save generated GLB:", storageError);
        throw new Error(
          "The GLB was generated, but it could not be saved in Recent Files. Please try again."
        );
      }

      clearGeneratedPreview();
      const previewUrl = URL.createObjectURL(blob);
      previewUrlRef.current = previewUrl;
      setGeneratedPreview({ url: previewUrl, filename });

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      try {
        link.href = url;
        link.setAttribute("download", filename);
        document.body.appendChild(link);
        link.click();
      } finally {
        link.parentNode?.removeChild(link);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }

      setGenerationStatus(GENERATION_STATUS.ready);
      setIsComplete(true);
      toast.success("GLB generated and downloaded.");
    } catch (err) {
      console.error("Generation error:", err);
      const message = getGenerationErrorMessage(err, requestState?.reason);
      setFeedback({ type: "error", message });
      setGenerationStatus(null);
      setIsComplete(false);
      toast.error(message);
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      if (requestState && activeRequestRef.current === requestState) {
        activeRequestRef.current = null;
      }
      setIsGenerating(false);
    }
  };

  const feedbackClasses =
    feedback?.type === "error"
      ? "border-red-400/30 bg-red-500/10 text-red-100"
      : "border-emerald-400/30 bg-emerald-500/10 text-emerald-100";

  return (
    <div className="space-y-8">
      {/* Hero */}
      <section className="text-center">
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-white">
          {heading}
        </h1>
        <p className="mt-3 text-sm sm:text-base text-white/70 max-w-2xl mx-auto">
          {subHeading}
        </p>
      </section>

      {/* Status panel */}
      {(isGenerating || isComplete) && (
        <section
          className="glass p-5 sm:p-6"
          aria-live="polite"
          aria-busy={isGenerating}
        >
          <div className="flex items-start sm:items-center justify-between gap-4">
            <h3 className="text-lg sm:text-xl font-semibold text-white">
              {isComplete ? "GLB Ready" : "Generating Rigged GLB"}
            </h3>

            {isGenerating && (
              <button
                type="button"
                onClick={cancelGeneration}
                className="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-2 text-sm font-medium text-white transition"
                aria-label="Cancel GLB generation"
              >
                Cancel
              </button>
            )}

            {isComplete && (
              <button
                type="button"
                onClick={resetProcess}
                className="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-2 text-sm font-medium text-white transition"
              >
                Create Another
              </button>
            )}
          </div>

          {isGenerating && generationStatus && (
            <div className="mt-4 space-y-4">
              <div className="flex items-start gap-3 text-white/80">
                <Clock
                  className="mt-0.5 h-5 w-5 animate-spin shrink-0"
                  aria-hidden="true"
                />
                <div>
                  <span className="text-sm sm:text-base font-medium text-white">
                    {generationStatus.title}
                  </span>
                  <p className="mt-1 text-sm text-white/60">
                    {generationStatus.detail}
                  </p>
                </div>
              </div>

              <div
                className="h-3 w-full overflow-hidden rounded-full bg-black/30"
                aria-hidden="true"
              >
                <div className="h-3 w-full rounded-full bg-gradient-to-r from-cyan-400 via-violet-500 to-pink-500 animate-pulse" />
              </div>
              <p className="text-xs text-white/45">
                Live backend progress is not available, so this shows the
                current request phase.
              </p>
            </div>
          )}

          {isComplete && (
            <div className="mt-4 flex items-start gap-3 text-green-300">
              <CheckCircle className="h-6 w-6 shrink-0" aria-hidden="true" />
              <div>
                <span className="text-sm sm:text-base font-medium">
                  {GENERATION_STATUS.ready.title}
                </span>
                <p className="mt-1 text-sm text-green-200/80">
                  {GENERATION_STATUS.ready.detail}
                </p>
              </div>
            </div>
          )}
        </section>
      )}

      {isComplete && generatedPreview && (
        <GlbPreview
          src={generatedPreview.url}
          filename={generatedPreview.filename}
          helperText="Inspect the generated GLB in-browser. The automatic download is still preserved."
        />
      )}

      {/* Main card */}
      {!isGenerating && !isComplete && (
        <section className="glass p-5 sm:p-8">
          <div className="text-center">
            <h2 className="text-xl sm:text-2xl font-semibold text-white">
              Upload Your 2D Character Image
            </h2>
            <p id="upload-help" className="mt-2 text-sm text-white/60">
              Accepted formats: JPG, PNG, or WebP. Max file size: {uploadLimit}.
            </p>
          </div>

          {feedback && (
            <div
              id="upload-feedback"
              className={`mt-5 rounded-xl border px-4 py-3 text-sm ${feedbackClasses}`}
              role={feedback.type === "error" ? "alert" : "status"}
              aria-live={feedback.type === "error" ? "assertive" : "polite"}
            >
              {feedback.message}
            </div>
          )}

          {/* Requirements */}
          <div className="mt-6 rounded-2xl bg-black/30 p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.06)]">
            <div className="flex items-center gap-2 text-white font-medium">
              <User className="h-5 w-5 text-white/80" aria-hidden="true" />
              <span>Image Requirements</span>
            </div>

            <ul className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-y-2 gap-x-6 pl-5 list-disc text-sm text-white/75 marker:text-white/35">
              <li>Character should be in T-pose or A-pose</li>
              <li>Clear, well-lit humanoid character</li>
              <li>Minimal background preferred</li>
              <li>Character should face forward</li>
              <li>Full body visible, head to feet</li>
            </ul>
          </div>

          {/* Upload + Preview */}
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                accept={ACCEPTED_IMAGE_TYPES}
                className="hidden"
              />

              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="w-full h-full min-h-[200px] rounded-2xl border border-dashed border-white/20 bg-white/[0.04] hover:bg-white/[0.06] transition p-10 text-center"
                aria-describedby={
                  feedback ? "upload-help upload-feedback" : "upload-help"
                }
                aria-label={`Upload a 2D character image. Accepted formats are JPG, PNG, and WebP. Maximum size is ${uploadLimit}.`}
              >
                <Upload
                  className="h-12 w-12 text-white/70 mx-auto mb-4"
                  aria-hidden="true"
                />
                <div className="text-base font-medium text-white">
                  Click to upload your 2D character image
                </div>
                <div className="mt-1 text-sm text-white/55">
                  JPG, PNG, or WebP up to {uploadLimit}
                </div>

                {file2d && (
                  <div className="mt-5 rounded-xl bg-black/30 px-4 py-3 text-left">
                    <div className="text-xs uppercase tracking-wide text-white/45">
                      Selected file
                    </div>
                    <div className="mt-1 truncate text-sm font-medium text-white">
                      {file2d.name}
                    </div>
                    <div className="mt-0.5 text-xs text-white/55">
                      {formatFileSize(file2d.size)}
                    </div>
                  </div>
                )}
              </button>
            </div>

            <div className="rounded-2xl bg-black/30 p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.06)] min-h-[200px]">
              {!uploadedImage ? (
                <div className="h-full flex items-center justify-center text-center text-sm text-white/55">
                  Preview will appear here after upload.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-xl bg-black/40 overflow-hidden">
                    <img
                      src={uploadedImage}
                      alt={`Preview of ${file2d?.name || "selected image"}`}
                      className="w-full max-h-64 object-contain"
                    />
                    <div className="absolute top-2 right-2 rounded-full bg-green-500 text-white px-3 py-1 text-xs font-semibold">
                      Ready
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={startGeneration}
                    className="w-full rounded-xl bg-white/12 hover:bg-white/16 text-white py-4 px-6 font-semibold transition"
                    aria-label={`Generate rigged GLB from ${
                      file2d?.name || "selected image"
                    }`}
                  >
                    Generate Rigged GLB
                  </button>
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default Home;

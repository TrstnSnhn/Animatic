import { useRef, useState } from "react";
import { Upload, Clock, CheckCircle, User } from "lucide-react";
import toast from "react-hot-toast";

const Home = () => {
  const heading = "Go from 2D Model to Fully Rigged in Seconds. No, Really.";
  const subHeading =
    "Stop wasting hours on manual rigging. Let our app do the heavy lifting, so you can focus on animating.";

  const [uploadedImage, setUploadedImage] = useState(null);
  const [file2d, setFile2d] = useState(null);

  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStep, setGenerationStep] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  const fileInputRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFile2d(file);

    const reader = new FileReader();
    reader.onload = (e) => setUploadedImage(e.target.result);
    reader.readAsDataURL(file);

    setIsComplete(false);
    setGenerationStep("");
  };

  const resetProcess = () => {
    setUploadedImage(null);
    setFile2d(null);
    setIsGenerating(false);
    setGenerationStep("");
    setIsComplete(false);
  };

  // Frontend only mock generation.
  // This keeps the UX and lets you deploy without any backend.
  const startGeneration = async () => {
    try {
      if (!uploadedImage || !file2d) {
        toast.error("Please upload an image first.");
        return;
      }

      setIsGenerating(true);
      setIsComplete(false);

      const steps = [
        "Preparing image...",
        "Analyzing pose...",
        "Detecting keypoints...",
        "Generating mesh...",
        "Creating bone armature...",
        "Applying texture mapping...",
        "Exporting GLB file...",
      ];

      let stepIndex = 0;
      setGenerationStep(steps[stepIndex]);

      const stepInterval = setInterval(() => {
        stepIndex = Math.min(stepIndex + 1, steps.length - 1);
        setGenerationStep(steps[stepIndex]);
      }, 900);

      // Simulate work time
      await new Promise((r) => setTimeout(r, 6500));
      clearInterval(stepInterval);

      // Optional: create a placeholder file so the user still "downloads" something.
      // This is NOT a real GLB. It is just a dummy file so the UI flow stays intact.
      const baseName = file2d?.name ? file2d.name.replace(/\.[^/.]+$/, "") : "character";
      const filename = `${baseName}.glb`;

      const placeholderText =
        "Animatic frontend demo build.\nBackend model is disabled for this deployment.\n";
      const blob = new Blob([placeholderText], { type: "model/gltf-binary" });
      const url = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", filename);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);

      URL.revokeObjectURL(url);

      setIsComplete(true);
      toast.success("Demo complete. Backend is disabled in this build.");
    } catch (err) {
      console.error("Demo generation error:", err);
      toast.error("Something went wrong. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

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
        <section className="rounded-2xl bg-white/5 backdrop-blur-xl p-5 sm:p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.08),0_20px_60px_rgba(0,0,0,0.55)]">
          <div className="flex items-start sm:items-center justify-between gap-4">
            <h3 className="text-lg sm:text-xl font-semibold text-white">
              {isComplete ? "Generation Complete!" : "Generating 3D Model..."}
            </h3>

            {isComplete && (
              <button
                onClick={resetProcess}
                className="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-2 text-sm font-medium text-white transition"
              >
                Create Another
              </button>
            )}
          </div>

          {isGenerating && (
            <div className="mt-4 space-y-4">
              <div className="flex items-center gap-3 text-white/80">
                <Clock className="h-5 w-5 animate-spin" />
                <span className="text-sm sm:text-base">{generationStep}</span>
              </div>

              <div className="h-3 w-full rounded-full bg-black/30">
                <div
                  className="h-3 rounded-full bg-gradient-to-r from-slate-400 to-pink-500 animate-pulse"
                  style={{ width: "60%" }}
                />
              </div>

              <p className="text-xs text-white/55">
                This is a frontend-only demo flow. No model is being run.
              </p>
            </div>
          )}

          {isComplete && (
            <div className="mt-4 flex items-center gap-3 text-green-300">
              <CheckCircle className="h-6 w-6" />
              <span className="text-sm sm:text-base">
                Demo done. A placeholder file was downloaded. Backend is disabled.
              </span>
            </div>
          )}
        </section>
      )}

      {/* Main card */}
      {!isGenerating && !isComplete && (
        <section className="rounded-2xl bg-white/5 backdrop-blur-xl p-5 sm:p-8 shadow-[0_0_0_1px_rgba(255,255,255,0.08),0_24px_70px_rgba(0,0,0,0.55)]">
          <div className="text-center">
            <h2 className="text-xl sm:text-2xl font-semibold text-white">
              Upload Your 2D Image
            </h2>
            <p className="mt-2 text-sm text-white/60">
              Frontend demo mode. Upload works. Model generation is disabled.
            </p>
          </div>

          {/* Requirements */}
          <div className="mt-6 rounded-2xl bg-black/30 p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.06)]">
            <div className="flex items-center gap-2 text-white font-medium">
              <User className="h-5 w-5 text-white/80" />
              <span>Image Requirements</span>
            </div>

            <ul className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-y-2 gap-x-6 text-sm text-white/75">
              <li>• Character should be in T-pose or A-pose</li>
              <li>• Clear, well-lit humanoid character</li>
              <li>• Minimal background preferred</li>
              <li>• Character should face forward</li>
              <li>• Full body visible, head to feet</li>
            </ul>
          </div>

          {/* Upload + Preview */}
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                accept="image/*"
                className="hidden"
              />

              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="w-full rounded-2xl border border-dashed border-white/20 bg-white/[0.04] hover:bg-white/[0.06] transition p-10 text-center"
              >
                <Upload className="h-12 w-12 text-white/70 mx-auto mb-4" />
                <div className="text-base font-medium text-white">
                  Click to upload your 2D image
                </div>
                <div className="mt-1 text-sm text-white/55">
                  Supports JPG, PNG, WebP
                </div>
              </button>
            </div>

            <div className="rounded-2xl bg-black/30 p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.06)]">
              {!uploadedImage ? (
                <div className="h-full flex items-center justify-center text-sm text-white/55">
                  Preview will appear here after upload.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-xl bg-black/40 overflow-hidden">
                    <img
                      src={uploadedImage}
                      alt="Uploaded"
                      className="w-full max-h-64 object-contain"
                    />
                    <div className="absolute top-2 right-2 rounded-full bg-green-500 text-white px-3 py-1 text-xs font-semibold">
                      Ready
                    </div>
                  </div>

                  <button
                    onClick={startGeneration}
                    className="w-full rounded-xl bg-white/12 hover:bg-white/16 text-white py-4 px-6 font-semibold transition"
                  >
                    Run Demo Generation
                  </button>

                  <p className="text-xs text-white/55 text-center">
                    Backend model is disabled for frontend deployment.
                  </p>
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

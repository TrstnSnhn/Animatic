import { useRef, useState } from "react";
import { Upload, Clock, CheckCircle, User } from "lucide-react";
import { saveFile } from "../helper/db";
import getBackendURL from "../config";
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

  const startGeneration = async () => {
    try {
      if (!uploadedImage || !file2d) {
        toast.error("Please upload an image first.");
        return;
      }

      setIsGenerating(true);
      setIsComplete(false);

      const href = `${getBackendURL()}/api/rig-character`;

      const formData = new FormData();
      formData.append("image", file2d);
      formData.append("pose", "t-pose");

      const steps = [
        "Analyzing 2D image structure...",
        "Pose estimation...",
        "Creating mesh...",
        "Creating texture maps...",
        "Exporting GLB file...",
      ];

      setGenerationStep(steps[0]);

      const response = await fetch(href, {
        method: "POST",
        body: formData,
      });

      for (let i = 1; i < steps.length; i++) {
        setGenerationStep(steps[i]);
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ error: "An unknown server error occurred." }));
        throw new Error(
          errorData.error || `HTTP error! status: ${response.status}`
        );
      }

      const disposition = response.headers.get("Content-Disposition");
      let filename = "download.glb";

      if (file2d && file2d.name) {
        filename = file2d.name.replace(/\.[^/.]+$/, "") + ".glb";
      } else if (disposition && disposition.includes("filename=")) {
        filename = disposition.split("filename=")[1].replace(/["']/g, "");
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      const base64Data = await blobToBase64(blob);
      const dataToStore = { fileData: base64Data, filename, image: file2d };

      await saveFile(dataToStore);

      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", filename);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);

      setIsComplete(true);
      toast.success("3D model generated successfully!");
    } catch (err) {
      console.error("Generation error:", err);
      toast.error(
        err?.message || "Error processing image. Please try a different image."
      );
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
        <section className="glass p-5 sm:p-6">
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
                  className="h-3 rounded-full bg-gradient-to-r from-cyan-400 via-violet-500 to-pink-500 animate-pulse"
                  style={{ width: "60%" }}
                />
              </div>
            </div>
          )}

          {isComplete && (
            <div className="mt-4 flex items-center gap-3 text-green-300">
              <CheckCircle className="h-6 w-6" />
              <span className="text-sm sm:text-base">
                Your GLB file is ready and downloading automatically!
              </span>
            </div>
          )}
        </section>
      )}

      {/* Main card */}
      {!isGenerating && !isComplete && (
        <section className="glass p-5 sm:p-8">
          <div className="text-center">
            <h2 className="text-xl sm:text-2xl font-semibold text-white">
              Upload Your 2D Image
            </h2>
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
                className="w-full h-full min-h-[200px] rounded-2xl border border-dashed border-white/20 bg-white/[0.04] hover:bg-white/[0.06] transition p-10 text-center"
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

            <div className="rounded-2xl bg-black/30 p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.06)] min-h-[200px]">
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
                    Generate 3D Model
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
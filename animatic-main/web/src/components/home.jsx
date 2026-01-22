import { useState, useRef } from "react";
import { presets } from "../utils";
import { Upload, Clock, CheckCircle, User,  } from 'lucide-react';
import { saveFile } from "../helper/db";
import getBackendURL from "../config";
import toast from "react-hot-toast";

const Home = () => {
    const heading = "Go from 2D Model to Fully Rigged in Seconds. No, Really."
    const subHeading = "Stop wasting hours on manual rigging. Let our app do the heavy lifting, so you can focus on animating."
    const [selectedPreset, setSelectedPreset] = useState(null);
    const [uploadedImage, setUploadedImage] = useState(null);
    const [file2d, setFile2d] = useState(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generationStep, setGenerationStep] = useState('');
    const [isComplete, setIsComplete] = useState(false);
    const [previewImage, setPreviewImage] = useState(null); // A URL for displaying a preview of the uploaded image
    const [glbUrl, setGlbUrl] = useState(null);

    const fileInputRef = useRef(null);

  const blobToBase64 = (blob) => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
  });

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
          setFile2d(file)
        const reader = new FileReader();
        reader.onload = (e) => setUploadedImage(e.target.result);
        reader.readAsDataURL(file);
        }
    };

    const startGeneration = async () => {
        try {

        if (!uploadedImage) return;
        
        setIsGenerating(true);
        setIsComplete(false);
        
          const href = `${getBackendURL()}/rig-character`
          
          const formData = new FormData();
          formData.append('image', file2d);
          formData.append('pose', "t-pose");

          const steps = [
          'Analyzing 2D image structure...',
          'Pose estimation...',
          `Creating mesh...`,
          'Creating texture maps...',
          'Exporting glb file...'
          ];
          setGenerationStep(steps[0]);

          const response = await fetch(href, {
            method: 'POST',
            body: formData,
          });
          
          for (let i = 1; i < steps?.length; i++) {
          setGenerationStep(steps[i]);
          await new Promise(resolve => setTimeout(resolve, 500));
          }
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: "An unknown server error occurred." }));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
          }

          const disposition = response.headers.get("Content-Disposition");
         let filename = "download.glb"; // fallback

        if (file2d && file2d.name) {
          filename = file2d.name.replace(/\.[^/.]+$/, "") + ".glb";
        } else if (disposition && disposition.includes("filename=")) {
          filename = disposition
            .split("filename=")[1]
            .replace(/["']/g, "");
        }

          const blob = await response.blob();
          const link = document.createElement('a');

          const url = URL.createObjectURL(blob);
          setGlbUrl(url);

          const base64Data = await blobToBase64(blob);
          const dataToStore = { fileData: base64Data, filename, image: file2d };

          await saveFile(dataToStore);

          link.href = url;
          link.setAttribute('download', filename);
          document.body.appendChild(link);
          link.click(); 
          link.parentNode.removeChild(link); 
          setIsComplete(true);
      } catch (err) {
        toast.error(err?.message || 'Error processing image. Please try different image.')
      } finally {
        setIsGenerating(false);
      }
        
    };

    const resetProcess = () => {
        setSelectedPreset(null);
        setUploadedImage(null);
        setIsGenerating(false);
        setGenerationStep('');
        setIsComplete(false);
    };
    return (
        <>
         <div className="max-w-7xl mx-auto px-4 py-8">
            <div className="items-center min-h-5/12 mb-5 align-middle">
                <p className="text-4xl font-bold text-slate-300 ">{heading}</p>
                <p className="text-slate-100 text-xl pt-5">{subHeading}</p>
            </div>
          <>
            {(isGenerating || isComplete) && (
              <div className="mb-8 bg-black/40 backdrop-blur-sm rounded-2xl border border-slate-500/30 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold text-white">
                    {isComplete ? 'Generation Complete!' : 'Generating 3D Model...'}
                  </h3>
                  {isComplete && (
                    <button
                      onClick={resetProcess}
                      className="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg font-medium transition-colors"
                    >
                      Create Another
                    </button>
                  )}
                </div>
                
                {isGenerating && (
                  <div className="space-y-4">
                    <div className="flex items-center space-x-3 text-slate-300">
                      <Clock className="w-5 h-5 animate-spin" />
                      <span className="text-lg">{generationStep}</span>
                    </div>
                    <div className="w-full bg-slate-900/50 rounded-full h-3">
                      <div className="bg-gradient-to-r from-slate-500 to-pink-500 h-3 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                    </div>
                  </div>
                )}
                
                {isComplete && (
                  <div className="flex items-center space-x-3 text-green-400">
                    <CheckCircle className="w-6 h-6" />
                    <span className="text-lg">Your GLB file is ready and downloading automatically!</span>
                  </div>
                )}
              </div>
            )}

            {/* Preset Selection */}
            {!isGenerating && !isComplete && (
              <div className="space-y-8">
                { false && <div className="text-center">
                  <h2 className="text-3xl font-bold text-white mb-3">Choose Your Model Type</h2>
                  <p className="text-slate-300 text-lg">Select the preset that best matches your 2D image</p>
                </div>}

               { false && <div className="grid grid-cols-2 gap-6 px-3 md:px-12">
                  {presets.map((preset) => {
                    
                    const IconComponent = preset.icon;
                    const isAvailable = preset.available
                    if (!isAvailable) {
                        return
                    }
                    return (
                      <div
                        key={preset.id}
                        onClick={() => setSelectedPreset(preset.id)}
                        className={`group cursor-pointer bg-black/40 backdrop-blur-sm rounded-2xl border-2 transition-all duration-300 hover:scale-105 ${
                          selectedPreset?.id === preset.id
                            ? 'border-slate-500 shadow-2xl shadow-slate-500/25'
                            : 'border-slate-500/20 hover:border-slate-400/50'
                        }`}
                      >
                        <div className="relative overflow-hidden rounded-t-2xl">
                          <img
                            src={preset.image}
                            alt={preset.name}
                            className="w-full h-48 object-cover transition-transform group-hover:scale-110"
                          />
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                          <div className="absolute bottom-4 left-4 flex items-center space-x-2">
                            <div className="w-8 h-8 bg-slate-600 rounded-lg flex items-center justify-center">
                              <IconComponent className="w-4 h-4 text-white" />
                            </div>
                            <h3 className="text-white font-semibold text-lg">{preset.name}</h3>
                          </div>
                        </div>
                        <div className="p-6">
                          <p className="text-slate-200 mb-4">{preset.description}</p>
                          <div className="space-y-2">
                            {preset.features.map((feature, idx) => (
                              <div key={idx} className="flex items-center space-x-2 text-sm text-slate-300">
                                <div className="w-1.5 h-1.5 bg-slate-500 rounded-full" />
                                <span>{feature}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        {selectedPreset?.id === preset.id && (
                          <div className="absolute -top-2 -right-2 w-8 h-8 bg-slate-500 rounded-full flex items-center justify-center shadow-lg">
                            <CheckCircle className="w-5 h-5 text-white" />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>}

                {/* Image Upload */}
                {(
                  <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-slate-500/30 mx-1 md:mx-12 p-4 md:p-8">
                    
                    <h3 className="text-2xl font-bold text-white mb-6 text-center">
                      Upload Your 2D Image
                    </h3>
                     {/* Image Requirements */}
                    <div className="bg-gray-800 mt-5 rounded-lg p-4 mb-6">
                      <h3 className="text-white font-semibold mb-2 flex items-center">
                        <User className="w-5 h-5 mr-2" />
                        Image Requirements
                      </h3>
                      <ul className="text-gray-300 text-left text-sm space-y-1">
                        <li>• Character should be in T-pose for best results</li>
                        <li>• Clear, well-lit humanoid character</li>
                        <li>• Minimal background preferred</li>
                        <li>• Character should face forward</li>
                      </ul>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <div>
                        <input
                          type="file"
                          ref={fileInputRef}
                          onChange={handleImageUpload}
                          accept="image/*"
                          className="hidden"
                        />
                        
                        <div
                          onClick={() => fileInputRef.current?.click()}
                          className="relative group cursor-pointer border-2 border-dashed border-slate-500/50 rounded-xl p-12 text-center hover:border-slate-400 transition-colors bg-slate-500/5"
                        >
                          <Upload className="w-16 h-16 text-slate-400 mx-auto mb-4 group-hover:scale-110 transition-transform" />
                          <p className="text-slate-300 text-lg mb-2">Click to upload your 2D image</p>
                          <p className="text-slate-400 text-sm">Supports JPG, PNG, WebP formats</p>
                        </div>
                      </div>
                      
                      {uploadedImage && (
                        <div className="space-y-4">
                          <div className="relative">
                            <img
                              src={uploadedImage}
                              alt="Uploaded"
                              className="w-full max-h-64 object-contain rounded-xl bg-black/50"
                            />
                            <div className="absolute top-2 right-2 bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                              Ready
                            </div>
                          </div>
                          
                          <button
                            onClick={startGeneration}
                            className="w-full bg-gradient-to-r from-slate-600 to-pink-600 hover:from-slate-700 hover:to-pink-700 text-white py-4 px-8 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 shadow-xl"
                          >
                            Generate 3D Model
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
          </div>
        </>
    )
}

export default Home;
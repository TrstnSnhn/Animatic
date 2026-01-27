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
    const [previewImage, setPreviewImage] = useState(null);
    const [glbUrl, setGlbUrl] = useState(null);

    const fileInputRef = useRef(null);

    // Convert file to base64
    const fileToBase64 = (file) => new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = (error) => reject(error);
    });

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
            if (!uploadedImage || !file2d) return;
            
            setIsGenerating(true);
            setIsComplete(false);
            
            const steps = [
                'Uploading image to server...',
                'CNN analyzing character pose...',
                'Detecting 21 keypoints...',
                'Generating mesh vertices...',
                'Creating bone armature...',
                'Applying texture mapping...',
                'Exporting GLB file...'
            ];
            
            // Start showing steps
            let stepIndex = 0;
            setGenerationStep(steps[stepIndex]);
            
            const stepInterval = setInterval(() => {
                stepIndex = (stepIndex + 1) % steps.length;
                setGenerationStep(steps[stepIndex]);
            }, 2000);

            // Convert image to base64 for Gradio API
            const base64Image = await fileToBase64(file2d);
            
            // Call Gradio API
            const baseUrl = getBackendURL();
            
            // First, get the API endpoint info
            const response = await fetch(`${baseUrl}/gradio_api/call/process_character_image`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: [base64Image]
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            const eventId = result.event_id;

            // Poll for the result
            const resultResponse = await fetch(`${baseUrl}/gradio_api/call/process_character_image/${eventId}`);
            
            if (!resultResponse.ok) {
                throw new Error(`Failed to get result: ${resultResponse.status}`);
            }

            // Parse SSE response
            const text = await resultResponse.text();
            const lines = text.split('\n');
            let data = null;
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        data = JSON.parse(line.slice(6));
                    } catch (e) {
                        // Continue parsing
                    }
                }
            }

            clearInterval(stepInterval);

            if (!data || !data[0]) {
                throw new Error('No GLB file returned from server');
            }

            // Get the GLB file URL
            const glbFileInfo = data[0];
            let glbDownloadUrl;
            
            if (typeof glbFileInfo === 'string') {
                glbDownloadUrl = glbFileInfo;
            } else if (glbFileInfo.url) {
                glbDownloadUrl = glbFileInfo.url;
            } else if (glbFileInfo.path) {
                glbDownloadUrl = `${baseUrl}/gradio_api/file=${glbFileInfo.path}`;
            } else {
                throw new Error('Invalid response format');
            }

            // Download the GLB file
            const glbResponse = await fetch(glbDownloadUrl);
            if (!glbResponse.ok) {
                throw new Error('Failed to download GLB file');
            }

            const blob = await glbResponse.blob();
            
            // Generate filename
            let filename = "character.glb";
            if (file2d && file2d.name) {
                filename = file2d.name.replace(/\.[^/.]+$/, "") + ".glb";
            }

            // Create download link
            const url = URL.createObjectURL(blob);
            setGlbUrl(url);

            // Save to IndexedDB
            const base64Data = await blobToBase64(blob);
            const dataToStore = { fileData: base64Data, filename, image: file2d };
            await saveFile(dataToStore);

            // Trigger download
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', filename);
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
            
            setIsComplete(true);
            toast.success('3D model generated successfully! 🎉');

        } catch (err) {
            console.error('Generation error:', err);
            toast.error(err?.message || 'Error processing image. Please try a different image.')
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
                        <li>• Character should be in T-pose or A-pose for best results</li>
                        <li>• Clear, well-lit humanoid character</li>
                        <li>• Minimal background preferred</li>
                        <li>• Character should face forward</li>
                        <li>• Full body visible (head to feet)</li>
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
                            🚀 Generate 3D Model
                          </button>
                          
                          <p className="text-slate-400 text-sm text-center">
                            Powered by CNN with 86.26% accuracy
                          </p>
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
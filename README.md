# Animatic

Animatic is a thesis-era ML web app that turns a front-facing 2D character image into a downloadable rigged GLB asset.

It does not perform full 3D reconstruction. The workflow predicts 2D character keypoints, uses them to build a simple mesh and armature, embeds the uploaded image as texture data, and returns a `.glb` file that can be previewed in the browser.

## Live Demo

- **Frontend:** <https://animatic-two.vercel.app/>
- **Backend:** Flask ML API hosted separately on Hugging Face Spaces and called by the frontend.

## Production Status

The deployed app has been verified with the live Vercel frontend and live Hugging Face Spaces backend. A representative 2D anime character image generated a valid GLB, triggered automatic download, loaded in the GLB preview, and appeared in Recent Files for preview, redownload, and deletion.

## Feature Highlights

- Upload JPG, PNG, or WebP character images.
- Validate file type and upload size before generation.
- Send image uploads to the ML API with `FormData`.
- Generate and auto-download a rigged `.glb` file.
- Preview generated GLB files in-browser.
- Save generated files locally with IndexedDB.
- Preview, redownload, and delete recent generated files.
- Keep the large ML model outside the static frontend deployment.

## Tech Stack

- **Frontend:** Vite, Preact, Tailwind CSS
- **Preview:** `@google/model-viewer`
- **Browser storage:** IndexedDB through `idb`
- **Backend:** Flask API
- **ML runtime:** Keras/TensorFlow model for 2D keypoint prediction
- **Frontend hosting:** Vercel
- **ML API hosting:** Hugging Face Spaces
- **Output:** GLB with simple mesh, armature, and embedded texture data

## Repository Structure

```text
animatic-main/
  web/      Vite + Preact frontend
  api/      Flask ML API and GLB generation code
  README.md Full project documentation
```

Additional thesis-era datasets, evaluation files, and experiment artifacts are kept in the repository for project context.

## Deployment Summary

- The Vercel project root should be `animatic-main/web`.
- The frontend should set `VITE_API_URL` to the deployed Hugging Face Spaces API URL.
- The Flask API and large ML model should remain on Hugging Face Spaces or another Python/ML-capable host.
- The trained model is expected through `CNN_MODEL_PATH`, or the API default model path documented in the full README.
- The large model is not meant to be served by Vercel.

## Full Documentation

See [animatic-main/README.md](animatic-main/README.md) for the complete project overview, architecture, local development setup, deployment notes, limitations, and planned improvements.

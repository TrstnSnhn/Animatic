# Animatic

Animatic is a thesis-era ML web app for turning a 2D character image into a downloadable rigged GLB asset. It does not perform full 3D reconstruction. Instead, it predicts 2D character keypoints, uses those keypoints to build a simple mesh and armature, embeds the uploaded image as texture data, and returns a `.glb` file with an in-browser preview.

The project connects a Vite + Preact upload interface to a hosted ML-backed Flask API. The static frontend is designed for Vercel, while the model-serving API is designed to run separately on Hugging Face Spaces.

## Live Demo

- **Frontend:** <https://animatic-two.vercel.app/>
- **Backend:** Flask ML API hosted on Hugging Face Spaces and called by the frontend. It is part of this project deployment, not a separately packaged public API product.

## Production Status

The current deployed app has been verified in production with the live Vercel frontend and live Hugging Face Spaces backend. A representative 2D anime character image successfully generated a valid `.glb`, triggered automatic download, loaded in the GLB preview, and appeared in Recent Files for preview, download, and deletion.

## Project Overview

Animatic provides a browser-based workflow for uploading a front-facing 2D character image, generating a basic rigged GLB, previewing it in-browser, and downloading it. The goal is to demonstrate an end-to-end ML web application: frontend upload flow, API integration, model-backed keypoint prediction, 3D asset generation basics, and deployment across separate frontend and backend hosts.

## Features

- Upload a JPG, PNG, or WebP 2D character image.
- Validate file type and upload size before generation.
- Send image uploads to the backend with `FormData`.
- Show honest generation states for upload, backend wait, preparation, saving, success, cancellation, and errors.
- Generate and auto-download a `.glb` file.
- Preview generated GLB files in-browser.
- Store generated GLB files locally with IndexedDB.
- Preview, redownload, and delete recent generated files.
- Keep the frontend lightweight by hosting the larger ML model separately.

## How It Works

1. A user uploads a 2D character image in the web app.
2. The frontend sends the image to the API using `FormData`.
3. The Flask API predicts 2D character keypoints with a Keras/TensorFlow model.
4. The API uses the predicted keypoints to create a simple mesh and armature structure.
5. The uploaded image is embedded as texture data.
6. The API returns a `.glb` file for download.
7. The frontend shows an in-browser GLB preview and keeps the download flow available.
8. The frontend can store generated files locally in the browser with IndexedDB.

## Tech Stack

- **Frontend:** Vite, Preact, Tailwind CSS
- **Preview:** `@google/model-viewer` for in-browser GLB viewing
- **Browser storage:** IndexedDB through `idb`
- **Backend API:** Flask for upload handling, keypoint prediction, and GLB generation
- **ML runtime:** Keras/TensorFlow model usage for 2D keypoint prediction
- **Frontend hosting:** Vercel static deployment
- **ML API hosting:** Hugging Face Spaces
- **Output format:** GLB with simple mesh, armature, and embedded texture data

## Architecture

- `animatic-main/web` contains the Vite + Preact single-page frontend.
- `animatic-main/api` contains the Flask API used for model inference and GLB generation.
- The frontend calls the API through `VITE_API_URL`, falling back to the configured Hugging Face Space URL when unset.
- The API expects the trained Keras model from `CNN_MODEL_PATH`, or the default `trained_model/best_model.keras` path.
- Generated files are downloaded immediately and can also be stored locally in the browser with IndexedDB.

## Technical Skills Demonstrated

- Full-stack web app development with a static frontend and separate ML API backend
- Vite + Preact frontend development
- Flask API development for image upload and GLB generation workflows
- Machine learning model integration with a hosted inference service
- Hugging Face Spaces deployment for the ML-backed API
- Vercel static frontend deployment
- Computer vision and 2D character keypoint prediction
- Keras/TensorFlow model usage for keypoint detection
- Keypoint-driven mesh and armature creation for rigged GLB output
- Texture embedding from the uploaded 2D character image
- API integration using `FormData` image uploads
- IndexedDB/local browser storage for generated GLB files
- Frontend state management for upload, generation, loading, success, and error states
- Deployment configuration and environment variable management
- UI/UX improvement planning for ML-driven user flows
- Technical documentation and project maintenance

## Portfolio Positioning

Animatic demonstrates practical experience building and deploying an ML-backed web application from a thesis project. It shows how a frontend product flow can connect to a hosted computer vision API and a basic 3D asset generation pipeline, while keeping the scope honest: the system generates a rigged GLB from predicted 2D keypoints rather than reconstructing a full 3D character.

## Production Validation

The deployed app was tested with a representative front-facing 2D anime character image. The live Hugging Face backend returned a valid GLB file, the Vercel frontend started the download, and both generated and recent-file previews loaded successfully.

The output should be understood as a simple textured rigged GLB generated from 2D keypoints. It is useful for demonstrating the workflow and asset pipeline, but it is not comparable to full 3D reconstruction or production character rigging tools.

## Current Limitations

- The project does not perform full 3D reconstruction.
- Output quality depends heavily on 2D keypoint prediction quality.
- The generated mesh and rig are intentionally simple.
- Input images work best when the full body is visible, front-facing, and close to a T-pose or A-pose.
- The large trained model is hosted separately and is not committed directly in this repo.
- Hugging Face Spaces cold starts or long processing times may affect response time.
- The frontend does not receive true backend progress updates while generation is running.
- Automated test coverage is still focused on validation utilities rather than full end-to-end flows.

## Future Improvements

- Improve model output quality and robustness across character styles.
- Add better keypoint confidence checks before GLB generation.
- Add richer GLB, mesh, and rig validation.
- Add sample inputs and expected output examples for reviewers.
- Add polished screenshots or a short demo GIF to the README.
- Add API observability and clearer backend logs for production debugging.
- Add a queued job flow if generation becomes slow or concurrent usage grows.
- Improve Hugging Face cold-start handling and user messaging.
- Expand automated frontend, API, and end-to-end tests.

## Local Development

### Frontend

```bash
cd animatic-main/web
npm install
npm run dev
```

Available frontend scripts:

```bash
npm run build
npm run preview
```

The frontend reads the API URL from `VITE_API_URL`. If it is not set, the current code falls back to the configured Hugging Face Space URL.

### API

```bash
cd animatic-main/api
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
flask --app main run
```

The API expects the trained Keras model to be available through `CNN_MODEL_PATH`, or at the default path `trained_model/best_model.keras`. The model file is large and is handled separately from the main app source.

## Deployment Notes

The frontend and ML API are deployed separately:

- **Frontend:** deploy `animatic-main/web` as the Vercel project root for the Vite static app.
- **API:** deploy `animatic-main/api` as the Flask/Gunicorn service on Hugging Face Spaces.
- **Configuration:** set the frontend `VITE_API_URL` value to the deployed Hugging Face Spaces API URL.
- **Model path:** make the trained Keras model available to the API through `CNN_MODEL_PATH`, or place it at the default `trained_model/best_model.keras` path expected by the API.
- **Large model files:** do not serve the trained model from Vercel. Keep model storage and Python inference in the Hugging Face Space.

This separation keeps the Vercel deployment lightweight while allowing the larger ML model and Python dependencies to run in the API environment.

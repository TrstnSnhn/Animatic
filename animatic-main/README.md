# Animatic

Animatic is a thesis-era ML web app for turning a 2D character image into a downloadable rigged GLB asset. It does not perform full 3D reconstruction. Instead, it predicts 2D character keypoints, uses those keypoints to build a simple mesh and armature, embeds the uploaded image as texture data, and returns a `.glb` file.

The project connects a Vite + Preact upload interface to a hosted ML-backed Flask API. The static frontend is designed for Vercel, while the model-serving API is designed to run separately on Hugging Face Spaces.

## Project Overview

Animatic provides a browser-based workflow for uploading a front-facing 2D character image and generating a basic rigged GLB. The goal is to demonstrate an end-to-end ML web application: frontend upload flow, API integration, model-backed keypoint prediction, 3D asset generation basics, and deployment across separate frontend and backend hosts.

## How It Works

1. A user uploads a 2D character image in the web app.
2. The frontend sends the image to the API using `FormData`.
3. The Flask API predicts 2D character keypoints with a Keras/TensorFlow model.
4. The API uses the predicted keypoints to create a simple mesh and armature structure.
5. The uploaded image is embedded as texture data.
6. The API returns a `.glb` file for download.
7. The frontend can store generated files locally in the browser with IndexedDB.

## Architecture

- **Frontend:** Vite + Preact single-page app
- **Backend API:** Flask API for upload handling, keypoint prediction, and GLB generation
- **ML hosting:** Hugging Face Spaces for the ML-backed API
- **Frontend hosting:** Vercel static deployment
- **Local browser storage:** IndexedDB for generated GLB files
- **Deployment configuration:** `VITE_API_URL` for API endpoint configuration and Vercel rewrites for SPA routing

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

## Current Limitations

- The project does not perform full 3D reconstruction.
- Output quality depends heavily on 2D keypoint prediction quality.
- The large trained model is hosted separately and is not committed directly in this repo.
- Hugging Face Spaces cold starts or long processing times may affect response time.
- The app needs stronger upload validation, request timeout handling, cleanup of generated temp files, and clearer user feedback.
- Deployment documentation is still minimal and should be expanded.

## Planned Improvements

- Add request timeout, cancel, and retry behavior for generation requests.
- Validate uploaded file type, size, and image readability before processing.
- Clean up temporary GLB files after API responses.
- Return safer, structured API errors instead of raw exception messages.
- Improve deployment documentation for Vercel and Hugging Face Spaces.
- Add an in-browser GLB preview after generation.
- Add tests for frontend request handling and API validation behavior.

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

- **Frontend:** deploy `animatic-main/web` as a Vite static app on Vercel.
- **API:** deploy `animatic-main/api` as the Flask/Gunicorn service on Hugging Face Spaces.
- **Configuration:** set the frontend `VITE_API_URL` value to the deployed Hugging Face Spaces API URL.

This separation keeps the Vercel deployment lightweight while allowing the larger ML model and Python dependencies to run in the API environment.

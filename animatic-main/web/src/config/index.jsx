const getBackendURL = () => {
    // HuggingFace Spaces Flask backend
    return import.meta.env.VITE_API_URL || "https://trstnsnhn-animatics.hf.space";
};

export default getBackendURL;
const getBackendURL = () => {
    return import.meta.env.VITE_API_URL || "https://trstnsnhn-animatics.hf.space";
}
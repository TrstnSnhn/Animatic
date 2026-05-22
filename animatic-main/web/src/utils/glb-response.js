const GLB_MAGIC = "glTF";

export function isJsonContentType(contentType = "") {
  return contentType.toLowerCase().includes("application/json");
}

export function getApiErrorMessage(payload, fallback = "The server could not process this image.") {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }

  if (typeof payload.error === "string" && payload.error.trim()) {
    return payload.error;
  }

  if (
    payload.error &&
    typeof payload.error === "object" &&
    typeof payload.error.message === "string" &&
    payload.error.message.trim()
  ) {
    return payload.error.message;
  }

  if (typeof payload.message === "string" && payload.message.trim()) {
    return payload.message;
  }

  return fallback;
}

export async function validateGlbBlob(blob) {
  if (!blob || blob.size < 4) {
    return {
      valid: false,
      message: "The backend response was not a valid GLB file. Please try again.",
    };
  }

  const header = await blob.slice(0, 4).arrayBuffer();
  const magic = new TextDecoder().decode(header);

  if (magic !== GLB_MAGIC) {
    return {
      valid: false,
      message: "The backend response was not a valid GLB file. Please try again.",
    };
  }

  return { valid: true };
}

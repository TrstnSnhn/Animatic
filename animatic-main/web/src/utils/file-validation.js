export const MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024; // Keep uploads conservative for hosted ML processing.

const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

export function validateImageFile(file) {
  if (!file) {
    return { valid: false, message: "Please choose an image file first." };
  }

  if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
    return {
      valid: false,
      message: "Please upload a JPG, PNG, or WebP image.",
    };
  }

  if (file.size <= 0) {
    return {
      valid: false,
      message: "The selected image appears to be empty.",
    };
  }

  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    return {
      valid: false,
      message: "Please upload an image smaller than 10 MB.",
    };
  }

  return { valid: true };
}

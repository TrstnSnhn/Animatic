const DEFAULT_GLB_CONTENT_TYPE = "model/gltf-binary";

export function dataUrlToBlob(dataUrl, contentType = DEFAULT_GLB_CONTENT_TYPE) {
  if (typeof dataUrl !== "string" || !dataUrl.includes(",")) {
    throw new Error("Invalid saved GLB data.");
  }

  const [, base64Data] = dataUrl.split(",", 2);
  if (!base64Data) {
    throw new Error("Invalid saved GLB data.");
  }

  let byteCharacters;
  try {
    byteCharacters = atob(base64Data);
  } catch {
    throw new Error("Invalid saved GLB data.");
  }
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);
    const byteNumbers = new Array(slice.length);

    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    byteArrays.push(new Uint8Array(byteNumbers));
  }

  return new Blob(byteArrays, { type: contentType });
}

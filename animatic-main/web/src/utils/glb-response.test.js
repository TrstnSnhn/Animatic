import test from "node:test";
import assert from "node:assert/strict";

import {
  getApiErrorMessage,
  isJsonContentType,
  validateGlbBlob,
} from "./glb-response.js";

test("recognizes JSON response content types", () => {
  assert.equal(isJsonContentType("application/json; charset=utf-8"), true);
  assert.equal(isJsonContentType("model/gltf-binary"), false);
});

test("validates GLB blobs by magic header", async () => {
  const blob = new Blob([new Uint8Array([0x67, 0x6c, 0x54, 0x46, 0, 0, 0, 0])]);
  const result = await validateGlbBlob(blob);

  assert.equal(result.valid, true);
});

test("rejects non-GLB blobs", async () => {
  const blob = new Blob([JSON.stringify({ error: "bad request" })], {
    type: "application/json",
  });
  const result = await validateGlbBlob(blob);

  assert.equal(result.valid, false);
  assert.match(result.message, /valid GLB/);
});

test("extracts safe API error messages", () => {
  assert.equal(
    getApiErrorMessage({ error: { message: "Upload is too large" } }),
    "Upload is too large"
  );
  assert.equal(
    getApiErrorMessage({ error: "Internal traceback detail" }),
    "Internal traceback detail"
  );
  assert.equal(getApiErrorMessage(null, "Fallback"), "Fallback");
});

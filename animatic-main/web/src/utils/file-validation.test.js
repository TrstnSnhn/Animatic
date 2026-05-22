import test from "node:test";
import assert from "node:assert/strict";

import {
  formatFileSize,
  MAX_UPLOAD_SIZE_BYTES,
  validateImageFile,
} from "./file-validation.js";

test("accepts supported image files within the size limit", () => {
  const result = validateImageFile({
    name: "character.png",
    type: "image/png",
    size: MAX_UPLOAD_SIZE_BYTES,
  });

  assert.equal(result.valid, true);
});

test("rejects unsupported file types", () => {
  const result = validateImageFile({
    name: "notes.txt",
    type: "text/plain",
    size: 1024,
  });

  assert.equal(result.valid, false);
  assert.match(result.message, /JPG, PNG, or WebP/);
});

test("rejects oversized files", () => {
  const result = validateImageFile({
    name: "huge.png",
    type: "image/png",
    size: MAX_UPLOAD_SIZE_BYTES + 1,
  });

  assert.equal(result.valid, false);
  assert.match(result.message, /10 MB/);
});

test("formats file sizes for upload feedback", () => {
  assert.equal(formatFileSize(0), "0 KB");
  assert.equal(formatFileSize(1536), "1.5 KB");
  assert.equal(formatFileSize(3 * 1024 * 1024), "3 MB");
});

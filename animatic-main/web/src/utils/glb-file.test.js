import test from "node:test";
import assert from "node:assert/strict";

import { dataUrlToBlob } from "./glb-file.js";

test("converts saved GLB data URLs back into binary blobs", async () => {
  const dataUrl = `data:model/gltf-binary;base64,${btoa("glTF-test")}`;

  const blob = dataUrlToBlob(dataUrl);
  const text = await blob.text();

  assert.equal(blob.type, "model/gltf-binary");
  assert.equal(text, "glTF-test");
});

test("rejects malformed saved GLB data", () => {
  assert.throws(() => dataUrlToBlob("not-a-data-url"), /Invalid saved GLB data/);
});

test("rejects invalid base64 GLB data safely", () => {
  assert.throws(
    () => dataUrlToBlob("data:model/gltf-binary;base64,%%%"),
    /Invalid saved GLB data/
  );
});

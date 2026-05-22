import io
import unittest

from validation import (
    MAX_REQUEST_SIZE_BYTES,
    MAX_UPLOAD_SIZE_BYTES,
    UploadValidationError,
    validate_uploaded_image,
)


class UploadStub:
    def __init__(self, filename, mimetype, data):
        self.filename = filename
        self.mimetype = mimetype
        self.stream = io.BytesIO(data)


class UploadValidationTests(unittest.TestCase):
    def test_accepts_supported_image_signature(self):
        upload = UploadStub(
            "character.png",
            "image/png",
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
        )

        validate_uploaded_image(upload, content_length=len(upload.stream.getvalue()))
        self.assertEqual(upload.stream.tell(), 0)

    def test_rejects_empty_filename(self):
        upload = UploadStub("", "image/png", b"\x89PNG\r\n\x1a\n")

        with self.assertRaises(UploadValidationError) as ctx:
            validate_uploaded_image(upload)

        self.assertEqual(ctx.exception.code, "missing_file")

    def test_rejects_oversized_upload(self):
        upload = UploadStub(
            "character.png",
            "image/png",
            b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1),
        )

        with self.assertRaises(UploadValidationError) as ctx:
            validate_uploaded_image(upload)

        self.assertEqual(ctx.exception.code, "file_too_large")

    def test_allows_multipart_overhead_above_file_limit(self):
        upload = UploadStub(
            "character.png",
            "image/png",
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
        )

        validate_uploaded_image(upload, content_length=MAX_UPLOAD_SIZE_BYTES + 256)

    def test_rejects_request_size_beyond_overhead_allowance(self):
        upload = UploadStub("character.png", "image/png", b"\x89PNG\r\n\x1a\n")

        with self.assertRaises(UploadValidationError) as ctx:
            validate_uploaded_image(upload, content_length=MAX_REQUEST_SIZE_BYTES + 1)

        self.assertEqual(ctx.exception.code, "file_too_large")

    def test_rejects_unsupported_mime_type(self):
        upload = UploadStub("notes.txt", "text/plain", b"hello")

        with self.assertRaises(UploadValidationError) as ctx:
            validate_uploaded_image(upload)

        self.assertEqual(ctx.exception.code, "unsupported_file_type")

    def test_rejects_image_mime_with_invalid_signature(self):
        upload = UploadStub("character.png", "image/png", b"not an image")

        with self.assertRaises(UploadValidationError) as ctx:
            validate_uploaded_image(upload)

        self.assertEqual(ctx.exception.code, "invalid_image")


if __name__ == "__main__":
    unittest.main()

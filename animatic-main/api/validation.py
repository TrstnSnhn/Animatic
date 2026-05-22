MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_SIZE_BYTES = MAX_UPLOAD_SIZE_BYTES + (512 * 1024)

ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class UploadValidationError(ValueError):
    def __init__(self, code, message, status_code=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _has_supported_image_signature(header):
    return (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or (len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP")
    )


def validate_uploaded_image(file_storage, content_length=None):
    if not file_storage or not getattr(file_storage, "filename", ""):
        raise UploadValidationError("missing_file", "No image file was provided.")

    if content_length and content_length > MAX_REQUEST_SIZE_BYTES:
        raise UploadValidationError(
            "file_too_large",
            "Please upload an image smaller than 10 MB.",
            413,
        )

    mimetype = (getattr(file_storage, "mimetype", "") or "").lower()
    if mimetype not in ALLOWED_IMAGE_MIME_TYPES:
        raise UploadValidationError(
            "unsupported_file_type",
            "Please upload a JPG, PNG, or WebP image.",
        )

    stream = getattr(file_storage, "stream", None)
    if stream is None:
        raise UploadValidationError("invalid_image", "The uploaded image could not be read.")

    position = stream.tell()
    stream.seek(0, 2)
    file_size = stream.tell()
    stream.seek(position)

    if file_size <= 0:
        raise UploadValidationError("invalid_image", "The uploaded image appears to be empty.")

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise UploadValidationError(
            "file_too_large",
            "Please upload an image smaller than 10 MB.",
            413,
        )

    header = stream.read(32)
    stream.seek(position)

    if not _has_supported_image_signature(header):
        raise UploadValidationError(
            "invalid_image",
            "The uploaded file does not appear to be a valid image.",
        )

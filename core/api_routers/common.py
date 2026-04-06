import json
from importlib import import_module
from typing import Any

from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse, Response


def api_module():
    return import_module("core.api")


def json_response(payload: Any, status_code: int = 200):
    return JSONResponse(content=payload, status_code=status_code)


def audio_response(audio_buffer, mime_type: str, filename: str):
    audio_buffer.seek(0)
    return Response(
        content=audio_buffer.read(),
        media_type=mime_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


def openai_error(message: str, status_code: int = 400, error_type: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


async def parse_mixed_request(request: Request) -> tuple[dict[str, Any], UploadFile | None]:
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        return payload if isinstance(payload, dict) else {}, None

    form = await request.form()
    payload = {}
    uploaded_audio = None
    for key, value in form.multi_items():
        if isinstance(value, UploadFile) or hasattr(value, "filename"):
            if key in {"prompt_audio", "reference_audio", "ref_audio"} and uploaded_audio is None:
                uploaded_audio = value
        else:
            payload[key] = value
    return payload, uploaded_audio

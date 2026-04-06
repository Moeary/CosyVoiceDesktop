from fastapi import APIRouter

from .common import api_module, json_response

router = APIRouter(tags=["System"])


@router.get("/health", summary="Health check")
@router.get("/api/health", include_in_schema=False)
async def health_check():
    api = api_module()
    try:
        characters = api.character_config.list_characters() if api.character_config else []
        return json_response(
            {
                "status": "ok",
                "model": "CosyVoice3-0.5B",
                "characters": characters,
            }
        )
    except Exception as exc:
        return json_response({"status": "error", "error": str(exc)}, status_code=500)

from fastapi import APIRouter, HTTPException

from app.models.schemas import SettingsUpdate, TestConnectionRequest
from app.services import settings_service as svc

router = APIRouter(prefix="/api/settings")


@router.get("")
def get_settings():
    return svc.get_masked_settings()


@router.put("")
def update_settings(body: SettingsUpdate):
    return svc.update_settings(body.model_dump(exclude_unset=True))


@router.post("/test-connection")
def test_connection(body: TestConnectionRequest):
    current = svc.get_raw_settings()

    if body.target == "neo4j":
        ok, message = svc.test_neo4j(
            body.neo4j_uri or current["neo4j_uri"],
            body.neo4j_user or current["neo4j_user"],
            body.neo4j_password or current["neo4j_password"],
        )
    elif body.target == "llm":
        provider = body.provider or current["provider"]
        api_key = body.api_key or svc.api_key_for_provider(current, provider)
        ok, message = svc.test_llm_provider(
            provider, api_key, body.model or current["model"], body.ollama_host or current["ollama_host"]
        )
    else:
        raise HTTPException(status_code=400, detail="target must be 'neo4j' or 'llm'")

    return {"ok": ok, "message": message}

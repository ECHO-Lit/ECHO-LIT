from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Reflecting a credential back to its sender turns any way of viewing this
# response -- a screenshot, a log, a proxy cache -- into a session handover.
REDACTED_HEADERS = {"cookie", "authorization", "proxy-authorization"}


@router.get("/debug/session")
async def get_session_info(request: Request):
    """Debug endpoint to see current session information"""
    session_id = getattr(request.state, 'sid', None)

    return JSONResponse({
        "session_id": session_id,
        "cookie_names": sorted(request.cookies),
        "headers": {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in REDACTED_HEADERS
        },
    })
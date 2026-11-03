"""
BTP AI Core → OpenAI API 프록시
OpenAI 클라이언트(Claude Code, Cursor 등)에서 SAP BTP AI Core의 GPT 모델을 사용할 수 있도록 변환
"""
import os, time, asyncio, json
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BTP AI Core GPT Proxy")

# ── AI Core 설정 ──────────────────────────────────────────
AICORE_AUTH_URL       = os.environ["AICORE_AUTH_URL"]
AICORE_CLIENT_ID      = os.environ["AICORE_CLIENT_ID"]
AICORE_CLIENT_SECRET  = os.environ["AICORE_CLIENT_SECRET"]
AICORE_BASE_URL       = os.environ["AICORE_BASE_URL"]
AICORE_RESOURCE_GROUP = os.environ.get("AICORE_RESOURCE_GROUP", "default")

# ── 모델 → Deployment ID 매핑 ────────────────────────────
DEPLOYMENT_MAP = {
    "gpt-6.1-sol":   "df19ce85dd38db56",
    "gpt-6.1":       "df19ce85dd38db56",
    "gpt-5.6-terra": "dab75bb61b5e40bb",
    "gpt-5.6-luna":  "d602431584eb9adb",
    "gpt-5.6-sol":   "ddd9854a0e8336e5",
    "gpt-5.6":       "ddd9854a0e8336e5",
    "gpt-5.5":       "d40440b2db1d3929",
    "gpt-5":         "d7edb2f2f9f4a2d0",
    "gpt-4.1":       "d087d107820fbfbc",
    "gpt-4o":        "dfe95f057c8e6182",
    "gpt-4o-mini":   "dfe95f057c8e6182",
}
DEFAULT_DEPLOYMENT = "d7edb2f2f9f4a2d0"  # gpt-5

def get_deployment_id(model: str) -> str:
    if not model:
        return DEFAULT_DEPLOYMENT
    m = model.lower()
    # 정확히 매핑되는 키 우선 (긴 것부터)
    for key in sorted(DEPLOYMENT_MAP.keys(), key=len, reverse=True):
        if key in m:
            return DEPLOYMENT_MAP[key]
    return DEFAULT_DEPLOYMENT

# ── 공유 HTTP 클라이언트 ──────────────────────────────────
_http_client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def startup():
    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=600,
        http2=True,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

@app.on_event("shutdown")
async def shutdown():
    if _refresh_task and not _refresh_task.done():
        _refresh_task.cancel()
    if _http_client:
        await _http_client.aclose()

# ── XSUAA 토큰 캐시 ──────────────────────────────────────
_token_cache = {"token": None, "expires_at": 0}
_token_lock  = asyncio.Lock()
_refresh_task: asyncio.Task | None = None

async def _fetch_token() -> dict:
    resp = await _http_client.post(
        f"{AICORE_AUTH_URL}/oauth/token",
        data={"grant_type": "client_credentials"},
        auth=(AICORE_CLIENT_ID, AICORE_CLIENT_SECRET),
    )
    resp.raise_for_status()
    return resp.json()

def _schedule_background_refresh(expires_at: float):
    global _refresh_task
    if _refresh_task and not _refresh_task.done():
        _refresh_task.cancel()
    _refresh_task = asyncio.create_task(_background_refresh(expires_at))

async def _background_refresh(expires_at: float):
    await asyncio.sleep(max(0, expires_at - time.time() - 120))
    async with _token_lock:
        data = await _fetch_token()
        _token_cache["token"]      = data["access_token"]
        _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
        logger.info("XSUAA 토큰 백그라운드 갱신 완료")
    _schedule_background_refresh(_token_cache["expires_at"])

async def get_access_token() -> str:
    async with _token_lock:
        if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
            return _token_cache["token"]
        data = await _fetch_token()
        _token_cache["token"]      = data["access_token"]
        _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
        _schedule_background_refresh(_token_cache["expires_at"])
        return _token_cache["token"]

# ── 헬스체크 ─────────────────────────────────────────────
@app.get("/healthz")
def health():
    return {"status": "ok"}

# ── 모델 목록 ─────────────────────────────────────────────
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-6.1-sol",   "object": "model", "owned_by": "sap-aicore"},
            {"id": "gpt-5.6-terra", "object": "model", "owned_by": "sap-aicore"},
            {"id": "gpt-5.6-luna",  "object": "model", "owned_by": "sap-aicore"},
            {"id": "gpt-5.6-sol",   "object": "model", "owned_by": "sap-aicore"},
            {"id": "gpt-5.5",       "object": "model", "owned_by": "sap-aicore"},
            {"id": "gpt-5",         "object": "model", "owned_by": "sap-aicore"},
            {"id": "gpt-4.1",       "object": "model", "owned_by": "sap-aicore"},
            {"id": "gpt-4o",        "object": "model", "owned_by": "sap-aicore"},
        ],
    }

# ── Chat Completions 프록시 ───────────────────────────────
@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    start = time.time()
    try:
        body = await request.json()
    except Exception:
        body = {}

    model      = body.get("model", "") or "gpt-5"
    deployment = get_deployment_id(model)
    stream     = body.get("stream", False)
    max_tokens = body.get("max_tokens", "?")

    logger.info(f"[REQ] model={model} deployment={deployment} stream={stream} max_tokens={max_tokens} keys={list(body.keys())}")

    # BTP AI Core는 model 필드 대신 deployment URL로 라우팅하므로 제거
    STRIP_FIELDS = {"model"}
    ai_body = {k: v for k, v in body.items() if k not in STRIP_FIELDS}

    # gpt-6.1-sol 등 reasoning 모델: tools + reasoning_effort 조합 불가
    # reasoning_effort가 있으면 tools 제거, 없으면 그대로
    if "tools" in ai_body and "reasoning_effort" in ai_body:
        logger.info(f"[REQ] tools+reasoning_effort 조합 불가 → tools 제거 (reasoning_effort={ai_body['reasoning_effort']})")
        del ai_body["tools"]
        ai_body.pop("tool_choice", None)

    # GPT: max_tokens → max_completion_tokens 변환
    if "max_tokens" in ai_body:
        ai_body["max_completion_tokens"] = ai_body.pop("max_tokens")

    token = await get_access_token()
    url   = f"{AICORE_BASE_URL}/v2/inference/deployments/{deployment}/v1/chat/completions"

    headers = {
        "Authorization":     f"Bearer {token}",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
        "Content-Type":      "application/json",
    }

    if stream:
        return StreamingResponse(
            _stream_chat(url, headers, ai_body, model, start),
            media_type="text/event-stream",
        )

    resp = await _http_client.post(url, headers=headers, json=ai_body)
    elapsed = time.time() - start

    if resp.status_code != 200:
        logger.error(f"[ERR] status={resp.status_code} elapsed={elapsed:.2f}s body={resp.text[:300]}")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    # model 필드를 요청한 모델명으로 덮어씌우기 (클라이언트 호환성)
    data["model"] = model
    usage = data.get("usage", {})
    logger.info(f"[RES] model={model} elapsed={elapsed:.2f}s "
                f"in={usage.get('prompt_tokens','-')} out={usage.get('completion_tokens','-')}")
    return JSONResponse(content=data)


async def _stream_chat(url, headers, body, model: str, start: float):
    first_chunk = True
    buffer = ""
    async with _http_client.stream("POST", url, headers=headers, json=body) as resp:
        if resp.status_code != 200:
            err_body = await resp.aread()
            logger.error(f"[STREAM ERR] status={resp.status_code} body={err_body.decode()[:500]} request_body={json.dumps(body)[:500]}")
            return
        async for chunk in resp.aiter_text():
            if first_chunk:
                logger.info(f"[STREAM] ttfb={time.time()-start:.2f}s")
                first_chunk = False
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                frame = frame.strip()
                if not frame:
                    continue
                if frame == "data: [DONE]":
                    yield "data: [DONE]\n\n"
                    continue
                if frame.startswith("data:"):
                    try:
                        data_str = frame[5:].strip()
                        parsed = json.loads(data_str)
                        parsed["model"] = model
                        yield f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"
                        continue
                    except Exception:
                        pass
                yield f"{frame}\n\n"
    # 버퍼 잔여 처리
    if buffer.strip():
        frame = buffer.strip()
        if frame == "data: [DONE]":
            yield "data: [DONE]\n\n"
        elif frame.startswith("data:"):
            try:
                parsed = json.loads(frame[5:].strip())
                parsed["model"] = model
                yield f"data: {json.dumps(parsed, ensure_ascii=False)}\n\n"
            except Exception:
                yield f"{frame}\n\n"
        else:
            yield f"{frame}\n\n"
    logger.info(f"[STREAM] done total={time.time()-start:.2f}s")


# ── catch-all ────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def catch_all(request: Request, path: str):
    body = await request.body()
    logger.warning(f"[CATCH-ALL] {request.method} /{path} | body: {body[:200]}")
    return JSONResponse(status_code=200, content={"caught": True, "method": request.method, "path": f"/{path}"})

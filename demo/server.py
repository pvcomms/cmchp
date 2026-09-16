"""CMCHP Demo Server"""
import os
import sys
from pathlib import Path

# Add src to path for local dev
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from cmchp.compressor import Compressor
from cmchp.expander import Expander
from cmchp.schema import CMCHPPacket

app = FastAPI(title="CMCHP Demo", version="0.1.0")

compressor = Compressor(api_key=os.environ.get("ANTHROPIC_API_KEY"))
expander = Expander()


class CompressRequest(BaseModel):
    conversation: list[dict]
    tool_history: list[dict] = []
    source_model: str = "claude-sonnet-4-6"
    target_model: str = "gpt-4o"
    token_budget: int = 4000


class ExpandRequest(BaseModel):
    packet: dict
    target_model: str | None = None


class HandoffRequest(BaseModel):
    packet: dict
    target_model: str
    user_message: str = "Continue the task."


@app.post("/api/compress")
async def compress(req: CompressRequest):
    try:
        packet = compressor.compress(
            conversation=req.conversation,
            tool_history=req.tool_history,
            source_model=req.source_model,
            target_model=req.target_model,
            token_budget=req.token_budget,
        )
        return packet.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/expand")
async def expand(req: ExpandRequest):
    try:
        packet = CMCHPPacket.model_validate(req.packet)
        expanded = expander.expand(packet, req.target_model)
        return {"expanded": expanded, "target_model": req.target_model or packet.target_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/handoff")
async def handoff(req: HandoffRequest):
    try:
        packet = CMCHPPacket.model_validate(req.packet)
        target = req.target_model.lower()

        if "claude" in target:
            from cmchp.models.claude import ClaudeAdapter
            adapter = ClaudeAdapter(api_key=os.environ.get("ANTHROPIC_API_KEY"), model=req.target_model)
        elif "gpt" in target or "o1" in target or "o3" in target or "o4" in target:
            from cmchp.models.openai_adapter import OpenAIAdapter
            adapter = OpenAIAdapter(api_key=os.environ.get("OPENAI_API_KEY"), model=req.target_model)
        elif "gemini" in target:
            from cmchp.models.gemini_adapter import GeminiAdapter
            adapter = GeminiAdapter(api_key=os.environ.get("GEMINI_API_KEY"), model=req.target_model)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model family: {req.target_model}")

        response = adapter.continue_from_packet(packet, req.user_message)
        return {"response": response, "model": req.target_model}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# Serve static files — must be last
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

import os
from datetime import datetime, timezone
from typing import Annotated

import google.genai as genai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from auth import check_rate_limit

load_dotenv()

app = FastAPI()

_db_client: AsyncIOMotorClient | None = None


def get_db():
    global _db_client
    if _db_client is None:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        _db_client = AsyncIOMotorClient(mongo_uri)
    return _db_client["deep_researcher"]


class SearchRequest(BaseModel):
    name: str


class SearchResponse(BaseModel):
    name: str
    summary: str
    stored_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
async def search_person(
    request: SearchRequest,
    _: Annotated[str, Depends(check_rate_limit)],
):
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"Provide a concise financial background summary for {name}. "
            "Include information about their known business ventures, investments, "
            "estimated net worth if publicly available, and any notable financial events. "
            "Only include publicly available information."
        )
        response = client.models.generate_content(
            model="gemini-1.5-flash", contents=prompt
        )
        summary = response.text
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc

    try:
        db = get_db()
        doc = {
            "name": name,
            "summary": summary,
            "created_at": datetime.now(timezone.utc),
        }
        result = await db["searches"].insert_one(doc)
        stored_id = str(result.inserted_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    return SearchResponse(name=name, summary=summary, stored_id=stored_id)



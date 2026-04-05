import json
import os
from datetime import datetime, timezone
from typing import Annotated

import google.genai as genai
from bson import ObjectId
from bson.errors import InvalidId
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


class FinancialProfile(BaseModel):
    overview: str
    net_worth: str
    business_ventures: list[str]
    investments: list[str]
    notable_events: list[str]
    risk_profile: str


class ReportRequest(BaseModel):
    name: str


class ReportResponse(BaseModel):
    name: str
    profile: FinancialProfile
    generated_at: str
    stored_id: str


def _parse_json_response(text: str) -> dict:
    """Parse JSON from a Gemini response, stripping markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


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


_REPORT_PROMPT_TEMPLATE = (
    "Generate a detailed financial profile report for {name} using the following JSON format:\n"
    "{{\n"
    '    "overview": "Brief overview of the person and their financial background",\n'
    '    "net_worth": "Estimated net worth if publicly available, otherwise Not publicly disclosed",\n'
    '    "business_ventures": ["List of known business ventures or companies"],\n'
    '    "investments": ["List of known investments, assets, or holdings"],\n'
    '    "notable_events": ["List of notable financial events, milestones, or controversies"],\n'
    '    "risk_profile": "Assessment of their investment style and risk tolerance"\n'
    "}}\n"
    "Only include publicly available information. Return valid JSON only, no markdown or additional text."
)


@app.post("/api/report", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
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
        prompt = _REPORT_PROMPT_TEMPLATE.format(name=name)
        response = client.models.generate_content(
            model="gemini-1.5-flash", contents=prompt
        )
        profile_data = _parse_json_response(response.text)
        profile = FinancialProfile(**profile_data)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to parse Gemini response: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc

    generated_at = datetime.now(timezone.utc)
    try:
        db = get_db()
        doc = {
            "name": name,
            "profile": profile.model_dump(),
            "generated_at": generated_at,
            "created_at": generated_at,
        }
        result = await db["reports"].insert_one(doc)
        stored_id = str(result.inserted_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    return ReportResponse(
        name=name,
        profile=profile,
        generated_at=generated_at.isoformat(),
        stored_id=stored_id,
    )


@app.get("/api/report/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    _: Annotated[str, Depends(check_rate_limit)],
):
    try:
        oid = ObjectId(report_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid report ID")

    try:
        db = get_db()
        doc = await db["reports"].find_one({"_id": oid})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    if doc is None:
        raise HTTPException(status_code=404, detail="Report not found")

    return ReportResponse(
        name=doc["name"],
        profile=FinancialProfile(**doc["profile"]),
        generated_at=doc["generated_at"].isoformat(),
        stored_id=report_id,
    )



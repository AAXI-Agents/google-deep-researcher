import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auth import reset_rate_limits  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)

_TEST_API_KEY = "test-key-12345"
_TEST_ENV = {"API_KEYS": _TEST_API_KEY, "GEMINI_API_KEY": "test-gemini-key"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _mock_insert_result(inserted_id="507f1f77bcf86cd799439011"):
    result = MagicMock()
    result.inserted_id = inserted_id
    return result


@patch("main.genai.Client")
@patch("main.get_db")
def test_search_person_success(mock_get_db, mock_client_cls):
    mock_response = MagicMock()
    mock_response.text = "Warren Buffett is a renowned investor..."
    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response
    mock_client = MagicMock()
    mock_client.models = mock_models
    mock_client_cls.return_value = mock_client

    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock(return_value=_mock_insert_result())
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db

    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/search",
            json={"name": "Warren Buffett"},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Warren Buffett"
    assert data["summary"] == "Warren Buffett is a renowned investor..."
    assert data["stored_id"] == "507f1f77bcf86cd799439011"


def test_search_person_empty_name():
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/search",
            json={"name": "   "},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_search_person_missing_gemini_key():
    with patch.dict(os.environ, {"API_KEYS": _TEST_API_KEY}, clear=True):
        response = client.post(
            "/api/search",
            json={"name": "Elon Musk"},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 500
    assert "Gemini API key" in response.json()["detail"]


@patch("main.genai.Client")
def test_search_person_gemini_error(mock_client_cls):
    mock_client_cls.side_effect = Exception("API unreachable")

    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/search",
            json={"name": "Jeff Bezos"},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 502
    assert "Gemini API error" in response.json()["detail"]


def test_search_person_missing_name_field():
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/search",
            json={},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


def test_search_missing_api_key_header():
    """Requests without X-API-Key must be rejected with 401."""
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post("/api/search", json={"name": "Someone"})
    assert response.status_code == 401
    assert response.json()["detail"] == "API key required"


def test_search_invalid_api_key():
    """Requests with an unrecognised key must be rejected with 401."""
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/search",
            json={"name": "Someone"},
            headers={"X-API-Key": "wrong-key"},
        )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_health_requires_no_api_key():
    """The /health endpoint must remain publicly accessible."""
    with patch.dict(os.environ, {}, clear=True):
        response = client.get("/health")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate-limiting tests
# ---------------------------------------------------------------------------


def test_search_rate_limit_exceeded():
    """After hitting the configured limit, requests must receive 429."""
    reset_rate_limits()
    rate_env = {**_TEST_ENV, "RATE_LIMIT_REQUESTS": "3", "RATE_LIMIT_WINDOW": "60"}

    mock_response = MagicMock()
    mock_response.text = "Summary text"
    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response
    mock_client = MagicMock()
    mock_client.models = mock_models

    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock(return_value=_mock_insert_result())
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    with (
        patch("main.genai.Client", return_value=mock_client),
        patch("main.get_db", return_value=mock_db),
        patch.dict(os.environ, rate_env, clear=True),
    ):
        # First 3 requests must pass the rate limiter and return 200.
        for _ in range(3):
            r = client.post(
                "/api/search",
                json={"name": "Test"},
                headers={"X-API-Key": _TEST_API_KEY},
            )
            assert r.status_code == 200

        # The 4th request must be rate-limited.
        response = client.post(
            "/api/search",
            json={"name": "Test"},
            headers={"X-API-Key": _TEST_API_KEY},
        )

    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded"
    assert "Retry-After" in response.headers

    reset_rate_limits()


# ---------------------------------------------------------------------------
# Helpers shared by report tests
# ---------------------------------------------------------------------------

_SAMPLE_PROFILE_JSON = """{
    "overview": "Warren Buffett is the chairman of Berkshire Hathaway.",
    "net_worth": "$100 billion",
    "business_ventures": ["Berkshire Hathaway"],
    "investments": ["Apple", "Coca-Cola"],
    "notable_events": ["Berkshire IPO 1965"],
    "risk_profile": "Value investor with long-term horizon"
}"""

_SAMPLE_PROFILE_DICT = {
    "overview": "Warren Buffett is the chairman of Berkshire Hathaway.",
    "net_worth": "$100 billion",
    "business_ventures": ["Berkshire Hathaway"],
    "investments": ["Apple", "Coca-Cola"],
    "notable_events": ["Berkshire IPO 1965"],
    "risk_profile": "Value investor with long-term horizon",
}


def _make_report_mocks(profile_json: str = _SAMPLE_PROFILE_JSON):
    """Return (mock_db, mock_client) configured for a report endpoint call."""
    mock_response = MagicMock()
    mock_response.text = profile_json
    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response
    mock_client = MagicMock()
    mock_client.models = mock_models

    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock(return_value=_mock_insert_result())
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    return mock_db, mock_client


# ---------------------------------------------------------------------------
# POST /api/report – happy path
# ---------------------------------------------------------------------------


@patch("main.genai.Client")
@patch("main.get_db")
def test_generate_report_success(mock_get_db, mock_client_cls):
    mock_db, mock_client = _make_report_mocks()
    mock_get_db.return_value = mock_db
    mock_client_cls.return_value = mock_client

    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/report",
            json={"name": "Warren Buffett"},
            headers={"X-API-Key": _TEST_API_KEY},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Warren Buffett"
    assert data["stored_id"] == "507f1f77bcf86cd799439011"
    profile = data["profile"]
    assert profile["overview"] == _SAMPLE_PROFILE_DICT["overview"]
    assert profile["net_worth"] == "$100 billion"
    assert "Berkshire Hathaway" in profile["business_ventures"]
    assert "Apple" in profile["investments"]
    assert profile["risk_profile"] == _SAMPLE_PROFILE_DICT["risk_profile"]
    assert "generated_at" in data


@patch("main.genai.Client")
@patch("main.get_db")
def test_generate_report_markdown_fenced_json(mock_get_db, mock_client_cls):
    """Gemini sometimes wraps JSON in markdown code fences; strip them."""
    fenced = "```json\n" + _SAMPLE_PROFILE_JSON + "\n```"
    mock_db, mock_client = _make_report_mocks(profile_json=fenced)
    mock_get_db.return_value = mock_db
    mock_client_cls.return_value = mock_client

    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/report",
            json={"name": "Warren Buffett"},
            headers={"X-API-Key": _TEST_API_KEY},
        )

    assert response.status_code == 200
    assert response.json()["profile"]["net_worth"] == "$100 billion"


# ---------------------------------------------------------------------------
# POST /api/report – validation & error cases
# ---------------------------------------------------------------------------


def test_generate_report_empty_name():
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/report",
            json={"name": "   "},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_generate_report_missing_name_field():
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/report",
            json={},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 422


def test_generate_report_missing_gemini_key():
    with patch.dict(os.environ, {"API_KEYS": _TEST_API_KEY}, clear=True):
        response = client.post(
            "/api/report",
            json={"name": "Warren Buffett"},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 500
    assert "Gemini API key" in response.json()["detail"]


@patch("main.genai.Client")
def test_generate_report_gemini_error(mock_client_cls):
    mock_client_cls.side_effect = Exception("API unreachable")

    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/report",
            json={"name": "Warren Buffett"},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 502
    assert "Gemini API error" in response.json()["detail"]


@patch("main.genai.Client")
def test_generate_report_invalid_json_response(mock_client_cls):
    """When Gemini returns non-JSON text, expect 502 with a parse error message."""
    mock_response = MagicMock()
    mock_response.text = "Sorry, I cannot provide that information."
    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response
    mock_client = MagicMock()
    mock_client.models = mock_models
    mock_client_cls.return_value = mock_client

    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post(
            "/api/report",
            json={"name": "Unknown Person"},
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 502
    assert "parse Gemini response" in response.json()["detail"]


def test_generate_report_missing_api_key_header():
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.post("/api/report", json={"name": "Warren Buffett"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/report/{report_id} – happy path
# ---------------------------------------------------------------------------


@patch("main.get_db")
def test_get_report_success(mock_get_db):
    stored_id = "507f1f77bcf86cd799439011"
    generated_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": stored_id,
            "name": "Warren Buffett",
            "profile": _SAMPLE_PROFILE_DICT,
            "generated_at": generated_at,
            "created_at": generated_at,
        }
    )
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db

    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.get(
            f"/api/report/{stored_id}",
            headers={"X-API-Key": _TEST_API_KEY},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Warren Buffett"
    assert data["stored_id"] == stored_id
    assert data["profile"]["net_worth"] == "$100 billion"
    assert data["generated_at"] == generated_at.isoformat()


# ---------------------------------------------------------------------------
# GET /api/report/{report_id} – error cases
# ---------------------------------------------------------------------------


@patch("main.get_db")
def test_get_report_not_found(mock_get_db):
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=None)
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db

    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.get(
            "/api/report/507f1f77bcf86cd799439011",
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_report_invalid_id():
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.get(
            "/api/report/not-a-valid-object-id",
            headers={"X-API-Key": _TEST_API_KEY},
        )
    assert response.status_code == 400
    assert "Invalid report ID" in response.json()["detail"]


def test_get_report_missing_api_key_header():
    with patch.dict(os.environ, _TEST_ENV, clear=True):
        response = client.get("/api/report/507f1f77bcf86cd799439011")
    assert response.status_code == 401


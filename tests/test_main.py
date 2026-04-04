import os
import sys
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


import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def _mock_insert_result(inserted_id="507f1f77bcf86cd799439011"):
    result = MagicMock()
    result.inserted_id = inserted_id
    return result


@patch("main.os.getenv")
@patch("main.genai.Client")
@patch("main.get_db")
def test_search_person_success(mock_get_db, mock_client_cls, mock_getenv):
    mock_getenv.side_effect = lambda key, default=None: (
        "test-api-key" if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") else default
    )

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

    response = client.post("/api/search", json={"name": "Warren Buffett"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Warren Buffett"
    assert data["summary"] == "Warren Buffett is a renowned investor..."
    assert data["stored_id"] == "507f1f77bcf86cd799439011"


def test_search_person_empty_name():
    response = client.post("/api/search", json={"name": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


@patch("main.os.getenv", return_value=None)
def test_search_person_missing_api_key(mock_getenv):
    response = client.post("/api/search", json={"name": "Elon Musk"})
    assert response.status_code == 500
    assert "Gemini API key" in response.json()["detail"]


@patch("main.os.getenv")
@patch("main.genai.Client")
def test_search_person_gemini_error(mock_client_cls, mock_getenv):
    mock_getenv.side_effect = lambda key, default=None: (
        "test-api-key" if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") else default
    )
    mock_client_cls.side_effect = Exception("API unreachable")

    response = client.post("/api/search", json={"name": "Jeff Bezos"})
    assert response.status_code == 502
    assert "Gemini API error" in response.json()["detail"]


def test_search_person_missing_name_field():
    response = client.post("/api/search", json={})
    assert response.status_code == 422


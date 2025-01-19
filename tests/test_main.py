import pytest
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app, BASE_SYSTEM_PROMPT
from unittest.mock import patch, MagicMock
import json

client = TestClient(app)

@pytest.fixture
def mock_github_user():
    return {
        "login": "testuser",
        "id": 12345
    }

@pytest.fixture
def mock_ingest():
    with patch("app.main.ingest") as mock_ingest:
        mock_ingest.return_value = (
            "Test repository summary",
            "test/file/structure",
            "test content"
        )
        yield mock_ingest

def test_set_repo_command(mock_github_user, mock_ingest):
    # Mock GitHub API response
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_github_user
        )
        
        # Create test payload
        payload = {
            "copilot_thread_id": "test-thread-123",
            "messages": [
                {
                    "role": "user",
                    "content": "/set https://github.com/cyclotruc/gitingest"
                }
            ]
        }
        
        # Make request with test client
        response = client.post(
            "/",
            json=payload,
            headers={}  # No auth header needed in test mode
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify response structure
        assert response_data["status"] == "success"
        assert len(response_data["messages"]) > 0
        
        # Verify system message was updated
        system_messages = [msg for msg in response_data["messages"] if msg["role"] == "system"]
        assert len(system_messages) > 0
        assert BASE_SYSTEM_PROMPT in system_messages[0]["content"]
        assert "Test repository summary" in system_messages[0]["content"]
        
        # Verify ingest was called
        mock_ingest.assert_called_once_with("https://github.com/cyclotruc/gitingest")

def test_set_repo_invalid_url(mock_github_user):
    # Mock GitHub API response
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_github_user
        )
        
        # Create test payload with invalid URL
        payload = {
            "copilot_thread_id": "test-thread-123",
            "messages": [
                {
                    "role": "user",
                    "content": "/set not-a-valid-url"
                }
            ]
        }
        
        # Make request with test client
        response = client.post(
            "/",
            json=payload,
            headers={}  # No auth header needed in test mode
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify response structure
        assert response_data["status"] == "success"
        assert len(response_data["messages"]) > 0
        
        # Verify error message
        assistant_messages = [msg for msg in response_data["messages"] if msg["role"] == "assistant"]
        assert len(assistant_messages) > 0
        assert "Error setting repository URL" in assistant_messages[0]["content"]

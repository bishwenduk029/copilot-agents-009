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
            headers={"X-GitHub-Token": "test-token"}
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.content)
        
        # Verify system message was updated
        assert BASE_SYSTEM_PROMPT in response_data["choices"][0]["message"]["content"]
        assert "Test repository summary" in response_data["choices"][0]["message"]["content"]
        assert "test/file/structure" in response_data["choices"][0]["message"]["content"]
        
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
            headers={"X-GitHub-Token": "test-token"}
        )
        
        # Verify response
        assert response.status_code == 200
        response_data = json.loads(response.content)
        
        # Verify error message
        assert "Error setting repository URL" in response_data["choices"][0]["message"]["content"]

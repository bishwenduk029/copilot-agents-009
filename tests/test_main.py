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
    # Clear cache before test
    from app.main import cache
    cache.clear()
    
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
        
        # Print the system message for validation
        print("\nFinal System Message:")
        print(system_messages[0]["content"])
        print("-" * 80)
        
        assert BASE_SYSTEM_PROMPT in system_messages[0]["content"]
        assert "Test repository summary" in system_messages[0]["content"]
        
        # Verify ingest was called and print mock return values
        mock_ingest.assert_called_once_with("https://github.com/cyclotruc/gitingest")
        print("\nMock Ingest Return Values:")
        print(f"Summary: {mock_ingest.return_value[0]}")
        print(f"File Tree: {mock_ingest.return_value[1]}")
        print(f"Content: {mock_ingest.return_value[2]}")
        print("-" * 80)

def test_set_repo_invalid_url(mock_github_user):
    """Test handling of invalid repository URLs"""
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

@pytest.mark.asyncio
async def test_set_repo_live_gitingest(mock_github_user):
    """Test actual gitingest integration with a real repository"""
    # Clear cache before test
    from app.main import cache
    cache.clear()
    
    # Use a small, public test repository
    test_repo_url = "https://github.com/octocat/Hello-World"
    
    # Create test payload
    payload = {
        "copilot_thread_id": "test-thread-123",
        "messages": [
            {
                "role": "user",
                "content": f"/set {test_repo_url}"
            }
        ]
    }
    
    try:
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
        
        # Print the system message for validation
        print("\nLive System Message:")
        print(system_messages[0]["content"])
        print("-" * 80)
        
        # Verify base prompt is present
        assert BASE_SYSTEM_PROMPT in system_messages[0]["content"]
        
        # Verify we got some actual repository context
        assert "Current repository context" in system_messages[0]["content"]
        assert "Summary:" in system_messages[0]["content"]
        assert "File Tree:" in system_messages[0]["content"]
        
        # Print actual repository data
        repo_data = system_messages[0]["content"].split("Current repository context:")[1]
        print("\nActual Repository Data:")
        print(repo_data)
        print("-" * 80)
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_cached_ingest_async_handling(mock_github_user):
    """Test async handling of cached_ingest function"""
    from app.main import cached_ingest, cache
    
    # Clear cache before test
    cache.clear()
    
    test_repo_url = "https://github.com/test/repo"
    test_data = ("Test summary", "test/file/structure", "test content")
    
    # Mock the thread pool execution
    with patch("asyncio.get_event_loop") as mock_loop, \
         patch("app.main.ingest") as mock_ingest:
        
        # Setup mock executor
        mock_executor = MagicMock()
        mock_loop.return_value.run_in_executor.return_value = test_data
        mock_ingest.return_value = test_data
        
        # Call cached_ingest
        result = await cached_ingest(test_repo_url)
        
        # Verify results
        assert result == test_data
        assert test_repo_url in cache
        
        # Verify thread pool was used
        mock_loop.return_value.run_in_executor.assert_called_once()
        
        # Verify cache was set
        cached_result = cache.get(test_repo_url)
        assert cached_result == test_data
        
        # Verify cache hit on second call
        result2 = await cached_ingest(test_repo_url)
        assert result2 == test_data

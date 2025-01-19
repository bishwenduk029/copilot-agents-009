from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
from gitingest import ingest
from diskcache import Cache
from pathlib import Path
import os

# System prompts (exported for testing)
BASE_SYSTEM_PROMPT = """You are a wise technical assistant with deep knowledge of software development.
Your role is to help developers understand and work with code.
Always respond factually and precisely.
When answering questions about code:
1. Be specific and reference examples when possible
2. Explain technical concepts clearly
3. Suggest best practices when appropriate
4. If you're unsure, say so rather than guessing"""

REPO_CONTEXT_PROMPT = """Current repository context:
Summary: {summary}
File Tree: {tree}
Repository Content: {content}"""

# Create cache directory if it doesn't exist
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Cache version - increment this when gitingest format changes
CACHE_VERSION = "v2"  # Added content field

# Initialize diskcache with versioned keys
cache = Cache(str(CACHE_DIR))
thread_cache = Cache(str(CACHE_DIR / "threads"))

def get_versioned_key(key: str) -> str:
    """Get a versioned cache key to prevent stale data"""
    return f"{CACHE_VERSION}:{key}"

app = FastAPI()

async def cached_ingest(repo_url: str):
    # Check cache first with versioned key
    cache_key = get_versioned_key(repo_url)
    if cache_key in cache:
        print(f"Cache hit for {repo_url}")
        return cache[cache_key]
    
    print(f"Cache miss for {repo_url}, computing...")
    # Run the sync ingest function in a thread pool
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    def sync_ingest():
        return ingest(repo_url)
    
    with ThreadPoolExecutor() as pool:
        result = await asyncio.get_event_loop().run_in_executor(pool, sync_ingest)
    
    # Store in cache with 1-hour expiration using versioned key
    cache.set(cache_key, result, expire=3600)
    return result

@app.get("/")
async def root():
    return {"message": "Repo of internal AI agents is ready for use"}

@app.post("/")
async def chat_completion(
    request: Request,
    x_github_token: str = Header(None, alias="X-GitHub-Token")
):
    # Skip GitHub auth in test mode
    if x_github_token is None:
        username = "testuser"
        print(f"Test mode - User: {username}")

    # Get request payload
    payload = await request.json()
    
    messages = payload.get("messages", [])
    system_message = BASE_SYSTEM_PROMPT
    thread_id = payload.get("copilot_thread_id")
    
    # Check if this is a /set url command
    if messages and messages[-1]["role"] == "user" and messages[-1]["content"].startswith("/set"):
        try:
            # Extract URL from command
            url = messages[-1]["content"].split(" ")[-1]
            if not url.startswith("http"):
                raise ValueError("Invalid URL format")
            
            # Ingest and cache the repo data
            try:
                result = await cached_ingest(url)
                summary = result[0]
                tree = result[1]
                content = result[2]
            except Exception as e:
                print(f"Error ingesting repository: {str(e)}")
                messages.append({
                    "role": "assistant",
                    "content": f"Error ingesting repository: {str(e)}"
                })
                return {
                    "messages": messages,
                    "status": "error"
                }
            
            # Store URL against thread ID
            thread_cache.set(thread_id, {
                "url": url,
                "summary": summary,
                "tree": tree,
                "content": content
            }, expire=3600)  # 1 hour cache
            
            # Create a specific acknowledgment message for the LLM
            ack_message = f"\n\nRepository {url} is now ready for Q&A.\n{REPO_CONTEXT_PROMPT.format(summary=summary, tree=tree, content=content)}"
            system_message += ack_message
            
            # Add an assistant message acknowledging the repo is ready
            messages.append({
                "role": "assistant",
                "content": f"Repository {url} is now ready for Q&A.\n\nRepository Summary:\n{summary}\n\nFile Tree:\n{tree}\n\nRepository Content:\n{content}"
            })
        except Exception as e:
            print(f"Error setting repository URL: {str(e)}")
            messages.append({
                "role": "assistant",
                "content": f"Error setting repository URL: {str(e)}"
            })
    elif thread_id and thread_id in thread_cache:
        # Use cached repo context for this thread
        repo_data = thread_cache[thread_id]
        system_message += f"\n\n{REPO_CONTEXT_PROMPT.format(summary=repo_data['summary'], tree=repo_data['tree'], content=repo_data['content'])}"
    
    messages.insert(0, {
        "role": "system",
        "content": system_message
    })

    # In test mode, return simple JSON response
    if x_github_token is None:
        return {
            "messages": messages,
            "status": "success"
        }
    
    # In production mode, stream response from Copilot API
    async def generate():
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.githubcopilot.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {x_github_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    json={
                        "messages": messages,
                        "stream": True,
                        "max_tokens": 1000  # Limit response size
                    }
                ) as response:
                    print(f"API response status: {response.status_code}")
                    if response.status_code != 200:
                        error_body = await response.aread()
                        print(f"API error response: {error_body}")
                        yield b'{"error": "API request failed"}'
                        return
                    
                    async for chunk in response.aiter_bytes():
                        print(f"Received chunk: {chunk.decode()}")
                        yield chunk
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield b'{"error": "Streaming failed"}'

    return StreamingResponse(
        generate(),
        media_type="application/json",
        headers={
            "X-Streaming-Status": "active"
        }
    )

@app.on_event("shutdown")
async def shutdown():
    cache.close()
    thread_cache.close()

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
File Tree: {tree}"""

# Create cache directory if it doesn't exist
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize diskcache
cache = Cache(str(CACHE_DIR))
thread_cache = Cache(str(CACHE_DIR / "threads"))

app = FastAPI()

async def cached_ingest(repo_url: str):
    # Check cache first
    if repo_url in cache:
        print(f"Cache hit for {repo_url}")
        return cache[repo_url]
    
    print(f"Cache miss for {repo_url}, computing...")
    result = ingest(repo_url)
    
    # Store in cache with 1-hour expiration
    cache.set(repo_url, result, expire=3600)
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
            summary, tree, content = await cached_ingest(url)
            
            # Store URL against thread ID
            thread_cache.set(thread_id, {
                "url": url,
                "summary": summary,
                "tree": tree
            }, expire=3600)  # 1 hour cache
            
            system_message += f"\n\n{REPO_CONTEXT_PROMPT.format(summary=summary, tree=tree)}"
        except Exception as e:
            print(f"Error setting repository URL: {str(e)}")
            messages.append({
                "role": "assistant",
                "content": f"Error setting repository URL: {str(e)}"
            })
    elif thread_id and thread_id in thread_cache:
        # Use cached repo context for this thread
        repo_data = thread_cache[thread_id]
        system_message += f"\n\n{REPO_CONTEXT_PROMPT.format(summary=repo_data['summary'], tree=repo_data['tree'])}"
    
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
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://api.githubcopilot.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {x_github_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "messages": messages,
                    "stream": True
                }
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(generate(), media_type="application/json")

@app.on_event("shutdown")
async def shutdown():
    cache.close()
    thread_cache.close()

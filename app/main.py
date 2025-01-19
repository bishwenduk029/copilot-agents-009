from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
from gitingest import ingest
from diskcache import Cache
from pathlib import Path
import os

# Create cache directory if it doesn't exist
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize diskcache
cache = Cache(str(CACHE_DIR))

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
    x_github_token: str = Header(..., alias="X-GitHub-Token")
):
    # Get GitHub user info
    async with httpx.AsyncClient() as client:
        try:
            user_response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {x_github_token}"}
            )
            user_response.raise_for_status()
            user_data = user_response.json()
            username = user_data["login"]
            print(f"User: {username}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="GitHub API error")

    # Get request payload
    payload = await request.json()
    print("Payload:", payload)

    # Add repository context if URL is provided
    if "repo_url" in payload:
        try:
            # Use cached ingest
            summary, tree, content = await cached_ingest(payload["repo_url"])
            
            # Add repository context to messages
            messages = payload.get("messages", [])
            messages.insert(0, {
                "role": "system",
                "content": f"Start every response with the user's name, which is @{username}"
            })
            messages.insert(0, {
                "role": "system",
                "content": f"Repository context:\nSummary: {summary}\nFile Tree: {tree}\n\nYou are a helpful assistant that replies to user messages as if you were the Blackbeard Pirate."
            })
        except Exception as e:
            print(f"Error ingesting repository: {str(e)}")
            messages = payload.get("messages", [])
            messages.insert(0, {
                "role": "system",
                "content": f"Start every response with the user's name, which is @{username}"
            })
            messages.insert(0, {
                "role": "system", 
                "content": "You are a helpful assistant that replies to user messages as if you were the Blackbeard Pirate."
            })
    else:
        messages = payload.get("messages", [])
        messages.insert(0, {
            "role": "system",
            "content": f"Start every response with the user's name, which is @{username}"
        })
        messages.insert(0, {
            "role": "system", 
            "content": "You are a helpful assistant that replies to user messages as if you were the Blackbeard Pirate."
        })

    # Stream response from Copilot API
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

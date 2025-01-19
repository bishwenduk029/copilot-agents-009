from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
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

Note: The full repository content is available but not shown here to conserve tokens. 
Ask specific questions about files or code sections and I'll retrieve the relevant parts.
"""

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

async def cached_ingest(repo_url: str, include_patterns: list[str] | None = None):
    # Check cache first with versioned key
    cache_key = get_versioned_key(repo_url)
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        print(f"Cache hit for {repo_url}")
        if not isinstance(cached_data, tuple) or len(cached_data) != 3:
            # Invalid cache entry - delete it
            cache.delete(cache_key)
            raise ValueError("Invalid cache format - clearing entry")
        return cached_data
    
    print(f"Cache miss for {repo_url}, computing...")
    # Run the sync ingest function in a thread pool
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    def sync_ingest():
        return ingest(repo_url, include_patterns=include_patterns)
    
    with ThreadPoolExecutor() as pool:
        result = await asyncio.get_event_loop().run_in_executor(pool, sync_ingest)
    
    # Store in cache with 1-hour expiration using versioned key
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("Invalid ingest result format")
    
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
                # Only include code files by default
                result = await cached_ingest(
                    url,
                    include_patterns=["*.py", "*.kt", "*.kts", "*.java", "*.gradle", "*.md", "*.json", "*.yaml", "*.yml", "*.ts"]
                )
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
            ack_message = f"\n\nRepository {url} is now ready for Q&A.\n{REPO_CONTEXT_PROMPT.format(summary=summary, tree=tree)}"
            system_message += ack_message
            
            # Add an assistant message acknowledging the repo is ready
            messages.append({
                "role": "assistant",
                "content": f"Repository {url} is now ready for Q&A.\n\nRepository Summary:\n{summary}\n\nFile Tree:\n{tree}"
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
    
    # Define available tools with thread context
    tools = []
    if thread_id:
        tools.append({
            "type": "function",
            "function": {
                "name": f"navigate_repository_content:{thread_id}",
                "description": "Navigate the ingested repository content to find specific information. Use this tool when asked deep technical questions about the codebase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The specific information being searched for in the repository"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional specific file path to search within"
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    # In production mode, stream response from Copilot API
    async def process_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
        """Process tool calls and return assistant messages with results"""
        messages = []
        for tool_call in tool_calls:
            if tool_call["function"]["name"].startswith("navigate_repository_content"):
                result = await execute_repo_navigation_tool(FunctionCall(
                    name=tool_call["function"]["name"],
                    arguments=tool_call["function"]["arguments"]
                ))
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call["id"]
                })
        return messages

    async def generate():
        # Add tools to the request if this is the first iteration
        use_tools = bool(tools)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                request_data = {
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 1000
                }
                
                if use_tools:
                    # Format tools according to GitHub Copilot API spec
                    request_data["tools"] = [{
                        "type": "function",
                        "function": {
                            "name": tool.function["name"],
                            "description": tool.function["description"],
                            "parameters": tool.function["parameters"]
                        }
                    } for tool in tools]
                    # Set tool_choice to none initially, let the model decide
                    request_data["tool_choice"] = None
                
                async with client.stream(
                    "POST",
                    "https://api.githubcopilot.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {x_github_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    json=request_data
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        print(f"API error response: {error_body}")
                        yield b'{"error": "API request failed"}'
                        return
                    
                    async for chunk in response.aiter_bytes():
                        # Check if the chunk contains tool calls
                        chunk_str = chunk.decode('utf-8')
                        if '"tool_calls":' in chunk_str:
                            # Parse the tool calls and process them
                            try:
                                import json
                                data = json.loads(chunk_str)
                                if data.get("choices") and data["choices"][0].get("message", {}).get("tool_calls"):
                                    tool_calls = data["choices"][0]["message"]["tool_calls"]
                                    tool_messages = await process_tool_calls(tool_calls)
                                    # Add tool responses to messages and make a new request
                                    messages.extend([msg.dict() for msg in tool_messages])
                                    # Update request data with new messages
                                    request_data["messages"] = messages
                                    # Continue streaming with updated messages
                                    continue
                            except Exception as e:
                                print(f"Error processing tool calls: {str(e)}")
                        
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

# Tool call processing types
class ToolCall(BaseModel):
    id: str
    type: str
    function: Dict[str, Any]

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ChatMessage(BaseModel):
    role: str
    content: str
    tool_call_id: Optional[str] = None

async def execute_repo_navigation_tool(function_call: FunctionCall) -> str:
    """Execute the repository navigation tool to search repository content"""
    from typing import Optional
    import json
    
    try:
        # Parse the function arguments
        args = json.loads(function_call.arguments)
        query = args.get("query", "").lower()
        file_path = args.get("file_path")
        
        # Get the cached repo data for this thread
        thread_id = function_call.name.split(":")[-1] if ":" in function_call.name else None
        if not thread_id or thread_id not in thread_cache:
            return "Error: No repository context available"
            
        repo_data = thread_cache[thread_id]
        content = repo_data.get("content", "")
        
        # Parse the content digest
        files = []
        current_file = None
        
        for line in content.splitlines():
            if line.startswith("===") and "File:" in line:
                # Start new file section
                if current_file:
                    files.append(current_file)
                file_path = line.split("File:")[1].strip()
                current_file = {
                    "path": file_path,
                    "content": ""
                }
            elif current_file:
                current_file["content"] += line + "\n"
        
        if current_file:
            files.append(current_file)
            
        # Search logic
        results = []
        
        for file in files:
            # Skip if specific file path was requested and doesn't match
            if file_path and file["path"].lower() != file_path.lower():
                continue
                
            # Check if query matches file path or content
            if query in file["path"].lower() or query in file["content"].lower():
                # Include first 200 chars and last 200 chars of content
                content_preview = file["content"][:200] + "..." + file["content"][-200:]
                results.append({
                    "file": file["path"],
                    "content_preview": content_preview.strip()
                })
                
                # Limit to 5 results to avoid overwhelming the LLM
                if len(results) >= 5:
                    break
                    
        if not results:
            return "No matching files found"
            
        # Format results for LLM
        response = "Found matching files:\n"
        for result in results:
            response += f"\nFile: {result['file']}\n"
            response += f"Preview: {result['content_preview']}\n"
            response += "---\n"
            
        return response
        
    except Exception as e:
        return f"Error executing repository navigation: {str(e)}"

@app.on_event("shutdown")
async def shutdown():
    cache.close()
    thread_cache.close()

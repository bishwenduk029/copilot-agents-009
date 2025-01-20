from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
import json
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

You have access to tools that can retrieve specific file contents from the repository.
When answering questions about code:
1. First check if you need to retrieve specific file contents using the navigate_repository_content tool
2. Be specific and reference examples when possible
3. Explain technical concepts clearly
4. Suggest best practices when appropriate
5. If you're unsure, say so rather than guessing

To use the tools:
1. Carefully examine the repository file tree above to identify relevant files
2. Verify the exact file path exists in the file tree before requesting it
3. Use the navigate_repository_content tool with:
   - file_path: exact path to the file as shown in the file tree
4. The tool will return matching file contents that you can use in your response

Important rules:
- Always check the file tree first to confirm the file exists
- Use the exact file path as shown in the file tree
- Never guess file paths - if you can't find it in the tree, ask for clarification"""

REPO_CONTEXT_PROMPT = """Current repository context:
Summary: {summary}
File Tree: {tree}

The full repository content is available through the navigate_repository_content tool.
When you need specific file contents:
1. Check the file tree above to identify relevant files
2. Use the tool to retrieve exact file contents
3. Use the retrieved contents to provide precise answers
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

async def cached_ingest(repo_url: str):
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
        return ingest(repo_url)
    
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
    
    # Handle /set command separately - just ingest and ack
    if messages and messages[-1]["role"] == "user" and messages[-1]["content"].startswith("/set"):
        try:
            # Extract URL from command
            url = messages[-1]["content"].split(" ")[-1]
            if not url.startswith("http"):
                raise ValueError("Invalid URL format")
            
            # Ingest and cache the repo data
            result = await cached_ingest(url)
            summary = result[0]
            tree = result[1]
            content = result[2]
            
            # Store URL against thread ID
            thread_cache.set(thread_id, {
                "url": url,
                "summary": summary,
                "tree": tree,
                "content": content
            }, expire=3600)  # 1 hour cache
            
            # Create ack message and return via SSE
            ack_message = f"Repository {url} is now ready for Q&A.\n\nRepository Summary:\n{summary}\n\nFile Tree:\n{tree}"
            messages.append({
                "role": "assistant",
                "content": ack_message
            })
            
            # Return streaming response with tools disabled
            return StreamingResponse(
                generate(),
                media_type="application/json",
                headers={
                    "X-Streaming-Status": "active"
                }
            )
        except Exception as e:
            print(f"Error setting repository URL: {str(e)}")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"Error setting repository URL: {str(e)}"
                }],
                "status": "error"
            }
    
    # For non-/set commands, use cached repo context if available
    if thread_id and thread_id in thread_cache:
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
        print(f"\n=== Auth Token ===")
        print(f"Token present: {'YES' if x_github_token else 'NO'}")
        
        tools = [{
            "type": "function",
            "function": {
                "name": "navigate_repository_content",
                "description": "Retrieve the contents of a specific file from the repository. Use this when you need to see the contents of a particular file to answer a question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string", 
                            "description": "The full path to the file to retrieve"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }]

    # Print tools configuration
    print(f"\n=== Tools Configuration ===")
    print(f"Tools enabled: {'YES' if tools else 'NO'}")
    if tools:
        print("Available tools:")
        for tool in tools:
            print(f"- {tool['function']['name']}")
            print(f"  Description: {tool['function']['description']}")
    
    # In production mode, stream response from Copilot API
    async def process_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
        """Process tool calls and return assistant messages with results"""
        messages = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            if function.get("name") == "navigate_repository_content":
                # Add thread ID to arguments
                args = json.loads(function.get("arguments", "{}"))
                args["thread_id"] = thread_id
                
                result = await execute_repo_navigation_tool(FunctionCall(
                    name=function.get("name", ""),
                    arguments=json.dumps(args)
                ))
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.get("id", "")
                })
        return messages

    async def generate():
        max_iterations = 3
        current_iteration = 0
        # Disable tools for /set command responses
        use_tools = bool(tools) and not any(
            msg["content"].startswith("/set") 
            for msg in messages 
            if msg["role"] == "user"
        )
        # Create local copy of messages to avoid scope issues
        local_messages = messages.copy()
        
        async def make_api_request(client, messages, use_tools, stream=False):
            """Make API request and handle response"""
            print(f"\n=== API Request ===")
            print(f"Using tools: {'YES' if use_tools else 'NO'}")
            print(f"Tool choice: {'required' if use_tools else 'none'}")
            print(f"Streaming: {'YES' if stream else 'NO'}")
            print(f"Token: {x_github_token}")
            
            request_data = {
                "messages": messages,
                "stream": stream,
                "tools": tools if use_tools else None,
                "tool_choice": "required" if use_tools else None,
                "model": "gpt-4o"
            }
            
            if stream:
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
                        print("\n=== API Error Details ===")
                        print(f"Status Code: {response.status_code}")
                        print("Response Headers:", dict(response.headers))
                        print("Response Body:", error_body)
                        print("=======================\n")
                        yield b'{"error": "API request failed"}'
                        return
                    
                    async for chunk in response.aiter_bytes():
                        yield chunk
            else:
                response = await client.post(
                    "https://api.githubcopilot.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {x_github_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    json=request_data
                )
                
                if response.status_code != 200:
                    print("\n=== API Error Details ===")
                    print(f"Status Code: {response.status_code}")
                    print("Response Headers:", dict(response.headers))
                    print("Response Body:", response.text)
                    print("=======================\n")
                    yield b'{"error": "API request failed"}'
                    return
                
                yield response.content

        async def handle_tool_calls(client, messages):
            """Handle tool calls and return updated messages"""
            nonlocal current_iteration, use_tools
            
            print(f"\n=== Tool Call Iteration {current_iteration + 1} ===")
            
            # Make non-streaming API request for tool calls
            async for chunk in make_api_request(client, messages, use_tools, stream=False):
                try:
                    # Parse the complete response
                    data = json.loads(chunk)
                    print(f"\n=== Complete Response ===")
                    print(json.dumps(data, indent=2))
                    
                    if data.get("choices"):
                        choice = data["choices"][0]
                        finish_reason = choice.get("finish_reason")
                        
                        message = choice.get("message", {})
                        tool_calls = message.get("tool_calls", [])
                        
                        if tool_calls:
                            print(f"\n=== Tool Calls Found ===")
                            print(f"Number of tool calls: {len(tool_calls)}")
                            
                            for i, tool_call in enumerate(tool_calls):
                                print(f"\nTool Call {i+1}:")
                                print(f"ID: {tool_call['id']}")
                                print(f"Type: {tool_call['type']}")
                                print(f"Function: {tool_call['function']['name']}")
                                print(f"Arguments: {tool_call['function']['arguments']}")
                            
                            # Process tool calls
                            tool_messages = await process_tool_calls(tool_calls)
                            messages.extend(tool_messages)
                            
                            # Update iteration state
                            current_iteration += 1
                            if current_iteration >= max_iterations - 1:
                                use_tools = False
                                print("Final iteration - disabling tools")
                            
                            return messages
                        elif finish_reason == "stop":
                            # No tool calls, just return the assistant message
                            messages.append({
                                "role": "assistant",
                                "content": message.get("content", "")
                            })
                            return messages
                        
                except json.JSONDecodeError as e:
                    print(f"\n=== JSON Decode Error ===")
                    print(f"Error: {str(e)}")
                    print(f"Chunk: {chunk}")
                    return messages
                    
            return messages

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                while current_iteration < max_iterations:
                    # Handle tool calls if needed
                    if use_tools:
                        # Store previous iteration count to detect if we made progress
                        prev_iteration = current_iteration
                        local_messages = await handle_tool_calls(client, local_messages)
                        
                        # If we didn't process any tool calls, break to avoid infinite loop
                        if current_iteration == prev_iteration:
                            print("No tool calls processed - breaking loop")
                            break
                            
                        if not use_tools:
                            # Make final streaming request without tools
                            async for chunk in make_api_request(client, local_messages, False, stream=True):
                                yield chunk
                            return
                    else:
                        # Make final streaming request without tools
                        async for chunk in make_api_request(client, local_messages, False, stream=True):
                            yield chunk
                        return
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield b'{"error": "Streaming failed"}'
            return

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
    """Execute the repository navigation tool to retrieve file contents"""
    from typing import Optional
    import json
    
    try:
        # Parse the function arguments
        args = json.loads(function_call.arguments)
        file_path = args.get("file_path")
        
        if not file_path:
            return "Error: No file path specified"
            
        # Get the cached repo data for this thread
        thread_id = function_call.name.split(":")[-1] if ":" in function_call.name else None
        if not thread_id or thread_id not in thread_cache:
            return "Error: No repository context available"
            
        repo_data = thread_cache[thread_id]
        content = repo_data.get("content", "")
            
        # Print debug info
        print("\n=== Repository Context ===")
        print(f"Thread ID: {thread_id}")
        print(f"Repository URL: {repo_data.get('url')}")
        print("\nRepository Summary:")
        print(repo_data.get('summary', ''))
        print("\nRepository Tree:")
        print(repo_data.get('tree', ''))
        print("\nRepository Content Preview:")
        print(content[:1000] + "..." if len(content) > 1000 else content)
        
        # Parse the content digest
        files = []
        current_file = None
        
        for line in content.splitlines():
            if line.startswith("===") and "File:" in line:
                # Start new file section
                if current_file:
                    files.append(current_file)
                current_file_path = line.split("File:")[1].strip()
                current_file = {
                    "path": current_file_path,
                    "content": ""
                }
            elif current_file:
                current_file["content"] += line + "\n"
        
        if current_file:
            files.append(current_file)
            
        # Find exact file match
        found_file = None
        for file in files:
            if file["path"].lower() == file_path.lower():
                found_file = file
                break
                
        if not found_file:
            return f"File not found: {file_path}"
            
        # Return full file contents
        return f"Contents of {found_file['path']}:\n\n{found_file['content']}"
        
    except Exception as e:
        return f"Error executing repository navigation: {str(e)}"

@app.on_event("shutdown")
async def shutdown():
    cache.close()
    thread_cache.close()

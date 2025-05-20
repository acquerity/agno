import time
import jwt
from fastapi import FastAPI, HTTPException, Depends, Header, Body, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from libs.agno.agno.agent.super_agent import SuperAgent
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, Dict
from collections import defaultdict
import json

SECRET_KEY = "SECRET_KEY"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60 * 24  # 1 day

# In-memory user store: {username: password}
users = {}

# In-memory advanced graph for each user: {username: {"nodes": [], "edges": []}}
advanced_graph_memory = defaultdict(lambda: {"nodes": [], "edges": []})

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all. For prod, restrict to Neurite's domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the SuperAgent
agent = SuperAgent()

# --- DEV USER AND API KEYS SETUP ---
DEV_USERNAME = "devuser"
DEV_PASSWORD = "SuperSecureDevPassword123!"
DEV_API_KEYS = {
    "claude": "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",
    "grok": "xai-jqeDmX7zXK17BXBrIe4Up3OeWmcogWMakAOnZnu7AGZKoNOpxUVElIIh64S5VahqmtJ6gJz4hWUEHdD7",
    "openai": "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",
    "google": "AIzaSyAT-EXdHmVfoPxwlDCPQN7Z4SG-Sdyz6v8",
}
# Add dev user to in-memory user store and protect API keys by login
users[DEV_USERNAME] = DEV_PASSWORD
user_api_keys = {DEV_USERNAME: DEV_API_KEYS}

class AgentRequest(BaseModel):
    prompt: str
    session_id: str = "default"

class AgentResponse(BaseModel):
    response: str

class UserAuthRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class APIKeysRequest(BaseModel):
    claude: str = ""
    grok: str = ""
    openai: str = ""
    google: str = ""

def create_access_token(username: str):
    expire = int(time.time()) + ACCESS_TOKEN_EXPIRE_SECONDS
    to_encode = {"sub": username, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(auth_header: str = Header(...)):
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = auth_header.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in users:
            raise HTTPException(status_code=401, detail="Invalid token user")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/agent/ask", response_model=AgentResponse)
def ask_agent(request: AgentRequest, username: str = Depends(verify_token)):
    try:
        # Use username as session_id for per-user memory
        session_id = request.session_id or username
        result = agent.run_task(request.prompt, session_id=session_id)
        return AgentResponse(response=str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/signup", response_model=TokenResponse)
def signup(user: UserAuthRequest):
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    users[user.username] = user.password
    token = create_access_token(user.username)
    return TokenResponse(access_token=token)

@app.post("/auth/login", response_model=TokenResponse)
def login(user: UserAuthRequest):
    if users.get(user.username) != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.username)
    return TokenResponse(access_token=token)

@app.post("/auth/dev_login", response_model=TokenResponse)
def dev_login():
    """Dev-only login endpoint for quick access."""
    token = create_access_token(DEV_USERNAME)
    return TokenResponse(access_token=token)

@app.get("/agent/tools")
def list_tools(username: str = Depends(verify_token)):
    """List all available tools registered with the agent."""
    tools = getattr(agent, 'toolkits', [])
    return {"tools": [str(t) for t in tools]}

@app.get("/agent/subagents")
def list_subagents(username: str = Depends(verify_token)):
    """List all available subagents registered with the agent."""
    subagents = getattr(agent, 'subagents', [])
    return {"subagents": [str(s) for s in subagents]}

@app.get("/agent/memory/graph")
def get_memory_graph(username: str = Depends(verify_token)):
    """
    Expose the user's conversation history as a simple graph.
    Each message is a node; edges connect user->agent for each exchange.
    """
    history = agent.session_memory.get(username, [])
    nodes = []
    edges = []
    for i, exchange in enumerate(history):
        user_node = {"id": f"user_{i}", "type": "user", "text": exchange["user"]}
        agent_node = {"id": f"agent_{i}", "type": "agent", "text": exchange["agent"]}
        nodes.extend([user_node, agent_node])
        edges.append({"from": user_node["id"], "to": agent_node["id"]})
        if i > 0:
            # Link previous agent to next user for continuity
            edges.append({"from": f"agent_{i-1}", "to": user_node["id"]})
    return JSONResponse(content={"nodes": nodes, "edges": edges})

def get_tool_metadata(tool):
    return {
        "name": getattr(tool, 'name', str(tool)),
        "description": getattr(tool, '__doc__', "")
    }

def get_subagent_metadata(subagent):
    return {
        "name": getattr(subagent, 'name', str(subagent)),
        "description": getattr(subagent, '__doc__', "")
    }

@app.get("/agent/tools/full")
def list_tools_full(username: str = Depends(verify_token)):
    """List all available tools with metadata."""
    tools = getattr(agent, 'toolkits', [])
    return {"tools": [get_tool_metadata(t) for t in tools]}

@app.get("/agent/subagents/full")
def list_subagents_full(username: str = Depends(verify_token)):
    """List all available subagents with metadata."""
    subagents = getattr(agent, 'subagents', [])
    return {"subagents": [get_subagent_metadata(s) for s in subagents]}

@app.post("/agent/tool/{tool_name}")
def invoke_tool(tool_name: str, params: dict = Body(...), username: str = Depends(verify_token)):
    """Invoke any tool by name with parameters."""
    tools = getattr(agent, 'toolkits', [])
    for tool in tools:
        if getattr(tool, 'name', str(tool)) == tool_name:
            try:
                # Most tools expect params as kwargs
                result = tool(**params)
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Tool error: {e}")
    raise HTTPException(status_code=404, detail="Tool not found")

@app.post("/agent/subagent/{subagent_name}")
def invoke_subagent(subagent_name: str, prompt: dict = Body(...), username: str = Depends(verify_token)):
    """Invoke any subagent by name with a prompt/task."""
    subagents = getattr(agent, 'subagents', [])
    for subagent in subagents:
        if getattr(subagent, 'name', str(subagent)) == subagent_name:
            try:
                # Most subagents expect a prompt/task
                result = subagent(prompt.get("prompt", ""))
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Subagent error: {e}")
    raise HTTPException(status_code=404, detail="Subagent not found")

@app.post("/agent/memory/graph/add")
def add_memory_node(role: str = Query(..., regex="^(user|agent)$"), text: str = Query(...), username: str = Depends(verify_token)):
    """
    Add a message node to the user's conversation history.
    Appends to the end as either a user or agent message.
    """
    history = agent.session_memory[username]
    if role == "user":
        history.append({"user": text, "agent": ""})
    elif role == "agent":
        if not history or history[-1]["agent"]:
            # No user message to attach to
            raise HTTPException(status_code=400, detail="Add a user message first")
        history[-1]["agent"] = text
    agent.session_memory[username] = history
    return {"status": "ok", "history": history}

@app.post("/agent/memory/graph/edit")
def edit_memory_node(index: int = Query(...), role: str = Query(..., regex="^(user|agent)$"), text: str = Query(...), username: str = Depends(verify_token)):
    """
    Edit a message node in the user's conversation history by index and role.
    """
    history = agent.session_memory[username]
    if index < 0 or index >= len(history):
        raise HTTPException(status_code=404, detail="Index out of range")
    if role not in ("user", "agent"):
        raise HTTPException(status_code=400, detail="Invalid role")
    history[index][role] = text
    agent.session_memory[username] = history
    return {"status": "ok", "history": history}

@app.post("/agent/memory/graph/delete")
def delete_memory_node(index: int = Query(...), username: str = Depends(verify_token)):
    """
    Delete a user/agent message pair from the user's conversation history by index.
    """
    history = agent.session_memory[username]
    if index < 0 or index >= len(history):
        raise HTTPException(status_code=404, detail="Index out of range")
    history.pop(index)
    agent.session_memory[username] = history
    return {"status": "ok", "history": history}

@app.get("/agent/knowledge/sources")
def list_knowledge_sources(username: str = Depends(verify_token)):
    """List all knowledge sources (URLs) in the agent's knowledge base."""
    sources = getattr(agent.knowledge_base, 'urls', [])
    return {"sources": sources}

@app.post("/agent/knowledge/sources/add")
def add_knowledge_source(url: str = Query(...), username: str = Depends(verify_token)):
    """Add a new URL to the agent's knowledge base."""
    if url not in agent.knowledge_base.urls:
        agent.knowledge_base.urls.append(url)
        # Optionally, trigger reload or embedding
    return {"status": "ok", "sources": agent.knowledge_base.urls}

@app.post("/agent/knowledge/sources/remove")
def remove_knowledge_source(url: str = Query(...), username: str = Depends(verify_token)):
    """Remove a URL from the agent's knowledge base."""
    if url in agent.knowledge_base.urls:
        agent.knowledge_base.urls.remove(url)
    return {"status": "ok", "sources": agent.knowledge_base.urls}

@app.get("/agent/team/subagents")
def list_team_subagents(username: str = Depends(verify_token)):
    """List all subagents currently registered with the agent."""
    subagents = getattr(agent, 'subagents', [])
    return {"subagents": [str(s) for s in subagents]}

@app.post("/agent/team/subagents/add")
def add_team_subagent(subagent_name: str = Query(...), username: str = Depends(verify_token)):
    """Add a subagent by name (if available in Agno registry)."""
    from agno.team import get_all_subagents
    all_subagents = get_all_subagents()
    for subagent in all_subagents:
        if getattr(subagent, 'name', str(subagent)) == subagent_name:
            agent.register_subagent(subagent)
            return {"status": "ok", "subagents": [str(s) for s in agent.subagents]}
    raise HTTPException(status_code=404, detail="Subagent not found in registry")

@app.post("/agent/team/subagents/remove")
def remove_team_subagent(subagent_name: str = Query(...), username: str = Depends(verify_token)):
    """Remove a subagent by name."""
    subagents = getattr(agent, 'subagents', [])
    for subagent in subagents:
        if getattr(subagent, 'name', str(subagent)) == subagent_name:
            agent.subagents.remove(subagent)
            return {"status": "ok", "subagents": [str(s) for s in agent.subagents]}
    raise HTTPException(status_code=404, detail="Subagent not found")

@app.post("/agent/memory/graph/custom/add_node")
def add_custom_node(
    node_type: str = Query(...),
    text: str = Query(...),
    metadata: Optional[Dict] = None,
    username: str = Depends(verify_token),
):
    """
    Add a custom node (with type, text, optional metadata) to the user's memory graph.
    Returns the node id.
    """
    graph = advanced_graph_memory[username]
    node_id = f"{node_type}_{len(graph['nodes'])}"
    node = {"id": node_id, "type": node_type, "text": text, "metadata": metadata or {}}
    graph["nodes"].append(node)
    return {"status": "ok", "node": node}

@app.post("/agent/memory/graph/custom/add_edge")
def add_custom_edge(
    from_id: str = Query(...),
    to_id: str = Query(...),
    label: Optional[str] = None,
    username: str = Depends(verify_token),
):
    """
    Add a custom edge (from, to, optional label) to the user's memory graph.
    """
    graph = advanced_graph_memory[username]
    edge = {"from": from_id, "to": to_id, "label": label or ""}
    graph["edges"].append(edge)
    return {"status": "ok", "edge": edge}

@app.post("/agent/memory/graph/custom/remove_node")
def remove_custom_node(node_id: str = Query(...), username: str = Depends(verify_token)):
    """
    Remove a node (and all connected edges) from the user's memory graph.
    """
    graph = advanced_graph_memory[username]
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != node_id]
    graph["edges"] = [e for e in graph["edges"] if e["from"] != node_id and e["to"] != node_id]
    return {"status": "ok"}

@app.post("/agent/memory/graph/custom/remove_edge")
def remove_custom_edge(from_id: str = Query(...), to_id: str = Query(...), username: str = Depends(verify_token)):
    """
    Remove an edge from the user's memory graph.
    """
    graph = advanced_graph_memory[username]
    graph["edges"] = [e for e in graph["edges"] if not (e["from"] == from_id and e["to"] == to_id)]
    return {"status": "ok"}

@app.get("/agent/memory/graph/custom/full")
def get_full_custom_graph(username: str = Depends(verify_token)):
    """
    Get the full custom memory graph (nodes and edges with types/metadata).
    """
    graph = advanced_graph_memory[username]
    return {"nodes": graph["nodes"], "edges": graph["edges"]}

@app.get("/user/api_keys")
def get_api_keys(username: str = Depends(verify_token)):
    """
    Secure endpoint: Get the current user's API keys. Only returns keys for the logged-in user.
    """
    return user_api_keys.get(username, {})

@app.post("/user/api_keys/update")
def update_api_keys(keys: APIKeysRequest, username: str = Depends(verify_token)):
    """
    Secure endpoint: Update the current user's API keys. Only updates keys for the logged-in user.
    """
    user_api_keys[username] = {
        "claude": keys.claude,
        "grok": keys.grok,
        "openai": keys.openai,
        "google": keys.google,
    }
    return {"status": "ok"}

@app.get("/agent/chat/history")
def get_chat_history(n: int = Query(10), session_id: str = Query("default"), username: str = Depends(verify_token)):
    """
    Get the last N agent responses (with model info) for chat/graph display.
    """
    history = agent.session_memory.get(session_id, [])
    return {"history": history[-n:]}

@app.get("/agent/models")
def get_available_models(username: str = Depends(verify_token)):
    """
    List available LLM models for the user (based on their API keys).
    """
    keys = user_api_keys.get(username, {})
    models = []
    if keys.get("claude"): models.append("Claude (Anthropic)")
    if keys.get("grok"): models.append("Grok (xAI)")
    if keys.get("openai"): models.append("OpenAI GPT-4o")
    if keys.get("google"): models.append("Google Gemini")
    return {"models": models}

@app.get("/agent/team/members")
def list_team_members(username: str = Depends(verify_token)):
    """
    List all team members (agents) for the current agent.
    """
    members = getattr(agent, 'team', [])
    return {"team": [str(m) for m in members]}

@app.post("/agent/team/members/add")
def add_team_member(member_name: str = Query(...), username: str = Depends(verify_token)):
    """
    Add a team member (agent) by name from registry.
    """
    from agno.team import get_all_subagents
    all_subagents = get_all_subagents()
    for subagent in all_subagents:
        if getattr(subagent, 'name', str(subagent)) == member_name:
            if not hasattr(agent, 'team') or agent.team is None:
                agent.team = []
            agent.team.append(subagent)
            return {"status": "ok", "team": [str(m) for m in agent.team]}
    raise HTTPException(status_code=404, detail="Team member not found in registry")

@app.post("/agent/team/members/remove")
def remove_team_member(member_name: str = Query(...), username: str = Depends(verify_token)):
    """
    Remove a team member (agent) by name.
    """
    if not hasattr(agent, 'team') or agent.team is None:
        agent.team = []
    for subagent in agent.team:
        if getattr(subagent, 'name', str(subagent)) == member_name:
            agent.team.remove(subagent)
            return {"status": "ok", "team": [str(m) for m in agent.team]}
    raise HTTPException(status_code=404, detail="Team member not found")

@app.post("/agent/team/assign_role")
def assign_team_role(member_name: str = Query(...), role: str = Query(...), username: str = Depends(verify_token)):
    """
    Assign a role to a team member.
    """
    if not hasattr(agent, 'team') or agent.team is None:
        agent.team = []
    for subagent in agent.team:
        if getattr(subagent, 'name', str(subagent)) == member_name:
            subagent.role = role
            return {"status": "ok", "team": [{"name": str(m), "role": getattr(m, 'role', None)} for m in agent.team]}
    raise HTTPException(status_code=404, detail="Team member not found")

@app.post("/agent/team/transfer_task")
def transfer_task_to_member(member_name: str = Query(...), task: str = Query(...), username: str = Depends(verify_token)):
    """
    Transfer a task to a team member and return their response.
    """
    if not hasattr(agent, 'team') or agent.team is None:
        agent.team = []
    for subagent in agent.team:
        if getattr(subagent, 'name', str(subagent)) == member_name:
            result = subagent(task)
            return {"result": result}
    raise HTTPException(status_code=404, detail="Team member not found")

@app.post("/agent/reasoning/stream")
def stream_reasoning(task: str = Query(...), session_id: str = Query("default"), username: str = Depends(verify_token)):
    """
    Stream reasoning steps, chain-of-thought, and self-reflection for a task.
    """
    def event_stream():
        # Use the agent's planner, chain_of_thought, and self_reflection
        plan = agent.planner.plan(task)
        history = agent.session_memory[session_id]
        for step in range(10):
            thought = agent.chain_of_thought.think(plan, history=history)
            action_result = agent.run_task(thought, session_id=session_id)
            reflection = agent.self_reflection.reflect(thought, action_result, history=history)
            event = {
                "step": step,
                "plan": plan,
                "thought": thought,
                "action_result": action_result,
                "reflection": reflection
            }
            yield f"data: {json.dumps(event)}\n\n"
            if "solved" in str(reflection).lower() or "complete" in str(reflection).lower():
                break
            plan = agent.planner.replan(task, history=history)
    return StreamingResponse(event_stream(), media_type="text/event-stream") 
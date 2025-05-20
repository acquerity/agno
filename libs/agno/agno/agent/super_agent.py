import logging
from collections import defaultdict
from agno.agent.base import Agent
from agno.tools import get_all_toolkits
from agno.team import get_all_subagents
from agno.knowledge.url import UrlKnowledge
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.cohere import CohereEmbedder
from agno.reranker.cohere import CohereReranker
from agno.tools.reasoning import ReasoningTools
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.memory.manager import MemoryManager
from agno.memory.summarizer import MemorySummarizer
from agno.storage.agent.sqlite import SqliteAgentStorage
import asyncio
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.reasoning.planner import AgentPlanner
from agno.reasoning.chain_of_thought import ChainOfThoughtReasoner
from agno.reasoning.self_reflection import SelfReflection
# If MCP server/client is available:
try:
    from agno.tools.mcp import MCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class SuperAgent(Agent):
    def __init__(self, name="SuperAgent", **kwargs):
        # Persistent agent and memory storage
        self.agent_storage = SqliteAgentStorage(table_name="superagent_sessions", db_file="tmp/agents.db")
        self.memory_db = SqliteMemoryDb(table_name="superagent_memory", db_file="tmp/agent_memory.db")
        self.memory_manager = MemoryManager(memory_db=self.memory_db, summarizer=MemorySummarizer())
        # Knowledge base (multi-source, RAG)
        self.knowledge_base = UrlKnowledge(
            urls=[
                "https://docs.agno.com/introduction/agents.md",
                # Add more URLs or sources as needed
            ],
            vector_db=LanceDb(
                uri="tmp/lancedb",
                table_name="superagent_docs",
                search_type=SearchType.hybrid,
                embedder=CohereEmbedder(id="embed-v4.0"),
                reranker=CohereReranker(model="rerank-v3.5"),
            ),
        )
        # Reasoning tools and hooks
        reasoning_tools = ReasoningTools(add_instructions=True)
        # Gather all toolkits and subagents
        all_toolkits = get_all_toolkits()
        all_subagents = get_all_subagents()
        # Optionally, add multimodal tools here
        multimodal_tools = []  # e.g., image, audio, code tools
        # Compose instructions
        instructions = [
            "Include sources in your response.",
            "Always search your knowledge before answering the question.",
            "Use step-by-step reasoning and validate your answers.",
        ]
        # Advanced reasoning/planning subsystems
        self.planner = AgentPlanner()
        self.chain_of_thought = ChainOfThoughtReasoner()
        self.self_reflection = SelfReflection()
        # MCP (Multi-Agent Control/Planning) integration
        self.mcp_client = MCPClient() if MCP_AVAILABLE else None
        # Enable all advanced memory/session/knowledge features
        agentic_memory = True
        user_memories = True
        session_summaries = True
        agentic_knowledge_filters = True
        add_memory_references = True
        add_session_summary_references = True
        add_history_to_messages = True
        num_history_runs = 10
        # Enable all debugging/monitoring/telemetry
        debug_mode = True
        monitoring = True
        telemetry = True
        # Enable all hooks (tool, reasoning, context)
        tool_hooks = kwargs.get('tool_hooks', [])
        # Team/multi-agent support
        team = kwargs.get('team', [])
        team_data = kwargs.get('team_data', {})
        # Init base Agent with all advanced features
        super().__init__(
            name=name,
            knowledge=self.knowledge_base,
            search_knowledge=True,
            tools=[reasoning_tools] + all_toolkits + multimodal_tools,
            subagents=all_subagents,
            instructions=instructions,
            memory_manager=self.memory_manager,
            agent_storage=self.agent_storage,
            markdown=True,
            # Advanced memory/session/knowledge
            enable_agentic_memory=agentic_memory,
            enable_user_memories=user_memories,
            enable_session_summaries=session_summaries,
            enable_agentic_knowledge_filters=agentic_knowledge_filters,
            add_memory_references=add_memory_references,
            add_session_summary_references=add_session_summary_references,
            add_history_to_messages=add_history_to_messages,
            num_history_runs=num_history_runs,
            # Debug/monitoring/telemetry
            debug_mode=debug_mode,
            monitoring=monitoring,
            telemetry=telemetry,
            # Hooks and team
            tool_hooks=tool_hooks,
            team=team,
            team_data=team_data,
            **kwargs
        )
        # In-memory session memory for fast access
        self.session_memory = defaultdict(list)
        # Setup logging
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)

    def select_model(self, task_description, api_keys=None):
        """
        Select the best LLM/model for the given task.
        Uses keywords and heuristics to route to Claude, Grok, OpenAI, or Google.
        api_keys: dict of API keys for each provider.
        """
        # Simple heuristic: can be expanded with more advanced logic
        task = task_description.lower()
        if any(x in task for x in ["philosophy", "ethics", "reasoning", "explain"]):
            # Claude is strong for reasoning and philosophy
            return Claude(id="claude-3-7-sonnet-latest", api_key=api_keys.get("claude"))
        elif any(x in task for x in ["real-time", "trending", "x.ai", "elon", "grok"]):
            return Groq(id="grok-1", api_key=api_keys.get("grok"))
        elif any(x in task for x in ["code", "python", "generate code", "openai"]):
            return OpenAIChat(id="gpt-4o", api_key=api_keys.get("openai"))
        elif any(x in task for x in ["vision", "image", "google", "gemini"]):
            return Gemini(id="gemini-1.5-pro", api_key=api_keys.get("google"))
        # Default: OpenAI
        return OpenAIChat(id="gpt-4o", api_key=api_keys.get("openai"))

    def run_task(self, task_description, session_id=None, api_keys=None, **kwargs):
        """
        Run a task autonomously using the best LLM/model for the task.
        Maintains per-session conversation history and persistent memory.
        """
        if session_id is None:
            session_id = "default"
        if api_keys is None:
            api_keys = {}
        self.logger.info(f"Session {session_id}: User prompt: {task_description}")
        # Model selection
        model = self.select_model(task_description, api_keys=api_keys)
        self.model = model
        # Retrieve conversation history
        history = self.session_memory[session_id]
        result = self.act(task_description, history=history, **kwargs)
        # Store in history
        history.append({"user": task_description, "agent": str(result), "model": str(model)})
        self.session_memory[session_id] = history
        self.logger.info(f"Session {session_id}: Agent response: {result}")
        self.memory_manager.add_memory(session_id, task_description, str(result))
        return result

    async def async_run_task(self, task_description, session_id=None, api_keys=None, **kwargs):
        """
        Async version: run a task using the best LLM/model for the task.
        """
        if session_id is None:
            session_id = "default"
        if api_keys is None:
            api_keys = {}
        self.logger.info(f"[ASYNC] Session {session_id}: User prompt: {task_description}")
        model = self.select_model(task_description, api_keys=api_keys)
        self.model = model
        history = self.session_memory[session_id]
        result = await self.aact(task_description, history=history, **kwargs) if hasattr(self, 'aact') else self.act(task_description, history=history, **kwargs)
        history.append({"user": task_description, "agent": str(result), "model": str(model)})
        self.session_memory[session_id] = history
        self.logger.info(f"[ASYNC] Session {session_id}: Agent response: {result}")
        self.memory_manager.add_memory(session_id, task_description, str(result))
        return result

    # Scaffold for multimodal orchestration (e.g., image, audio, code tools)
    def multimodal_run(self, task_description, modality, session_id=None, **kwargs):
        """
        Run a task with a specified modality (e.g., 'image', 'audio', 'code').
        """
        # Example: route to appropriate tool or subagent based on modality
        for tool in getattr(self, 'toolkits', []):
            if hasattr(tool, 'modality') and tool.modality == modality:
                return tool(task_description, **kwargs)
        for subagent in getattr(self, 'subagents', []):
            if hasattr(subagent, 'modality') and subagent.modality == modality:
                return subagent(task_description, **kwargs)
        # Fallback to default
        return self.run_task(task_description, session_id=session_id, **kwargs)

    def autonomous_loop(self, task_description, session_id=None, api_keys=None, max_steps=10, **kwargs):
        """
        Structured autonomous loop: think, plan, act, reflect, and iterate until task is solved or max_steps reached.
        """
        if session_id is None:
            session_id = "default"
        if api_keys is None:
            api_keys = {}
        history = self.session_memory[session_id]
        plan = self.planner.plan(task_description)
        for step in range(max_steps):
            thought = self.chain_of_thought.think(plan, history=history)
            action_result = self.run_task(thought, session_id=session_id, api_keys=api_keys, **kwargs)
            reflection = self.self_reflection.reflect(thought, action_result, history=history)
            history.append({
                "step": step,
                "plan": plan,
                "thought": thought,
                "action_result": action_result,
                "reflection": reflection
            })
            # Optionally, use MCP for multi-agent orchestration
            if self.mcp_client:
                self.mcp_client.coordinate_agents(task_description, context=history)
            # Check if task is solved (can be improved with better criteria)
            if "solved" in str(reflection).lower() or "complete" in str(reflection).lower():
                break
            # Update plan for next iteration
            plan = self.planner.replan(task_description, history=history)
        self.session_memory[session_id] = history
        return history[-1] if history else None 
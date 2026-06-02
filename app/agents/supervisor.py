r"""
app/agents/supervisor.py
------------------------
LangGraph multi-step supervisor orchestrator for ScholarAgent AI.

Graph topology
--------------

    START -> supervisor_node <---+
                  |              |
          should_continue()      |
          /            \         |
    "end"              "tools"   |
      |                  |       |
     END          call_tool_node-+
                  (results appended to messages,
                   control returns to supervisor_node)

Design decisions
----------------
* The supervisor injects the live student profile into a system-level prompt
  preamble on every iteration so the LLM always has full context, even mid-loop.
* `should_continue` inspects the last assistant message for tool_calls; if none
  are present (or the tool-call budget is exhausted) it routes to END.
* A hard cap (`MAX_TOOL_ITERATIONS`) prevents runaway loops from token-bombing
  a local Ollama instance with bounded VRAM.
* Tools are bound to the LLM via `bind_tools`; the LangChain tool-use layer
  handles JSON Schema generation and call-result round-tripping automatically.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from app.agents.tools import AGENT_TOOLS
from app.schemas.models import AgentState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.1"

# Hard cap on tool-invocation cycles per request.
# Prevents infinite loops when the model repeatedly schedules tools.
MAX_TOOL_ITERATIONS: int = 6

# ─────────────────────────────────────────────
# LLM initialisation
# ─────────────────────────────────────────────


def _build_llm() -> ChatOllama:
    """
    Instantiate and return a ChatOllama client with tool-binding pre-applied.

    ChatOllama automatically converts LangChain @tool definitions into the
    Ollama-native function-calling schema when `bind_tools` is called.

    Returns:
        A `ChatOllama` instance with AGENT_TOOLS bound.

    Raises:
        RuntimeError: If Ollama is not reachable (surfaced at first inference).
    """
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,          # Low temp for deterministic tool selection
        num_predict=2048,          # Max tokens per generation step
        # format="json" left off — we want free-form + tool-call output
    )
    return llm.bind_tools(AGENT_TOOLS)


# Singleton LLM — shared across all requests in the same worker process.
# Re-creating it per-request would flush the Ollama KV cache unnecessarily.
_LLM_WITH_TOOLS = _build_llm()


# ─────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────


def _build_system_prompt(student_profile: dict[str, Any]) -> str:
    """
    Construct the system-level context prompt injected at the top of every LLM call.

    The prompt is regenerated on each supervisor iteration so profile data is
    always current (supports future profile-mutation mid-session).

    Args:
        student_profile: Serialised StudentProfile data dict.

    Returns:
        Multi-line system prompt string.
    """
    gpa: float = student_profile.get("gpa", 0.0)
    major: str = student_profile.get("major", "Undeclared")
    residency: str = student_profile.get("residency", "Unknown")
    interests: list[str] = student_profile.get("interests", [])
    interests_str: str = ", ".join(interests) if interests else "None specified"

    return f"""You are ScholarAgent AI, an expert autonomous scholarship analyst.
Your mission is to help students discover and qualify for scholarships using planning, tool-driven search,
iterative evaluation, and evidence grounding.

══════════════════════════════════════════
ACTIVE STUDENT PROFILE
══════════════════════════════════════════
  GPA:       {gpa:.2f} / 4.00
  Major:     {major}
  Residency: {residency}
  Interests: {interests_str}
══════════════════════════════════════════

WORKFLOW:
1. Understand the user's request and explicitly decompose it into subgoals.
   Identify constraints such as location, academic level, GPA, field of study, and residency.
2. Call plan_search_queries_tool first to generate multiple optimized search queries.
3. Use search_scholarships_tool iteratively over the best query variants. The tool accepts either a single `query` or a list of `queries`.
4. After each retrieval round, call evaluate_search_results_tool to assess relevance, Korean/Seoul coverage,
   academic source quality, and hard-constraint satisfaction.
5. If results are insufficient, refine or broaden the search strategy and repeat.
6. Optionally use extract_document_text_tool to inspect university pages or official PDF documents
   and ground the final recommendations in source evidence.
7. Stop only after you have a strong list of grounded candidates, then return a final synthesis.

RESPONSE STANDARDS:
• Produce structured, evidence-grounded recommendations for each scholarship.
• Include title, provider, URL, relevance score, and concise explanation grounded in retrieved content.
• Prioritize South Korea / Seoul sources when the user's profile or request is location-constrained.
• Do not invent scholarship details or eligibility facts that are not grounded in tool output.
• If evidence is weak, call extract_document_text_tool before finalizing.
• Keep the final answer concise, factual, and free of hallucinated narrative.

CRITICAL: Use tools for planning, search, evaluation, and extraction.  Do NOT answer without sufficient evidence.
You have access to the following tools: {[t.name for t in AGENT_TOOLS]}
"""


# ─────────────────────────────────────────────
# Graph node implementations
# ─────────────────────────────────────────────


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    Primary reasoning node: prepends the system prompt and runs the LLM step.

    On each invocation:
    1. Builds a fresh system prompt containing the active student profile.
    2. Prepends it as a `SystemMessage` so the LLM always has full context.
    3. Invokes the tool-bound LLM with the current message history.
    4. Appends the resulting `AIMessage` (which may contain tool_calls) to state.

    Args:
        state: Current graph state, including messages and student_profile.

    Returns:
        Partial state update dict — LangGraph merges this into the running state.
    """
    logger.info(
        "supervisor_node | messages=%d tool_calls_so_far=%d",
        len(state["messages"]),
        state.get("tool_call_count", 0),
    )

    student_profile: dict[str, Any] = state.get("student_profile", {})
    system_prompt: str = _build_system_prompt(student_profile)

    # Prepend system message — ChatOllama accepts the full messages list
    messages_with_system = [SystemMessage(content=system_prompt)] + list(
        state["messages"]
    )

    try:
        response: AIMessage = _LLM_WITH_TOOLS.invoke(messages_with_system)  # type: ignore[assignment]
    except Exception as exc:
        logger.error("supervisor_node LLM invocation failed: %s", exc, exc_info=True)
        # Surface error as an AI message so the graph can still resolve gracefully
        response = AIMessage(
            content=(
                f"I encountered an error communicating with the local model: {exc}. "
                "Please ensure Ollama is running at http://localhost:11434 with "
                f"the '{OLLAMA_MODEL}' model loaded."
            )
        )

    if (
        not getattr(response, "tool_calls", None)
        and state.get("tool_call_count", 0) == 0
        and not state.get("fallback_forced", False)
        and len(state.get("messages", [])) == 1
    ):
        logger.warning(
            "supervisor_node | no tool calls produced on first pass; injecting fallback plan_search_queries_tool"
        )
        forced_tool_call = {
            "id": "force-plan-1",
            "name": "plan_search_queries_tool",
            "args": {
                "prompt": state["messages"][0].content,
                "student_profile": student_profile,
                "location_constraint": student_profile.get("residency", "South Korea") or "South Korea",
            },
        }
        fallback_message = AIMessage(
            content="Fallback: the agent did not use tools, so a planning tool call is injected.",
            tool_calls=[forced_tool_call],
        )
        return {"messages": [response, fallback_message], "fallback_forced": True}

    # `add_messages` reducer will append `response` to state["messages"]
    return {"messages": [response]}


def call_tool_node(state: AgentState) -> dict[str, Any]:
    """
    Tool execution node: resolves all tool_calls in the last AI message.

    LangGraph routes here when `should_continue` returns "tools".  This node:
    1. Extracts all pending tool_calls from the last `AIMessage`.
    2. Looks up each tool by name in AGENT_TOOLS.
    3. Invokes the tool with the provided arguments.
    4. Automatically injects search_mode for search_scholarships_tool.
    5. Wraps results in `ToolMessage` objects (required by LangChain tool-use protocol).
    6. Returns all results as a message batch — `add_messages` appends them.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with tool result messages and incremented counter.
    """
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("call_tool_node reached with no tool_calls — returning empty")
        return {"messages": [], "tool_call_count": state.get("tool_call_count", 0)}

    tool_results: list[ToolMessage] = []

    # Build a name→callable index for O(1) dispatch
    tool_index: dict[str, Any] = {t.name: t for t in AGENT_TOOLS}
    
    # Extract search_mode from state (default to "general")
    search_mode = state.get("search_mode", "general")

    for tool_call in last_message.tool_calls:
        tool_name: str = tool_call["name"]
        tool_args: dict[str, Any] = tool_call.get("args", {})
        tool_call_id: str = tool_call["id"]

        # Automatically inject search_mode for search_scholarships_tool
        if tool_name == "search_scholarships_tool":
            tool_args["search_mode"] = search_mode
            logger.info(
                "call_tool_node | injected search_mode=%r into search_scholarships_tool",
                search_mode,
            )

        logger.info(
            "call_tool_node | executing tool=%r args=%s", tool_name, tool_args
        )

        try:
            if tool_name not in tool_index:
                raise KeyError(f"Unknown tool: '{tool_name}'")

            tool_fn = tool_index[tool_name]
            raw_result = tool_fn.invoke(tool_args)

            # Ensure result is a plain string (ToolMessage.content must be str)
            if not isinstance(raw_result, str):
                raw_result = json.dumps(raw_result, default=str)

            tool_results.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    content=raw_result,
                )
            )
            logger.debug("call_tool_node | tool=%r completed", tool_name)

        except Exception as exc:
            logger.error(
                "call_tool_node | tool=%r raised %s: %s",
                tool_name, type(exc).__name__, exc, exc_info=True,
            )
            tool_results.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    content=json.dumps({"error": str(exc)}),
                )
            )

    new_count: int = state.get("tool_call_count", 0) + len(tool_results)
    logger.info(
        "call_tool_node | cycle complete tool_call_count=%d", new_count
    )

    return {
        "messages": tool_results,
        "tool_call_count": new_count,
    }


# ─────────────────────────────────────────────
# Conditional edge router
# ─────────────────────────────────────────────


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge function that decides the next graph node after `supervisor_node`.

    Routing logic (evaluated in priority order):
    1. If the tool-call budget is exhausted → END (safety valve).
    2. If the last AI message contains tool_calls → route to "tools".
    3. Otherwise (pure text response) → END.

    Args:
        state: Current graph state after `supervisor_node` has appended its output.

    Returns:
        "tools" to route to `call_tool_node`, or "end" to route to END.
    """
    tool_call_count: int = state.get("tool_call_count", 0)

    if tool_call_count >= MAX_TOOL_ITERATIONS:
        logger.warning(
            "should_continue | MAX_TOOL_ITERATIONS (%d) reached — forcing END",
            MAX_TOOL_ITERATIONS,
        )
        return "end"

    last_message = state["messages"][-1] if state["messages"] else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.debug(
            "should_continue → tools | %d pending tool_calls",
            len(last_message.tool_calls),
        )
        return "tools"

    logger.debug("should_continue → end | no pending tool_calls")
    return "end"


# ─────────────────────────────────────────────
# Graph compilation
# ─────────────────────────────────────────────


def build_graph() -> Any:
    """
    Assemble and compile the LangGraph `StateGraph` for ScholarAgent AI.

    Node registration:
        "supervisor"  → supervisor_node
        "tools"       → call_tool_node

    Edge configuration:
        START → "supervisor"
        "supervisor" ─(conditional)─► "tools" or END
        "tools" → "supervisor"   (loop back for next reasoning step)

    Returns:
        A compiled LangGraph runnable (supports `.invoke()` and `.astream()`).
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tools", call_tool_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional routing after each supervisor pass
    graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # Tool results always loop back to the supervisor for the next reasoning step
    graph.add_edge("tools", "supervisor")

    compiled = graph.compile()
    logger.info("ScholarAgent LangGraph compiled successfully.")
    return compiled


# Module-level compiled graph singleton — avoids re-compilation per request.
_COMPILED_GRAPH = build_graph()


def get_compiled_graph() -> Any:
    """Return the module-level compiled LangGraph instance."""
    return _COMPILED_GRAPH

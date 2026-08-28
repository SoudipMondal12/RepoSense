from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.state import RepoSenseState
from app.graph.nodes import RepoSenseNodes


def build_repo_sense_graph(
    api_key: str,
    indexer,
    code_graph=None,
):
    """
    Build the RepoSense LangGraph.

    The MemorySaver checkpointer enables conversation state to survive
    multiple graph.invoke() calls when the same thread_id is supplied.

    Example:

        config = {"configurable": {"thread_id": "repo-owner/repo"}}

        graph.invoke(first_state, config=config)
        graph.invoke({"question": "Follow-up..."}, config=config)
    """

    print("[LangGraph] Building RepoSense graph...")

    nodes = RepoSenseNodes(
        api_key=api_key,
        indexer=indexer,
        code_graph=code_graph,
    )

    graph = StateGraph(RepoSenseState)

    # --------------------------------------------------------
    # ROUTER
    # --------------------------------------------------------

    graph.add_node("router", nodes.route_node)

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    graph.add_node("search", nodes.search_node)
    graph.add_node("search_answer", nodes.search_answer_node)

    # --------------------------------------------------------
    # EXPLAIN
    # --------------------------------------------------------

    graph.add_node("explain", nodes.explain_node)
    graph.add_node("explain_answer", nodes.explain_answer_node)

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    graph.add_node("debug", nodes.debug_node)
    graph.add_node("debug_related", nodes.debug_related_node)
    graph.add_node("debug_evidence", nodes.debug_evidence_node)
    graph.add_node("debug_analysis", nodes.debug_analysis_node)
    graph.add_node("debug_answer", nodes.debug_answer_node)

    # --------------------------------------------------------
    # CONVERSATION MEMORY
    # --------------------------------------------------------

    graph.add_node(
        "save_conversation",
        nodes.save_conversation_node,
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    graph.add_edge(START, "router")

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "router",
        lambda state: state["intent"],
        {
            "search": "search",
            "explain": "explain",
            "debug": "debug",
        },
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    graph.add_edge("search", "search_answer")
    graph.add_edge("search_answer", "save_conversation")

    # --------------------------------------------------------
    # EXPLAIN
    # --------------------------------------------------------

    graph.add_edge("explain", "explain_answer")
    graph.add_edge("explain_answer", "save_conversation")

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    graph.add_edge("debug", "debug_related")
    graph.add_edge("debug_related", "debug_evidence")
    graph.add_edge("debug_evidence", "debug_analysis")
    graph.add_edge("debug_analysis", "debug_answer")
    graph.add_edge("debug_answer", "save_conversation")

    # --------------------------------------------------------
    # SAVE -> END
    # --------------------------------------------------------

    graph.add_edge("save_conversation", END)

    # --------------------------------------------------------
    # CHECKPOINTER
    # --------------------------------------------------------
    #
    # MemorySaver is intentionally local/in-memory for this step.
    # The Streamlit process must keep the compiled graph instance alive.
    # A later production step can replace this with a persistent DB.
    #

    checkpointer = MemorySaver()

    compiled_graph = graph.compile(
        checkpointer=checkpointer
    )

    print("[LangGraph] Graph compiled successfully.")
    print("[LangGraph] Conversation checkpointer: MemorySaver")

    return compiled_graph

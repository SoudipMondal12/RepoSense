from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# TEST
# ============================================================

def main():
    print("=" * 80)
    print("RepoSense - Step 4.3 Conversation Context Test")
    print("=" * 80)

    print("\n[1/5] Loading state and node helpers...")

    from app.graph.state import RepoSenseState
    from app.graph.nodes import RepoSenseNodes

    print("Imports: OK")

    # --------------------------------------------------------
    # First conversation
    # --------------------------------------------------------

    print("\n[2/5] Creating first conversation turn...")

    state: RepoSenseState = {
        "owner": "test-owner",
        "repo": "test-repo",
        "branch": "main",
        "question": "Where is authentication implemented?",
        "messages": [],
    }

    print("Question 1:")
    print(state["question"])

    # Simulate the saved answer from the first graph turn.
    from langchain_core.messages import HumanMessage, AIMessage

    state["messages"] = [
        HumanMessage(
            content="Where is authentication implemented?"
        ),
        AIMessage(
            content=(
                "Authentication is implemented in "
                "auth/service.py, specifically "
                "authenticate_user()."
            )
        ),
    ]

    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    print("\n[3/5] Creating follow-up question...")

    state["question"] = "Explain that function."

    history = RepoSenseNodes.get_conversation_context(
        state,
        max_messages=6,
        max_chars=3500,
    )

    retrieval_query = RepoSenseNodes.build_retrieval_query(
        state
    )

    print("Question 2:")
    print(state["question"])

    print("\nConversation context:")
    print(history)

    print("\nRetrieval query:")
    print(retrieval_query)

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    print("\n[4/5] Validating context awareness...")

    assert "Where is authentication implemented?" in history
    assert "auth/service.py" in history
    assert "authenticate_user()" in history
    assert "Explain that function." in retrieval_query
    assert "auth/service.py" in retrieval_query

    # Strict bounded-context checks.
    assert len(history) <= 3500
    assert len(retrieval_query) <= 6000

    print("Conversation context: OK")
    print("Follow-up context: OK")
    print("Context limits: OK")

    # --------------------------------------------------------
    # No-history behavior
    # --------------------------------------------------------

    print("\n[5/5] Testing first-question behavior...")

    first_state: RepoSenseState = {
        "question": "What is the topic of this repository?",
        "messages": [],
    }

    first_query = RepoSenseNodes.build_retrieval_query(
        first_state
    )

    assert first_query == (
        "What is the topic of this repository?"
    )

    print("First-question behavior: OK")

    print("\n" + "=" * 80)
    print("STEP 4.3 TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()

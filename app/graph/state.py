from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RepoSenseState(TypedDict, total=False):

    # ========================================================
    # REPOSITORY
    # ========================================================

    owner: str
    repo: str
    branch: str

    # ========================================================
    # CURRENT QUESTION
    # ========================================================

    question: str

    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    # ========================================================
    # ROUTING
    # ========================================================

    intent: str

    # ========================================================
    # RETRIEVAL
    # ========================================================

    documents: list
    related_documents: list
    evidence: str

    # ========================================================
    # DEBUG
    # ========================================================

    diagnosis: str
    confidence: str

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    answer: str
    sources: list
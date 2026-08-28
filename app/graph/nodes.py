from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.state import RepoSenseState
from app.graph.router import RepoSenseRouter
from app.rag.context import format_retrieved_context


class RepoSenseNodes:
    """
    LangGraph nodes for RepoSense.

    LLM:
        Groq

    Embeddings / retrieval:
        Existing repository indexer

    Code relationships:
        Existing CodeRelationshipGraph
    """

    # ========================================================
    # TOKEN / CONTEXT LIMITS
    # ========================================================

    # Groq currently reports an 8,000 TPM limit for the
    # selected model on the user's free/on-demand tier.
    #
    # We deliberately stay below that limit because the
    # actual request also contains instructions and question.
    MAX_EXPLAIN_CONTEXT_CHARS = 18000
    MAX_DEBUG_CONTEXT_CHARS = 15000
    MAX_DOCUMENT_CHARS = 4000

    # Keep prompts comfortably below the model's TPM limit.
    MAX_EXPLAIN_DOCUMENTS = 5
    MAX_DEBUG_DOCUMENTS = 4
    MAX_RELATED_DOCUMENTS = 6

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        api_key: str,
        indexer,
        code_graph=None,
    ):

        self.api_key = api_key

        self.indexer = indexer

        self.code_graph = code_graph

        # ----------------------------------------------------
        # Router
        # ----------------------------------------------------

        self.router = RepoSenseRouter(
            api_key=api_key
        )

        # ----------------------------------------------------
        # Groq LLM
        # ----------------------------------------------------

        from app.llm.groq import GroqLLMService

        llm_service = GroqLLMService(
            api_key=api_key,
            model="openai/gpt-oss-120b",
            temperature=0.1,
        )

        self.llm = llm_service.get_model()

        print(
            "[RepoSenseNodes] Groq LLM initialized."
        )

    # ========================================================
    # TEXT EXTRACTION
    # ========================================================

    @staticmethod
    def extract_text(content) -> str:

        if isinstance(content, str):

            return content.strip()

        if isinstance(content, list):

            parts = []

            for block in content:

                if isinstance(block, dict):

                    text = block.get(
                        "text"
                    )

                    if text:

                        parts.append(
                            str(text)
                        )

                else:

                    text = getattr(
                        block,
                        "text",
                        None,
                    )

                    if text:

                        parts.append(
                            str(text)
                        )

            return "\n".join(
                parts
            ).strip()

        return str(
            content
        ).strip()

    # ========================================================
    # CONVERSATION CONTEXT
    # ========================================================

    @classmethod
    def get_conversation_context(
        cls,
        state: RepoSenseState,
        max_messages: int = 6,
        max_chars: int = 3500,
    ) -> str:

        messages = state.get("messages", []) or []
        recent = messages[-max_messages:]
        parts = []
        total = 0

        for message in recent:
            if isinstance(message, HumanMessage):
                role = "USER"
            elif isinstance(message, AIMessage):
                role = "REPOSENSE"
            else:
                role = str(getattr(message, "type", "message")).upper()

            content = cls.extract_text(getattr(message, "content", ""))
            if not content:
                continue

            remaining = max_chars - total
            if remaining <= 0:
                break

            content = content[:remaining]
            block = f"{role}: {content}"
            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    @classmethod
    def build_retrieval_query(
        cls,
        state: RepoSenseState,
    ) -> str:

        question = state.get("question", "").strip()
        history = cls.get_conversation_context(
            state, max_messages=4, max_chars=2500
        )

        if not history:
            return question

        return (
            "Previous repository conversation:\n\n"
            f"{history}\n\n"
            "Current question:\n\n"
            f"{question}"
        )[:6000]

    @classmethod
    def print_conversation_debug(
        cls,
        state: RepoSenseState,
    ) -> None:

        messages = state.get("messages", []) or []
        print(
            f"[LangGraph] Conversation messages: {len(messages)}"
        )
        if messages:
            context = cls.get_conversation_context(
                state, max_messages=4, max_chars=2000
            )
            print("[LangGraph] Conversation context:")
            print(context)

    # ========================================================
    # CONTEXT LIMITING
    # ========================================================

    @classmethod
    def _limit_document(
        cls,
        document: Document,
        max_chars: int | None = None,
    ) -> Document:

        if max_chars is None:

            max_chars = (
                cls.MAX_DOCUMENT_CHARS
            )

        content = (
            document.page_content
            or ""
        )

        if len(content) <= max_chars:

            return document

        truncated = (
            content[:max_chars]
            + "\n\n"
            "[Code truncated for LLM context limit.]"
        )

        return Document(
            page_content=truncated,
            metadata=document.metadata.copy(),
        )

    @classmethod
    def _limit_documents(
        cls,
        documents: list[Document],
        max_documents: int,
        max_total_chars: int,
    ) -> list[Document]:

        selected = []

        total_chars = 0

        for document in documents:

            if len(selected) >= max_documents:

                break

            limited = cls._limit_document(
                document
            )

            content_length = len(
                limited.page_content
            )

            if (
                total_chars + content_length
                > max_total_chars
            ):

                remaining = (
                    max_total_chars
                    - total_chars
                )

                if remaining <= 500:

                    break

                limited = cls._limit_document(
                    document,
                    max_chars=remaining,
                )

                content_length = len(
                    limited.page_content
                )

            selected.append(
                limited
            )

            total_chars += content_length

            if total_chars >= max_total_chars:

                break

        print(
            "[ContextLimiter] "
            f"Selected {len(selected)} "
            f"documents / "
            f"{total_chars} chars"
        )

        return selected

    @classmethod
    def _build_context(
        cls,
        documents: list[Document],
        max_documents: int,
        max_chars: int,
    ) -> tuple[str, list[Document]]:

        limited_documents = (
            cls._limit_documents(
                documents=documents,
                max_documents=max_documents,
                max_total_chars=max_chars,
            )
        )

        if not limited_documents:

            return "", []

        context = format_retrieved_context(
            limited_documents
        )

        # Final safety limit because the formatter itself
        # may add metadata around the code.
        if len(context) > max_chars:

            context = (
                context[:max_chars]
                + "\n\n"
                "[Context truncated to stay "
                "within the LLM request limit.]"
            )

        print(
            "[ContextLimiter] Final context: "
            f"{len(context)} characters"
        )

        return (
            context,
            limited_documents,
        )

    # ========================================================
    # ROUTER
    # ========================================================

    def route_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        question = state["question"]

        self.print_conversation_debug(state)
        history = self.get_conversation_context(
            state, max_messages=4, max_chars=2500
        )
        routing_input = question
        if history:
            routing_input = (
                f"Conversation history:\n{history}\n\n"
                f"Current question:\n{question}"
            )[:5000]

        print(
            f"\n[LangGraph] Routing: {question}"
        )

        intent = self.router.route(
            routing_input
        )

        print(
            f"[LangGraph] Intent: {intent}"
        )

        return {
            **state,
            "intent": intent,
        }

    # ========================================================
    # SEARCH
    # ========================================================

    def search_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        question = state["question"]
        retrieval_query = self.build_retrieval_query(state)

        print(
            "[LangGraph] Search node"
        )
        print(
            f"[LangGraph] Retrieval query size: {len(retrieval_query)} characters"
        )

        documents = self.indexer.search(
            query=retrieval_query,
            k=5,
        )

        print(
            "[LangGraph] Retrieved "
            f"{len(documents)} documents"
        )

        return {
            **state,
            "documents": documents,
        }

    # ========================================================
    # SEARCH ANSWER
    # ========================================================

    def search_answer_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        documents = state.get(
            "documents",
            [],
        )

        if not documents:

            return {
                **state,
                "answer": (
                    "I could not find relevant "
                    "code in the indexed repository."
                ),
            }

        lines = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            metadata = document.metadata

            file_path = metadata.get(
                "file_path",
                "Unknown",
            )

            symbol = metadata.get(
                "symbol_name",
                "",
            )

            symbol_type = metadata.get(
                "symbol_type",
                "code",
            )

            start_line = metadata.get(
                "start_line",
                "?",
            )

            end_line = metadata.get(
                "end_line",
                "?",
            )

            symbol_text = (
                f" â€” `{symbol}`"
                if symbol
                else ""
            )

            lines.append(
                f"{index}. "
                f"`{file_path}`"
                f"{symbol_text}"
                f" ({symbol_type}, "
                f"lines {start_line}-{end_line})"
            )

        answer = (
            "### ðŸ”Ž Relevant Code\n\n"
            + "\n".join(lines)
        )

        return {
            **state,
            "answer": answer,
        }

    # ========================================================
    # EXPLAIN RETRIEVAL
    # ========================================================

    def explain_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        question = state["question"]
        retrieval_query = self.build_retrieval_query(state)

        print(
            "[LangGraph] Explain retrieval node"
        )
        print(
            f"[LangGraph] Retrieval query size: {len(retrieval_query)} characters"
        )

        documents = self.indexer.search(
            query=retrieval_query,
            k=6,
        )

        print(
            "[LangGraph] Retrieved "
            f"{len(documents)} documents"
        )

        return {
            **state,
            "documents": documents,
        }

    # ========================================================
    # EXPLAIN ANSWER
    # ========================================================

    def explain_answer_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        question = state["question"]

        documents = state.get(
            "documents",
            [],
        )

        print(
            "[LangGraph] Preparing explanation context..."
        )

        context, selected_documents = (
            self._build_context(
                documents=documents,
                max_documents=(
                    self.MAX_EXPLAIN_DOCUMENTS
                ),
                max_chars=(
                    self.MAX_EXPLAIN_CONTEXT_CHARS
                ),
            )
        )

        if not context:

            return {
                **state,
                "answer": (
                    "I could not find enough "
                    "repository code to answer "
                    "this question."
                ),
            }

        print(
            "[LangGraph] Explanation context:"
            f" {len(context)} characters"
        )

        conversation = self.get_conversation_context(
            state, max_messages=6, max_chars=3000
        )

        prompt = f"""
You are RepoSense, an expert software
repository analysis assistant.

You are answering an ongoing conversation about the repository.

PREVIOUS CONVERSATION:
{conversation if conversation else "No previous conversation."}

Explain the user's question using ONLY
the supplied repository evidence.

USER QUESTION:
{question}

REPOSITORY CONTEXT:
{context}

Rules:

1. Use the retrieved code as the source of truth.
2. Do not invent implementation details.
3. Mention exact file paths.
4. Mention symbols when available.
5. Mention line numbers when available.
6. Explain relationships between components.
7. If the context is insufficient, say so clearly.
8. Do not claim to have executed the repository.
9. Keep the explanation concise but useful.

Provide a technically accurate developer-friendly
explanation.
"""

        prompt_chars = len(prompt)

        print(
            "[LangGraph] Explanation prompt:"
            f" {prompt_chars} characters"
        )

        print(
            "[LangGraph] Documents sent to LLM:"
            f" {len(selected_documents)}"
        )

        print(
            "[LangGraph] Generating explanation..."
        )

        try:

            response = self.llm.invoke(
                prompt
            )

        except Exception as exc:

            print(
                "[LangGraph] Explanation LLM error:"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            raise

        answer = self.extract_text(
            response.content
        )

        return {
            **state,
            "answer": answer,
        }

    # ========================================================
    # DEBUG â€” PRIMARY RETRIEVAL
    # ========================================================

    def debug_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        question = state["question"]
        retrieval_query = self.build_retrieval_query(state)

        print(
            "[LangGraph] DEBUG: "
            "retrieving primary code..."
        )
        print(
            f"[LangGraph] DEBUG: Retrieval query size: {len(retrieval_query)} characters"
        )

        documents = self.indexer.search(
            query=retrieval_query,
            k=6,
        )

        print(
            "[LangGraph] DEBUG: "
            f"{len(documents)} primary documents"
        )

        return {
            **state,
            "documents": documents,
        }

    # ========================================================
    # DEBUG â€” CODE GRAPH TRACING
    # ========================================================

    def debug_related_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        documents = state.get(
            "documents",
            [],
        )

        if not documents:

            return {
                **state,
                "related_documents": [],
            }

        if self.code_graph is None:

            print(
                "[LangGraph] DEBUG: "
                "Code graph unavailable."
            )

            return {
                **state,
                "related_documents": [],
            }

        print(
            "[LangGraph] DEBUG: "
            "tracing code relationships..."
        )

        related_nodes = []

        seen = set()

        # ----------------------------------------------------
        # Get symbols from vector retrieval.
        # ----------------------------------------------------

        target_symbols = []

        for document in documents:

            metadata = document.metadata

            symbol = metadata.get(
                "symbol_name",
                "",
            )

            if not symbol:

                continue

            normalized = (
                symbol
                .lower()
                .strip()
            )

            if normalized in seen:

                continue

            seen.add(
                normalized
            )

            target_symbols.append(
                symbol
            )

        print(
            "[LangGraph] DEBUG: "
            f"Target symbols: {target_symbols}"
        )

        # ----------------------------------------------------
        # Trace symbols.
        # ----------------------------------------------------

        for symbol in target_symbols[:5]:

            # ------------------------------------------------
            # Callers
            # ------------------------------------------------

            try:

                callers = (
                    self.code_graph.find_callers(
                        symbol
                    )
                )

                for node in callers:

                    key = node.key

                    existing_keys = {
                        item.key
                        for item in related_nodes
                    }

                    if key not in existing_keys:

                        related_nodes.append(
                            node
                        )

            except Exception as exc:

                print(
                    f"[LangGraph] Caller tracing "
                    f"failed for {symbol}: {exc}"
                )

            # ------------------------------------------------
            # Dependencies
            # ------------------------------------------------

            try:

                dependencies = (
                    self.code_graph.find_dependencies(
                        symbol
                    )
                )

                for node in dependencies:

                    key = node.key

                    existing_keys = {
                        item.key
                        for item in related_nodes
                    }

                    if key not in existing_keys:

                        related_nodes.append(
                            node
                        )

            except Exception as exc:

                print(
                    f"[LangGraph] Dependency tracing "
                    f"failed for {symbol}: {exc}"
                )

            # ------------------------------------------------
            # Related
            # ------------------------------------------------

            try:

                related = (
                    self.code_graph.find_related(
                        symbol
                    )
                )

                for node in related:

                    key = node.key

                    existing_keys = {
                        item.key
                        for item in related_nodes
                    }

                    if key not in existing_keys:

                        related_nodes.append(
                            node
                        )

            except Exception as exc:

                print(
                    f"[LangGraph] Related tracing "
                    f"failed for {symbol}: {exc}"
                )

        # ----------------------------------------------------
        # Limit expansion.
        # ----------------------------------------------------

        related_nodes = (
            related_nodes[
                :self.MAX_RELATED_DOCUMENTS
            ]
        )

        print(
            "[LangGraph] DEBUG: "
            f"Found {len(related_nodes)} "
            "related graph nodes"
        )

        # ----------------------------------------------------
        # Convert CodeNode -> LangChain Document
        # ----------------------------------------------------

        related_documents = []

        for node in related_nodes:

            if node.document is None:

                continue

            original = node.document

            metadata = {
                "file_path": original.file_path,
                "symbol_name": original.symbol_name,
                "symbol_type": original.symbol_type,
                "start_line": original.start_line,
                "end_line": original.end_line,
                "language": original.language,
                "repository": original.repository,
                "relationship_source": "code_graph",
            }

            related_documents.append(
                Document(
                    page_content=original.content,
                    metadata=metadata,
                )
            )

        print(
            "[LangGraph] DEBUG: "
            f"Converted {len(related_documents)} "
            "related documents"
        )

        return {
            **state,
            "related_documents": related_documents,
        }

    # ========================================================
    # DEBUG â€” BUILD EVIDENCE
    # ========================================================

    def debug_evidence_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        primary = state.get(
            "documents",
            [],
        )

        related = state.get(
            "related_documents",
            [],
        )

        print(
            "[LangGraph] DEBUG: "
            "building expanded evidence..."
        )

        # ----------------------------------------------------
        # Limit primary evidence
        # ----------------------------------------------------

        primary_context, primary_selected = (
            self._build_context(
                documents=primary,
                max_documents=(
                    self.MAX_DEBUG_DOCUMENTS
                ),
                max_chars=(
                    self.MAX_DEBUG_CONTEXT_CHARS // 2
                ),
            )
        )

        # ----------------------------------------------------
        # Limit graph evidence
        # ----------------------------------------------------

        related_context, related_selected = (
            self._build_context(
                documents=related,
                max_documents=(
                    self.MAX_RELATED_DOCUMENTS
                ),
                max_chars=(
                    self.MAX_DEBUG_CONTEXT_CHARS // 2
                ),
            )
        )

        evidence = f"""
PRIMARY SEMANTIC RETRIEVAL
==========================

{primary_context}


CODE GRAPH RELATED CODE
=======================

{related_context}
"""

        # ----------------------------------------------------
        # Final hard limit
        # ----------------------------------------------------

        if len(evidence) > self.MAX_DEBUG_CONTEXT_CHARS:

            evidence = (
                evidence[
                    :self.MAX_DEBUG_CONTEXT_CHARS
                ]
                + "\n\n"
                "[Evidence truncated.]"
            )

        print(
            "[LangGraph] DEBUG: "
            f"Primary documents used: "
            f"{len(primary_selected)}"
        )

        print(
            "[LangGraph] DEBUG: "
            f"Related documents used: "
            f"{len(related_selected)}"
        )

        print(
            "[LangGraph] DEBUG: "
            f"Final evidence size: "
            f"{len(evidence)} characters"
        )

        return {
            **state,
            "evidence": evidence,
        }

    # ========================================================
    # DEBUG â€” GROQ ANALYSIS
    # ========================================================

    def debug_analysis_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        question = state["question"]

        evidence = state.get(
            "evidence",
            "",
        )

        print(
            "[LangGraph] DEBUG: "
            "analyzing expanded evidence..."
        )

        # ----------------------------------------------------
        # Safety limit
        # ----------------------------------------------------

        if len(evidence) > self.MAX_DEBUG_CONTEXT_CHARS:

            evidence = (
                evidence[
                    :self.MAX_DEBUG_CONTEXT_CHARS
                ]
                + "\n\n"
                "[Evidence truncated.]"
            )

        conversation = self.get_conversation_context(
            state, max_messages=6, max_chars=2500
        )

        prompt = f"""
You are RepoSense's repository debugging engine.

PREVIOUS CONVERSATION:
{conversation if conversation else "No previous conversation."}

Investigate the user's debugging request using
ONLY the supplied repository evidence.

USER REQUEST:

{question}

REPOSITORY EVIDENCE:

{evidence}

Your job is to determine whether the evidence
supports a real problem.

Do NOT assume a bug exists.

Use exactly one confidence level:

CONFIRMED
LIKELY
POSSIBLE
INSUFFICIENT_EVIDENCE

Return:

CONFIDENCE:
<level>

DIAGNOSIS:
<short diagnosis>

LOCATION:
<exact file/symbol/lines if identifiable>

CODE PATH:
<describe relevant caller/dependency chain>

EVIDENCE:
<specific repository evidence>

WHY:
<explain the reasoning>

SUGGESTED_FIX:
<concrete fix only when justified>

POTENTIAL_SIDE_EFFECTS:
<possible consequences>

RELATED_CODE:
<important related files/symbols>

Rules:

1. Never invent code.
2. Never claim you executed the repository.
3. Never claim a bug is confirmed without evidence.
4. Use exact paths from the supplied evidence.
5. Clearly distinguish fact from inference.
6. If evidence is insufficient, say so.
7. Keep the response technically focused.
"""

        print(
            "[LangGraph] DEBUG: "
            f"Prompt size: {len(prompt)} characters"
        )

        print(
            "[LangGraph] DEBUG: "
            "Generating diagnosis with Groq..."
        )

        try:

            response = self.llm.invoke(
                prompt
            )

        except Exception as exc:

            print(
                "[LangGraph] DEBUG: "
                "Groq analysis failed."
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            raise

        diagnosis = self.extract_text(
            response.content
        )

        match = re.search(
            r"CONFIDENCE:\s*"
            r"(CONFIRMED|LIKELY|POSSIBLE|"
            r"INSUFFICIENT_EVIDENCE)",
            diagnosis,
            flags=re.IGNORECASE,
        )

        if match:

            confidence = (
                match.group(1)
                .upper()
            )

        else:

            confidence = (
                "INSUFFICIENT_EVIDENCE"
            )

        print(
            "[LangGraph] DEBUG: "
            f"Confidence: {confidence}"
        )

        return {
            **state,
            "diagnosis": diagnosis,
            "confidence": confidence,
        }

    # ========================================================
    # DEBUG â€” FINAL ANSWER
    # ========================================================

    def debug_answer_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        diagnosis = state.get(
            "diagnosis",
            "",
        )

        confidence = state.get(
            "confidence",
            "INSUFFICIENT_EVIDENCE",
        )

        documents = state.get(
            "documents",
            [],
        )

        related = state.get(
            "related_documents",
            [],
        )

        all_documents = (
            documents + related
        )

        sources = []

        seen = set()

        for document in all_documents:

            metadata = document.metadata

            file_path = metadata.get(
                "file_path",
                "Unknown",
            )

            symbol = metadata.get(
                "symbol_name",
                "",
            )

            start_line = metadata.get(
                "start_line",
                "?",
            )

            end_line = metadata.get(
                "end_line",
                "?",
            )

            key = (
                file_path,
                symbol,
                start_line,
                end_line,
            )

            if key in seen:

                continue

            seen.add(
                key
            )

            sources.append(
                {
                    "file": file_path,
                    "symbol": symbol,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

        answer = (
            "## ðŸ› Debug Analysis\n\n"
            f"**Confidence:** `{confidence}`\n\n"
            f"{diagnosis}\n\n"
            "---\n\n"
            "### ðŸ“š Repository Sources\n\n"
        )

        if sources:

            for source in sources:

                symbol_text = (
                    f" â€” `{source['symbol']}`"
                    if source["symbol"]
                    else ""
                )

                answer += (
                    f"- `{source['file']}`"
                    f"{symbol_text}"
                    f" "
                    f"(lines "
                    f"{source['start_line']}-"
                    f"{source['end_line']})\n"
                )

        else:

            answer += (
                "No repository sources were found."
            )

        return {
            **state,
            "answer": answer,
            "sources": sources,
        }

    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    def save_conversation_node(
        self,
        state: RepoSenseState,
    ) -> RepoSenseState:

        question = state.get("question", "").strip()
        answer = state.get("answer", "").strip()
        messages = state.get("messages", []) or []

        new_messages = []

        if question:
            last_user = None
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    last_user = self.extract_text(message.content)
                    break

            if last_user != question:
                new_messages.append(
                    HumanMessage(content=question)
                )

        if answer:
            new_messages.append(
                AIMessage(content=answer)
            )

        print(
            f"[LangGraph] Saving {len(new_messages)} conversation messages"
        )

        return {
            **state,
            "messages": new_messages,
        }

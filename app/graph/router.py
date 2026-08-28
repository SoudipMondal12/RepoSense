from __future__ import annotations

from app.llm.groq import GroqLLMService


class RepoSenseRouter:
    """
    Routes repository questions using Groq.

    Supported intents:
        - search
        - explain
        - debug
    """

    VALID_INTENTS = {
        "search",
        "explain",
        "debug",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is required."
            )

        service = GroqLLMService(
            api_key=api_key,
            model=model,
            temperature=0.0,
        )

        self.llm = service.get_model()

    @staticmethod
    def _extract_text(content) -> str:

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    text = item.get("text")

                    if text:
                        parts.append(
                            str(text)
                        )

                else:

                    text = getattr(
                        item,
                        "text",
                        None,
                    )

                    if text:
                        parts.append(
                            str(text)
                        )

            return "\n".join(parts).strip()

        return str(content).strip()

    @staticmethod
    def _normalize_intent(
        value: str,
    ) -> str:

        value = (
            value
            .strip()
            .lower()
            .replace("`", "")
            .replace('"', "")
            .replace("'", "")
            .replace(".", "")
            .strip()
        )

        # Handle accidental verbose responses.
        for intent in (
            "debug",
            "explain",
            "search",
        ):

            if value == intent:
                return intent

        if "debug" in value:
            return "debug"

        if "explain" in value:
            return "explain"

        if "search" in value:
            return "search"

        return "search"

    def route(
        self,
        question: str,
    ) -> str:

        if not question or not question.strip():

            return "search"

        prompt = f"""
You are the intent router for RepoSense,
an AI assistant that helps users understand
and debug GitHub repositories.

Classify the user's question into exactly ONE
of these intents:

search
explain
debug

Definitions:

SEARCH:
Use when the user wants to find code, files,
symbols, implementations, locations, or specific
repository information.

Examples:
- Where is authentication implemented?
- Which file contains the database connection?
- Find the login function.
- Where is AgentRegistry defined?

EXPLAIN:
Use when the user wants an explanation or
understanding of how code works.

Examples:
- Explain the authentication flow.
- How does AgentRegistry work?
- Explain this class.
- How does the database layer work?

DEBUG:
Use when the user wants to find, investigate,
diagnose, or fix a possible error, bug, failure,
exception, or unexpected behavior.

Examples:
- Find the bug in the login function.
- Why does authentication fail?
- What could cause this error?
- Debug the database connection.
- Why is this function returning None?

Return ONLY one word:

search

OR

explain

OR

debug

USER QUESTION:
{question}
"""

        response = self.llm.invoke(
            prompt
        )

        content = self._extract_text(
            response.content
        )

        return self._normalize_intent(
            content
        )
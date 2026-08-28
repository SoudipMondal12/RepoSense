from __future__ import annotations

from app.llm.groq import GroqLLMService


class RepoSenseRAGChain:
    """
    RAG answer generation using Groq.

    Retrieval remains separate from generation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
    ):

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is required."
            )

        service = GroqLLMService(
            api_key=api_key,
            model=model,
            temperature=temperature,
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

    def answer(
        self,
        question: str,
        context: str,
    ) -> str:

        prompt = f"""
You are RepoSense, an AI assistant for
understanding and debugging GitHub repositories.

Answer the user's question using ONLY the
repository context supplied below.

USER QUESTION:
{question}

REPOSITORY CONTEXT:
{context}

Instructions:

1. Use the repository context as the source of truth.
2. Do not invent files, functions, classes, or behavior.
3. Mention exact file paths when relevant.
4. Mention symbols and line numbers when available.
5. If the context is insufficient, clearly say so.
6. Keep the answer technically accurate.
7. Explain your reasoning when useful.

Provide a clear developer-friendly answer.
"""

        response = self.llm.invoke(
            prompt
        )

        return self._extract_text(
            response.content
        )
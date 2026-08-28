from __future__ import annotations

from langchain_groq import ChatGroq


class GroqLLMService:
    """
    Groq LLM service for RepoSense.

    Uses Groq as the generation provider while keeping
    embeddings/retrieval independent from the LLM.
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

        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def get_model(self) -> ChatGroq:
        """
        Create and return the LangChain ChatGroq model.
        """

        return ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
        )
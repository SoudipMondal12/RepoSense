from __future__ import annotations

import time
from typing import List

from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient


class HuggingFaceEmbeddings(Embeddings):
    """
    LangChain-compatible Hugging Face hosted embeddings.

    The model is NOT downloaded locally.
    Embeddings are generated through Hugging Face Inference API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 4,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        timeout: int = 120,
    ):

        if not api_key:
            raise ValueError(
                "HF_TOKEN is required."
            )

        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
            timeout=timeout,
        )

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        text: str,
    ) -> List[float]:

        if not text:
            return []

        result = self._embed([text])

        return result[0]

    # ========================================================
    # DOCUMENT EMBEDDINGS
    # ========================================================

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        if not texts:
            return []

        all_embeddings = []

        total = len(texts)

        for start in range(
            0,
            total,
            self.batch_size,
        ):

            end = min(
                start + self.batch_size,
                total,
            )

            batch = texts[start:end]

            print(
                "[HF Embeddings] "
                f"Embedding {start + 1}-{end} "
                f"of {total}"
            )

            embeddings = self._embed(
                batch
            )

            all_embeddings.extend(
                embeddings
            )

        return all_embeddings

    # ========================================================
    # INTERNAL EMBEDDING
    # ========================================================

    def _embed(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        last_exception = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:

                print(
                    "[HF Embeddings] "
                    f"Request attempt "
                    f"{attempt}/"
                    f"{self.max_retries}"
                )

                result = (
                    self.client.feature_extraction(
                        texts,
                        model=self.model,
                    )
                )

                vectors = self._convert_result(
                    result
                )

                if len(vectors) != len(texts):

                    raise RuntimeError(
                        "Hugging Face returned "
                        f"{len(vectors)} embeddings "
                        f"for {len(texts)} texts."
                    )

                print(
                    "[HF Embeddings] "
                    f"Success - dimension: "
                    f"{len(vectors[0])}"
                )

                return vectors

            except Exception as exc:

                last_exception = exc

                print(
                    "[HF Embeddings] ERROR:"
                )

                print(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                if attempt < self.max_retries:

                    sleep_time = (
                        self.retry_delay
                        * attempt
                    )

                    print(
                        "[HF Embeddings] "
                        f"Retrying in "
                        f"{sleep_time}s..."
                    )

                    time.sleep(
                        sleep_time
                    )

        raise RuntimeError(
            "Hugging Face embedding request "
            f"failed after {self.max_retries} "
            "attempts."
        ) from last_exception

    # ========================================================
    # RESPONSE CONVERSION
    # ========================================================

    @staticmethod
    def _convert_result(
        result,
    ) -> List[List[float]]:

        # --------------------------------------------
        # Python list
        # --------------------------------------------

        if isinstance(
            result,
            list,
        ):

            if not result:

                return []

            # One vector:
            #
            # [0.12, 0.34, ...]
            #
            if isinstance(
                result[0],
                (int, float),
            ):

                return [
                    [
                        float(x)
                        for x in result
                    ]
                ]

            # Multiple vectors:
            #
            # [
            #   [0.1, 0.2, ...],
            #   [0.3, 0.4, ...]
            # ]
            #

            if isinstance(
                result[0],
                list,
            ):

                return [

                    [
                        float(x)
                        for x in vector
                    ]

                    for vector in result
                ]

        # --------------------------------------------
        # NumPy / tensor-like result
        # --------------------------------------------

        if hasattr(
            result,
            "tolist",
        ):

            converted = result.tolist()

            return HuggingFaceEmbeddings._convert_result(
                converted
            )

        raise TypeError(
            "Unsupported Hugging Face "
            "embedding response type: "
            f"{type(result)}"
        )

    # ========================================================
    # VECTOR STORE COMPATIBILITY
    # ========================================================

    def get_embeddings(
        self,
    ) -> Embeddings:

        return self


# ============================================================
# REPOSENSE SERVICE WRAPPER
# ============================================================

class HuggingFaceEmbeddingService:
    """
    RepoSense-compatible embedding service.

    RepositoryVectorStore expects:

        embedding_service.get_embeddings()

    This class provides that interface.
    """

    DEFAULT_MODEL = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        batch_size: int = 4,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        timeout: int = 120,
    ):

        self.api_key = api_key

        self.model = (
            model
            or self.DEFAULT_MODEL
        )

        self.batch_size = batch_size

        self.max_retries = max_retries

        self.retry_delay = retry_delay

        self.timeout = timeout

        self._embeddings = (
            HuggingFaceEmbeddings(

                api_key=self.api_key,

                model=self.model,

                batch_size=self.batch_size,

                max_retries=self.max_retries,

                retry_delay=self.retry_delay,

                timeout=self.timeout,
            )
        )

        print(
            "[HuggingFaceEmbeddingService]"
            " initialized."
        )

        print(
            f"[HF] Model: {self.model}"
        )

        print(
            "[HF] Provider: hf-inference"
        )

        print(
            "[HF] Local download: NO"
        )

    # ========================================================
    # REQUIRED METHOD
    # ========================================================

    def get_embeddings(
        self,
    ) -> Embeddings:

        return self._embeddings

    # ========================================================
    # DIRECT METHODS
    # ========================================================

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        return (
            self._embeddings
            .embed_documents(
                texts
            )
        )

    def embed_query(
        self,
        text: str,
    ) -> List[float]:

        return (
            self._embeddings
            .embed_query(
                text
            )
        )
from __future__ import annotations

import hashlib
import re
from typing import Any

from langchain_core.documents import Document
from langchain_chroma import Chroma


class RepositoryVectorStore:
    """
    Repository-isolated Chroma vector store.

    Every repository receives its own collection.

    Example:

        owner/repo-a
            ↓
        reposense_<hash-a>

        owner/repo-b
            ↓
        reposense_<hash-b>

    This prevents vectors from different repositories
    from being mixed together.
    """

    def __init__(
        self,
        embedding_service,
        owner: str,
        repo: str,
        branch: str = "main",
        persist_directory: str = "data/vectorstore",
    ):

        self.embedding_service = embedding_service

        self.owner = owner
        self.repo = repo
        self.branch = branch

        self.persist_directory = (
            persist_directory
        )

        self.repository_key = self._make_repository_key()

        self.collection_name = (
            f"reposense_{self.repository_key}"
        )

        self.vector_store = (
            Chroma(
                collection_name=self.collection_name,
                embedding_function=(
                    embedding_service.get_embeddings()
                ),
                persist_directory=(
                    self.persist_directory
                ),
            )
        )


    # ========================================================
    # REPOSITORY ID
    # ========================================================

    def _make_repository_key(self) -> str:

        raw = (
            f"{self.owner.lower()}/"
            f"{self.repo.lower()}/"
            f"{self.branch.lower()}"
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:20]


    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        documents: list[Document],
    ) -> int:

        if not documents:

            return 0


        ids = []

        for index, document in enumerate(
            documents
        ):

            metadata = (
                document.metadata
            )

            file_path = metadata.get(
                "file_path",
                "unknown",
            )

            symbol_name = metadata.get(
                "symbol_name",
                "",
            )

            start_line = metadata.get(
                "start_line",
                0,
            )

            raw_id = (
                f"{self.owner}/"
                f"{self.repo}/"
                f"{file_path}/"
                f"{symbol_name}/"
                f"{start_line}/"
                f"{index}"
            )

            document_id = hashlib.sha256(
                raw_id.encode("utf-8")
            ).hexdigest()

            ids.append(
                document_id
            )


        self.vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

        return len(documents)


    # ========================================================
    # CLEAR REPOSITORY INDEX
    # ========================================================

    def clear(self) -> None:

        collection = (
            self.vector_store._collection
        )

        existing = collection.get()

        ids = existing.get(
            "ids",
            [],
        )

        if ids:

            collection.delete(
                ids=ids
            )


    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def semantic_search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[Document, float]]:

        if not query.strip():

            return []


        results = (
            self.vector_store
            .similarity_search_with_score(
                query,
                k=k,
            )
        )

        return results


    # ========================================================
    # LOAD ALL DOCUMENTS
    # ========================================================

    def get_all_documents(
        self,
    ) -> list[Document]:

        collection = (
            self.vector_store._collection
        )

        data = collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        documents = []

        raw_documents = (
            data.get(
                "documents",
                [],
            )
        )

        raw_metadatas = (
            data.get(
                "metadatas",
                [],
            )
        )


        for content, metadata in zip(
            raw_documents,
            raw_metadatas,
        ):

            if content is None:

                continue

            if metadata is None:

                metadata = {}


            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )


        return documents


    # ========================================================
    # CODE-AWARE TOKENIZATION
    # ========================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:

        if not text:

            return set()


        text = text.lower()


        # Preserve identifiers such as:
        #
        # authenticate_user
        # createJWTToken
        # database_connection

        parts = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*",
            text,
        )


        tokens = set()


        for part in parts:

            tokens.add(part)

            # Split snake_case
            for piece in part.split("_"):

                if len(piece) >= 2:

                    tokens.add(piece.lower())


            # Split camelCase approximately
            camel_parts = re.findall(
                r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+",
                part,
            )


            for piece in camel_parts:

                if len(piece) >= 2:

                    tokens.add(
                        piece.lower()
                    )


        return tokens


    # ========================================================
    # LEXICAL SCORE
    # ========================================================

    def _lexical_score(
        self,
        query: str,
        document: Document,
    ) -> float:

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:

            return 0.0


        metadata = (
            document.metadata
        )


        symbol_name = str(
            metadata.get(
                "symbol_name",
                "",
            )
        )


        file_path = str(
            metadata.get(
                "file_path",
                "",
            )
        )


        symbol_type = str(
            metadata.get(
                "symbol_type",
                "",
            )
        )


        content = (
            document.page_content
        )


        # ----------------------------------------------------
        # Different importance for code fields
        # ----------------------------------------------------

        symbol_tokens = self._tokenize(
            symbol_name
        )

        file_tokens = self._tokenize(
            file_path
        )

        type_tokens = self._tokenize(
            symbol_type
        )

        content_tokens = self._tokenize(
            content
        )


        # ----------------------------------------------------
        # Exact symbol matching gets highest importance
        # ----------------------------------------------------

        symbol_overlap = (
            len(
                query_tokens
                & symbol_tokens
            )
            / max(
                len(query_tokens),
                1,
            )
        )


        file_overlap = (
            len(
                query_tokens
                & file_tokens
            )
            / max(
                len(query_tokens),
                1,
            )
        )


        type_overlap = (
            len(
                query_tokens
                & type_tokens
            )
            / max(
                len(query_tokens),
                1,
            )
        )


        content_overlap = (
            len(
                query_tokens
                & content_tokens
            )
            / max(
                len(query_tokens),
                1,
            )
        )


        score = (
            symbol_overlap * 0.50
            +
            file_overlap * 0.25
            +
            content_overlap * 0.20
            +
            type_overlap * 0.05
        )


        return min(
            score,
            1.0,
        )


    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        semantic_k: int = 15,
    ) -> list[Document]:

        if not query.strip():

            return []


        # ====================================================
        # 1. Semantic candidates
        # ====================================================

        semantic_results = (
            self.semantic_search(
                query,
                k=semantic_k,
            )
        )


        # ====================================================
        # 2. All documents for lexical matching
        # ====================================================

        all_documents = (
            self.get_all_documents()
        )


        # ====================================================
        # Candidate dictionary
        # ====================================================

        candidates: dict[str, dict[str, Any]] = {}


        # ----------------------------------------------------
        # Semantic results
        # ----------------------------------------------------

        for rank, (
            document,
            distance,
        ) in enumerate(
            semantic_results,
            start=1,
        ):

            key = self._document_key(
                document
            )


            # Lower Chroma distance = better.
            #
            # Convert to a bounded similarity-like score.

            semantic_score = (
                1.0
                /
                (
                    1.0
                    +
                    max(
                        float(distance),
                        0.0,
                    )
                )
            )


            candidates[key] = {
                "document": document,
                "semantic_score": semantic_score,
                "semantic_rank": rank,
                "lexical_score": 0.0,
                "lexical_rank": None,
            }


        # ====================================================
        # 3. Lexical ranking
        # ====================================================

        lexical_scored = []


        for document in all_documents:

            score = (
                self._lexical_score(
                    query,
                    document,
                )
            )

            lexical_scored.append(
                (
                    document,
                    score,
                )
            )


        lexical_scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )


        # ----------------------------------------------------
        # Add lexical candidates
        # ----------------------------------------------------

        for rank, (
            document,
            lexical_score,
        ) in enumerate(
            lexical_scored[:semantic_k],
            start=1,
        ):

            key = self._document_key(
                document
            )


            if key not in candidates:

                candidates[key] = {
                    "document": document,
                    "semantic_score": 0.0,
                    "semantic_rank": None,
                    "lexical_score": lexical_score,
                    "lexical_rank": rank,
                }

            else:

                candidates[key][
                    "lexical_score"
                ] = lexical_score

                candidates[key][
                    "lexical_rank"
                ] = rank


        # ====================================================
        # 4. Reciprocal Rank Fusion
        # ====================================================

        RRF_K = 60.0


        for item in candidates.values():

            semantic_rank = (
                item["semantic_rank"]
            )

            lexical_rank = (
                item["lexical_rank"]
            )


            semantic_rrf = 0.0

            lexical_rrf = 0.0


            if semantic_rank:

                semantic_rrf = (
                    1.0
                    /
                    (
                        RRF_K
                        +
                        semantic_rank
                    )
                )


            if lexical_rank:

                lexical_rrf = (
                    1.0
                    /
                    (
                        RRF_K
                        +
                        lexical_rank
                    )
                )


            # RRF provides ranking stability.
            #
            # Direct lexical score is added because
            # exact code identifiers are extremely useful.

            final_score = (
                semantic_rrf * 0.55
                +
                lexical_rrf * 0.25
                +
                item["lexical_score"] * 0.20
            )


            item["final_score"] = (
                final_score
            )


        # ====================================================
        # 5. Final ranking
        # ====================================================

        ranked = sorted(
            candidates.values(),
            key=lambda item: item[
                "final_score"
            ],
            reverse=True,
        )


        # ====================================================
        # 6. Return top K
        # ====================================================

        return [
            item["document"]
            for item in ranked[:k]
        ]


    # ========================================================
    # DOCUMENT KEY
    # ========================================================

    @staticmethod
    def _document_key(
        document: Document,
    ) -> str:

        metadata = (
            document.metadata
        )


        return "|".join(
            [
                str(
                    metadata.get(
                        "file_path",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "symbol_name",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "start_line",
                        "",
                    )
                ),
            ]
        )
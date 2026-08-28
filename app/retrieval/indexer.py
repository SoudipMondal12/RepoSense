from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.retrieval.vector_store import (
    RepositoryVectorStore,
)


class RepositoryIndexer:
    """
    High-level repository indexing and retrieval service.
    """

    def __init__(
        self,
        embedding_service,
        owner: str | None = None,
        repo: str | None = None,
        branch: str = "main",
        persist_directory: str = "data/vectorstore",
    ):

        if not owner:

            raise ValueError(
                "Repository owner is required."
            )


        if not repo:

            raise ValueError(
                "Repository name is required."
            )


        self.owner = owner
        self.repo = repo
        self.branch = branch


        # ----------------------------------------------------
        # Ensure local vector directory exists
        # ----------------------------------------------------

        Path(
            persist_directory
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


        # ----------------------------------------------------
        # Repository-specific vector store
        # ----------------------------------------------------

        self.vector_store = (
            RepositoryVectorStore(
                embedding_service=(
                    embedding_service
                ),
                owner=owner,
                repo=repo,
                branch=branch,
                persist_directory=(
                    persist_directory
                ),
            )
        )


    # ========================================================
    # INDEX
    # ========================================================

    def index(
        self,
        documents: list[Document],
        clear_existing: bool = True,
    ) -> int:

        if not documents:

            return 0


        # ----------------------------------------------------
        # Clear previous version of THIS repository only
        # ----------------------------------------------------

        if clear_existing:

            self.vector_store.clear()


        # ----------------------------------------------------
        # Add documents
        # ----------------------------------------------------

        count = (
            self.vector_store.add_documents(
                documents
            )
        )


        return count


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:

        return (
            self.vector_store.hybrid_search(
                query=query,
                k=k,
                semantic_k=max(
                    10,
                    k * 3,
                ),
            )
        )


    # ========================================================
    # SEMANTIC SEARCH ONLY
    # ========================================================

    def semantic_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:

        results = (
            self.vector_store.semantic_search(
                query,
                k=k,
            )
        )

        return [
            document
            for document, _score
            in results
        ]


    # ========================================================
    # REPOSITORY INFO
    # ========================================================

    def get_collection_name(
        self,
    ) -> str:

        return (
            self.vector_store.collection_name
        )
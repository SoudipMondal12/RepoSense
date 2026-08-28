import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import streamlit as st

from app.core.models import (
    CodeDocument,
)

from app.embeddings.gemini import (
    GeminiEmbeddingService,
)

from app.retrieval.indexer import (
    RepositoryIndexer,
)


def main():

    print("=" * 80)
    print(
        "RepoScense - Vector Store Test"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    api_key = st.secrets[
        "GOOGLE_API_KEY"
    ]

    # --------------------------------------------------------
    # Embedding service
    # --------------------------------------------------------

    embedding_service = (
        GeminiEmbeddingService(
            api_key=api_key
        )
    )

    # --------------------------------------------------------
    # Create indexer
    # --------------------------------------------------------

    indexer = RepositoryIndexer(
        embedding_service
    )

    # --------------------------------------------------------
    # Fake code documents for testing
    # --------------------------------------------------------

    documents = [

        CodeDocument(
            id="test_auth",
            repository="test/repository",
            file_path="auth/service.py",
            language="python",
            content="""
def authenticate_user(username, password):
    user = get_user(username)

    if user is None:
        return None

    if verify_password(
        password,
        user.password_hash
    ):
        return create_token(user)

    return None
""",
            start_line=1,
            end_line=15,
            document_type="function",
            symbol_name="authenticate_user",
            symbol_type="function",
        ),

        CodeDocument(
            id="test_database",
            repository="test/repository",
            file_path="database/users.py",
            language="python",
            content="""
def get_user(username):
    return database.query_user(username)
""",
            start_line=1,
            end_line=3,
            document_type="function",
            symbol_name="get_user",
            symbol_type="function",
        ),

        CodeDocument(
            id="test_token",
            repository="test/repository",
            file_path="auth/token.py",
            language="python",
            content="""
def create_token(user):
    return jwt.encode(
        {"user_id": user.id},
        SECRET_KEY
    )
""",
            start_line=1,
            end_line=6,
            document_type="function",
            symbol_name="create_token",
            symbol_type="function",
        ),
    ]

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    print(
        "\nIndexing documents..."
    )

    count = indexer.index(
        documents
    )

    print(
        f"Indexed {count} documents."
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    query = (
        "Where is user authentication "
        "implemented?"
    )

    print(
        f"\nQuery:"
        f" {query}"
    )

    print(
        "\nSearching..."
    )

    results = indexer.search(
        query,
        k=3,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SEARCH RESULTS"
    )

    print(
        "=" * 80
    )

    for index, document in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nResult {index}"
        )

        print(
            f"File:"
            f" {document.metadata.get('file_path')}"
        )

        print(
            f"Symbol:"
            f" {document.metadata.get('symbol_name')}"
        )

        print(
            f"Lines:"
            f" {document.metadata.get('start_line')}"
            f"-"
            f"{document.metadata.get('end_line')}"
        )

        print(
            "\nCode:"
        )

        print(
            document.page_content
        )


if __name__ == "__main__":
    main()
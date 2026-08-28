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


from app.embeddings.gemini import (
    GeminiEmbeddingService,
)


from app.retrieval.indexer import (
    RepositoryIndexer,
)


def print_results(
    title,
    results,
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


    for index, document in enumerate(
        results,
        start=1,
    ):

        metadata = (
            document.metadata
        )


        print(
            f"\n#{index}"
        )

        print(
            f"File: "
            f"{metadata.get('file_path')}"
        )

        print(
            f"Symbol: "
            f"{metadata.get('symbol_name')}"
        )

        print(
            f"Type: "
            f"{metadata.get('symbol_type')}"
        )

        print(
            f"Lines: "
            f"{metadata.get('start_line')}"
            "-"
            f"{metadata.get('end_line')}"
        )


def main():

    print("=" * 80)
    print(
        "RepoSense - Step 4B Retrieval Test"
    )
    print("=" * 80)


    # ========================================================
    # API KEY
    # ========================================================

    api_key = st.secrets[
        "GOOGLE_API_KEY"
    ]


    # ========================================================
    # EMBEDDINGS
    # ========================================================

    print(
        "\nInitializing Gemini embeddings..."
    )

    embedding_service = (
        GeminiEmbeddingService(
            api_key=api_key
        )
    )


    # ========================================================
    # REPOSITORY
    # ========================================================

    owner = "KRISH-619-REY"

    repo = (
        "Enhanced-Question-Answer-Chatbot-"
        "Prototype-with-Multi-Model-LLM-"
        "Benchmarking"
    )

    branch = "main"


    print(
        f"\nRepository:"
        f"\n{owner}/{repo}"
    )


    # ========================================================
    # INDEXER
    # ========================================================

    print(
        "\nLoading repository-specific index..."
    )

    indexer = RepositoryIndexer(
        embedding_service=embedding_service,
        owner=owner,
        repo=repo,
        branch=branch,
    )


    print(
        "\nCollection:"
    )

    print(
        indexer.get_collection_name()
    )


    # ========================================================
    # TEST QUERIES
    # ========================================================

    queries = [

        "Where is authentication implemented?",

        "Where are users retrieved from the database?",

        "How are agents registered?",

        "Where is the AgentRegistry class defined?",

        "Where are API credentials configured?",

    ]


    # ========================================================
    # SEARCH
    # ========================================================

    for query in queries:

        print(
            "\n"
            + "#" * 80
        )

        print(
            f"QUERY: {query}"
        )

        print(
            "#" * 80
        )


        results = (
            indexer.search(
                query=query,
                k=5,
            )
        )


        print_results(
            "HYBRID RESULTS",
            results,
        )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 4B RETRIEVAL TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
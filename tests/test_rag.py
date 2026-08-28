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


from app.graph.workflow import (
    build_repo_sense_graph,
)


def main():

    print("=" * 80)

    print(
        "RepoSense - Step 5B LangGraph Test"
    )

    print("=" * 80)


    # ========================================================
    # API
    # ========================================================

    api_key = st.secrets[
        "GOOGLE_API_KEY"
    ]


    # ========================================================
    # TEST REPOSITORY
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
    # EMBEDDINGS
    # ========================================================

    print(
        "\nInitializing embeddings..."
    )

    embeddings = (
        GeminiEmbeddingService(
            api_key=api_key
        )
    )


    # ========================================================
    # INDEX
    # ========================================================

    print(
        "Loading repository index..."
    )

    indexer = RepositoryIndexer(
        embedding_service=embeddings,
        owner=owner,
        repo=repo,
        branch=branch,
    )


    print(
        "Collection:"
    )

    print(
        indexer.get_collection_name()
    )


    # ========================================================
    # GRAPH
    # ========================================================

    print(
        "\nBuilding LangGraph..."
    )

    graph = build_repo_sense_graph(
        api_key=api_key,
        indexer=indexer,
    )


    print(
        "Graph compiled successfully."
    )


    # ========================================================
    # QUESTIONS
    # ========================================================

    questions = [

        "Where is authentication implemented?",

        "Explain how the AgentRegistry works.",

        "Why could authenticate_user fail?",

    ]


    # ========================================================
    # RUN
    # ========================================================

    for number, question in enumerate(
        questions,
        start=1,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"TEST {number}"
        )

        print(
            f"QUESTION:\n{question}"
        )

        print(
            "=" * 80
        )


        initial_state = {

            "question": question,

            "owner": owner,

            "repo": repo,

            "branch": branch,

        }


        try:

            result = graph.invoke(
                initial_state
            )


            print(
                "\n"
                "ROUTED INTENT:"
            )

            print(
                result.get(
                    "intent"
                )
            )


            print(
                "\nANSWER:"
            )

            print(
                result.get(
                    "answer",
                    "No answer generated.",
                )
            )


            documents = result.get(
                "documents",
                [],
            )


            print(
                "\nSOURCES:"
            )


            for index, document in enumerate(
                documents,
                start=1,
            ):

                metadata = (
                    document.metadata
                )


                print(
                    f"{index}. "
                    f"{metadata.get('file_path')}"
                    f" | "
                    f"{metadata.get('symbol_name')}"
                )


        except Exception as exc:

            print(
                "\nERROR:"
            )

            print(
                repr(exc)
            )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 5B COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
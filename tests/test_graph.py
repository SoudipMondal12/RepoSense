from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LANGCHAIN DOCUMENT
# ============================================================

from langchain_core.documents import Document


# ============================================================
# STREAMLIT SECRETS
# ============================================================

import streamlit as st


# ============================================================
# REPOSENSE IMPORTS
# ============================================================

from app.github.client import (
    GitHubClient,
    GitHubError,
)

from app.ingestion.repository import (
    RepositoryIngestor,
)

from app.embeddings.gemini import (
    GeminiEmbeddingService,
)

from app.retrieval.indexer import (
    RepositoryIndexer,
)

from app.analysis.relationship_graph import (
    CodeRelationshipGraph,
)

from app.graph.workflow import (
    build_repo_sense_graph,
)


# ============================================================
# DEBUG HELPERS
# ============================================================

def print_header(title: str):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_error(message: str, exc: Exception):

    print()
    print("-" * 80)
    print(message)
    print("-" * 80)

    print(
        f"{type(exc).__name__}: {exc}"
    )

    print()

    print("TRACEBACK:")

    traceback.print_exc()


def elapsed(start: float) -> str:

    return f"{time.perf_counter() - start:.2f}s"


# ============================================================
# API KEYS
# ============================================================

def get_google_api_key() -> str:

    try:

        key = st.secrets[
            "GOOGLE_API_KEY"
        ]

    except Exception:

        raise RuntimeError(
            "GOOGLE_API_KEY was not found "
            "in .streamlit/secrets.toml"
        )

    if not key:

        raise RuntimeError(
            "GOOGLE_API_KEY is empty."
        )

    return key


def get_groq_api_key() -> str:

    try:

        key = st.secrets[
            "GROQ_API_KEY"
        ]

    except Exception:

        raise RuntimeError(
            "GROQ_API_KEY was not found "
            "in .streamlit/secrets.toml"
        )

    if not key:

        raise RuntimeError(
            "GROQ_API_KEY is empty."
        )

    return key


# ============================================================
# CODEDOCUMENT -> LANGCHAIN DOCUMENT
# ============================================================

def convert_code_documents(
    code_documents,
) -> list[Document]:

    print()
    print(
        "Converting CodeDocument objects "
        "to LangChain Document objects..."
    )

    print(
        f"Input objects: "
        f"{len(code_documents)}"
    )

    langchain_documents = []

    for index, code_document in enumerate(
        code_documents,
        start=1,
    ):

        try:

            # ------------------------------------------------
            # Get CodeDocument attributes safely
            # ------------------------------------------------

            content = getattr(
                code_document,
                "content",
                None,
            )

            file_path = getattr(
                code_document,
                "file_path",
                "",
            )

            repository = getattr(
                code_document,
                "repository",
                "",
            )

            symbol_name = getattr(
                code_document,
                "symbol_name",
                "",
            )

            symbol_type = getattr(
                code_document,
                "symbol_type",
                "",
            )

            parent_symbol = getattr(
                code_document,
                "parent_symbol",
                "",
            )

            language = getattr(
                code_document,
                "language",
                "",
            )

            start_line = getattr(
                code_document,
                "start_line",
                None,
            )

            end_line = getattr(
                code_document,
                "end_line",
                None,
            )

            document_type = getattr(
                code_document,
                "document_type",
                "",
            )


            # ------------------------------------------------
            # Validate content
            # ------------------------------------------------

            if content is None:

                print(
                    f"WARNING: Document {index} "
                    f"has no content."
                )

                continue


            if not isinstance(
                content,
                str,
            ):

                content = str(
                    content
                )


            if not content.strip():

                print(
                    f"WARNING: Document {index} "
                    f"has empty content."
                )

                continue


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {

                "repository": repository,

                "file_path": file_path,

                "symbol_name": symbol_name,

                "symbol_type": symbol_type,

                "parent_symbol": parent_symbol,

                "language": language,

                "start_line": start_line,

                "end_line": end_line,

                "document_type": document_type,
            }


            # ------------------------------------------------
            # Create LangChain Document
            # ------------------------------------------------

            document = Document(

                page_content=content,

                metadata=metadata,

            )


            langchain_documents.append(
                document
            )


        except Exception as exc:

            print(
                f"WARNING: Could not convert "
                f"document {index}"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )


    print()
    print(
        "Conversion complete."
    )

    print(
        f"Output LangChain documents: "
        f"{len(langchain_documents)}"
    )


    # --------------------------------------------------------
    # Show sample
    # --------------------------------------------------------

    if langchain_documents:

        sample = (
            langchain_documents[0]
        )

        print()
        print(
            "Sample LangChain document:"
        )

        print(
            f"  page_content type: "
            f"{type(sample.page_content).__name__}"
        )

        print(
            f"  page_content length: "
            f"{len(sample.page_content)}"
        )

        print(
            f"  file_path: "
            f"{sample.metadata.get('file_path')}"
        )

        print(
            f"  symbol: "
            f"{sample.metadata.get('symbol_name')}"
        )

        print(
            f"  lines: "
            f"{sample.metadata.get('start_line')}-"
            f"{sample.metadata.get('end_line')}"
        )


    return langchain_documents


# ============================================================
# VECTOR STORE VALIDATION
# ============================================================

def validate_vector_retrieval(
    indexer: RepositoryIndexer,
):

    print()
    print(
        "-" * 80
    )

    print(
        "VECTOR RETRIEVAL SANITY CHECK"
    )

    print(
        "-" * 80
    )


    test_queries = [

        "what is this repository about?",

        "where is authentication implemented?",

        "what does the main application do?",
    ]


    total_results = 0


    for query in test_queries:

        print()
        print(
            f"Query: {query}"
        )


        start = time.perf_counter()


        try:

            results = indexer.search(
                query=query,
                k=5,
            )


            duration = elapsed(
                start
            )


            print(
                f"Results: {len(results)}"
            )

            print(
                f"Time: {duration}"
            )


            if results:

                total_results += len(
                    results
                )


                for index, document in enumerate(
                    results,
                    start=1,
                ):

                    metadata = (
                        document.metadata
                    )


                    print(
                        f"  {index}. "
                        f"{metadata.get('file_path', 'Unknown')}"
                    )


                    symbol = metadata.get(
                        "symbol_name",
                        "",
                    )


                    if symbol:

                        print(
                            f"      Symbol: {symbol}"
                        )


            else:

                print(
                    "  WARNING: No results."
                )


        except Exception as exc:

            print_error(
                "VECTOR SEARCH ERROR",
                exc,
            )


    print()

    if total_results > 0:

        print(
            "VECTOR RETRIEVAL STATUS: PASS"
        )

        return True


    print(
        "VECTOR RETRIEVAL STATUS: FAIL"
    )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = (
        time.perf_counter()
    )


    print_header(
        "RepoSense - Dynamic Repository "
        "+ Groq + LangGraph Test"
    )


    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "\n[1/9] Loading API keys..."
    )


    try:

        google_api_key = (
            get_google_api_key()
        )

        groq_api_key = (
            get_groq_api_key()
        )


        print(
            "Google API key: OK"
        )

        print(
            "Groq API key: OK"
        )


    except Exception as exc:

        print_error(
            "API KEY ERROR",
            exc,
        )

        return


    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n[2/9] Repository input"
    )

    print(
        "-" * 80
    )

    print(
        "Enter ANY public GitHub repository URL."
    )

    print(
        "Example:"
    )

    print(
        "https://github.com/username/repository"
    )


    github_url = input(
        "\nGitHub URL > "
    ).strip()


    if not github_url:

        print(
            "\nERROR: GitHub URL is empty."
        )

        return


    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n[3/9] Connecting to GitHub..."
    )


    github_token = st.secrets.get(
        "GITHUB_TOKEN",
        None,
    )


    github = GitHubClient(
        token=github_token
    )


    try:

        start = time.perf_counter()


        owner, repo = (
            github.parse_repo_url(
                github_url
            )
        )


        print(
            f"Owner: {owner}"
        )

        print(
            f"Repository: {repo}"
        )

        print(
            f"Parsing time: "
            f"{elapsed(start)}"
        )


    except Exception as exc:

        print_error(
            "GITHUB URL ERROR",
            exc,
        )

        return


    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n[4/9] Getting repository information..."
    )


    try:

        start = time.perf_counter()


        repository_info = (
            github.get_repository(
                owner,
                repo,
            )
        )


        branch = (
            repository_info[
                "default_branch"
            ]
        )


        print(
            f"Repository: "
            f"{repository_info.get('full_name')}"
        )


        print(
            f"Default branch: "
            f"{branch}"
        )


        description = (
            repository_info.get(
                "description"
            )
        )


        if description:

            print(
                f"Description: "
                f"{description}"
            )


        print(
            f"Repository lookup time: "
            f"{elapsed(start)}"
        )


    except Exception as exc:

        print_error(
            "REPOSITORY INFORMATION ERROR",
            exc,
        )

        return


    # ========================================================
    # STEP 5
    # ========================================================

    print(
        "\n[5/9] Ingesting repository..."
    )


    try:

        start = time.perf_counter()


        ingestor = RepositoryIngestor(
            github
        )


        code_documents = (
            ingestor.ingest(
                owner=owner,
                repo=repo,
                branch=branch,
                max_files=100,
            )
        )


        print()
        print(
            "Ingestion complete."
        )


        print(
            f"CodeDocument objects: "
            f"{len(code_documents)}"
        )


        print(
            f"Ingestion time: "
            f"{elapsed(start)}"
        )


        if not code_documents:

            print(
                "\nERROR:"
            )

            print(
                "Repository ingestion produced "
                "zero CodeDocument objects."
            )

            return


        # ----------------------------------------------------
        # Show document statistics
        # ----------------------------------------------------

        print()
        print(
            "First 5 ingested documents:"
        )


        for index, document in enumerate(
            code_documents[:5],
            start=1,
        ):

            print(
                f"  {index}. "
                f"{getattr(document, 'file_path', 'Unknown')}"
            )


            print(
                f"      symbol: "
                f"{getattr(document, 'symbol_name', '')}"
            )


            content = getattr(
                document,
                "content",
                "",
            )


            print(
                f"      content length: "
                f"{len(content)}"
            )


    except Exception as exc:

        print_error(
            "INGESTION ERROR",
            exc,
        )

        return


    # ========================================================
    # STEP 6
    # ========================================================

    print(
        "\n[6/9] Building code relationship graph..."
    )


    try:

        start = time.perf_counter()


        code_graph = (
            CodeRelationshipGraph(
                code_documents
            )
        )


        graph_summary = (
            code_graph.summary()
        )


        print(
            "Code graph ready."
        )


        print(
            f"Nodes: "
            f"{graph_summary['nodes']}"
        )


        print(
            f"Relationships: "
            f"{graph_summary['relationships']}"
        )


        relationship_types = (
            graph_summary.get(
                "relationship_types",
                {},
            )
        )


        if relationship_types:

            print()
            print(
                "Relationship types:"
            )


            for (
                relationship_type,
                count,
            ) in relationship_types.items():

                print(
                    f"  {relationship_type}: "
                    f"{count}"
                )


        print(
            f"Graph build time: "
            f"{elapsed(start)}"
        )


    except Exception as exc:

        print_error(
            "CODE GRAPH ERROR",
            exc,
        )

        return


    # ========================================================
    # STEP 7
    # ========================================================

    print(
        "\n[7/9] Building vector index..."
    )


    try:

        start = time.perf_counter()


        # ----------------------------------------------------
        # Gemini embedding service
        # ----------------------------------------------------

        embeddings = (
            GeminiEmbeddingService(
                api_key=google_api_key
            )
        )


        print(
            "Gemini embedding service: OK"
        )


        # ----------------------------------------------------
        # Repository indexer
        # ----------------------------------------------------

        indexer = RepositoryIndexer(

            embedding_service=embeddings,

            owner=owner,

            repo=repo,

            branch=branch,

        )


        print(
            "Repository indexer: OK"
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # CodeDocument != LangChain Document
        #
        # Convert them before indexing.
        # ----------------------------------------------------

        langchain_documents = (
            convert_code_documents(
                code_documents
            )
        )


        if not langchain_documents:

            print(
                "\nERROR:"
            )

            print(
                "Conversion produced zero "
                "LangChain documents."
            )

            return


        # ----------------------------------------------------
        # Index
        # ----------------------------------------------------

        print()
        print(
            "Indexing documents..."
        )


        index_start = (
            time.perf_counter()
        )


        indexed_count = (
            indexer.index(
                documents=langchain_documents,
                clear_existing=True,
            )
        )


        print()
        print(
            f"Documents indexed: "
            f"{indexed_count}"
        )


        print(
            f"Indexing time: "
            f"{elapsed(index_start)}"
        )


        if indexed_count <= 0:

            print(
                "\nERROR:"
            )

            print(
                "Vector store indexed zero documents."
            )

            return


        print(
            f"Total vector-index stage time: "
            f"{elapsed(start)}"
        )


        # ----------------------------------------------------
        # Test retrieval immediately
        # ----------------------------------------------------

        retrieval_ok = (
            validate_vector_retrieval(
                indexer
            )
        )


        if not retrieval_ok:

            print()
            print(
                "STOPPING BEFORE LANGGRAPH."
            )

            print(
                "The vector store is not returning "
                "documents, so LangGraph would also "
                "receive empty context."
            )

            return


    except Exception as exc:

        print_error(
            "VECTOR INDEX ERROR",
            exc,
        )

        return


    # ========================================================
    # STEP 8
    # ========================================================

    print(
        "\n[8/9] Building LangGraph..."
    )


    try:

        start = time.perf_counter()


        graph = build_repo_sense_graph(

            api_key=groq_api_key,

            indexer=indexer,

            code_graph=code_graph,

        )


        print(
            "LangGraph compiled successfully."
        )


        print(
            f"LangGraph build time: "
            f"{elapsed(start)}"
        )


    except Exception as exc:

        print_error(
            "LANGGRAPH BUILD ERROR",
            exc,
        )

        return


    # ========================================================
    # STEP 9
    # ========================================================

    print(
        "\n[9/9] Repository ready."
    )


    print_header(
        "REPOSENSE IS READY"
    )


    print(
        f"\nRepository:"
        f" {owner}/{repo}"
    )


    print(
        f"Branch:"
        f" {branch}"
    )


    print(
        f"Code documents:"
        f" {len(code_documents)}"
    )


    print(
        f"Vector documents:"
        f" {indexed_count}"
    )


    print(
        f"Graph nodes:"
        f" {graph_summary['nodes']}"
    )


    print(
        f"Graph relationships:"
        f" {graph_summary['relationships']}"
    )


    print(
        "\nEmbedding provider:"
        " Google Gemini"
    )


    print(
        "LLM provider:"
        " Groq"
    )


    print(
        "LLM model:"
        " openai/gpt-oss-120b"
    )


    print(
        "\nTotal startup time:"
        f" {elapsed(overall_start)}"
    )


    # ========================================================
    # INTERACTIVE CHAT
    # ========================================================

    print_header(
        "INTERACTIVE REPOSITORY CHAT"
    )


    print(
        "\nAsk questions about:"
    )

    print(
        f"  {owner}/{repo}"
    )


    print(
        "\nType 'exit' to stop."
    )


    while True:

        print(
            "\n"
            + "-" * 80
        )


        question = input(
            "Question > "
        ).strip()


        if not question:

            print(
                "Please enter a question."
            )

            continue


        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "\nRepoSense session ended."
            )

            break


        # ====================================================
        # QUESTION
        # ====================================================

        print()
        print(
            f"Question: {question}"
        )


        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        initial_state = {

            "question": question,

            "owner": owner,

            "repo": repo,

            "branch": branch,

        }


        # ----------------------------------------------------
        # Run graph
        # ----------------------------------------------------

        print(
            "\nRunning LangGraph..."
        )


        graph_start = (
            time.perf_counter()
        )


        try:

            result = graph.invoke(
                initial_state
            )


        except Exception as exc:

            print_error(
                "GRAPH EXECUTION ERROR",
                exc,
            )

            continue


        print()
        print(
            f"Graph execution time: "
            f"{elapsed(graph_start)}"
        )


        # ====================================================
        # INTENT
        # ====================================================

        intent = result.get(
            "intent",
            "unknown",
        )


        print()
        print(
            f"Intent: {intent}"
        )


        # ====================================================
        # PRIMARY DOCUMENTS
        # ====================================================

        primary_documents = (
            result.get(
                "documents",
                [],
            )
        )


        print(
            f"Primary documents: "
            f"{len(primary_documents)}"
        )


        if primary_documents:

            print()
            print(
                "Retrieved sources:"
            )


            for index, document in enumerate(
                primary_documents,
                start=1,
            ):

                metadata = (
                    document.metadata
                )


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


                print(
                    f"\n  {index}. "
                    f"{file_path}"
                )


                if symbol:

                    print(
                        f"      Symbol: "
                        f"{symbol}"
                    )


                print(
                    f"      Lines: "
                    f"{start_line}-{end_line}"
                )


        else:

            print(
                "\nWARNING:"
            )

            print(
                "LangGraph retrieved ZERO documents."
            )


        # ====================================================
        # RELATED DOCUMENTS
        # ====================================================

        related_documents = result.get(
            "related_documents",
            [],
        )


        if related_documents:

            print()
            print(
                "Code graph expansion:"
            )


            print(
                f"Related documents: "
                f"{len(related_documents)}"
            )


            for index, document in enumerate(
                related_documents,
                start=1,
            ):

                metadata = (
                    document.metadata
                )


                print(
                    f"  {index}. "
                    f"{metadata.get('file_path', 'Unknown')}"
                )


                symbol = metadata.get(
                    "symbol_name",
                    "",
                )


                if symbol:

                    print(
                        f"      Symbol: "
                        f"{symbol}"
                    )


        # ====================================================
        # DEBUG INFORMATION
        # ====================================================

        if intent == "debug":

            confidence = result.get(
                "confidence",
                "UNKNOWN",
            )


            evidence = result.get(
                "evidence",
                "",
            )


            print()
            print(
                "Debug information:"
            )


            print(
                f"  Confidence: "
                f"{confidence}"
            )


            print(
                f"  Evidence length: "
                f"{len(evidence)} characters"
            )


        # ====================================================
        # ANSWER
        # ====================================================

        answer = result.get(
            "answer",
            "",
        )


        print_header(
            "REPOSENSE ANSWER"
        )


        if answer:

            print(
                answer
            )

        else:

            print(
                "No answer was returned."
            )


    print()
    print(
        "=" * 80
    )

    print(
        "TEST COMPLETE"
    )

    print(
        "=" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
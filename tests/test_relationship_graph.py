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


from app.github.client import (
    GitHubClient,
)


from app.ingestion.repository import (
    RepositoryIngestor,
)


from app.analysis.relationship_graph import (
    CodeRelationshipGraph,
)


def main():

    print("=" * 80)

    print(
        "RepoSense - Code Relationship Graph Test"
    )

    print("=" * 80)


    # ========================================================
    # API KEY
    # ========================================================

    github_token = (
        st.secrets.get(
            "GITHUB_TOKEN",
            None,
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


    print(
        f"\nRepository:"
        f"\n{owner}/{repo}"
    )


    # ========================================================
    # GITHUB
    # ========================================================

    github = GitHubClient(
        token=github_token
    )


    # ========================================================
    # INGESTION
    # ========================================================

    print(
        "\nIngesting repository..."
    )


    ingestor = RepositoryIngestor(
        github
    )


    branch = (
        github.get_default_branch(
            owner,
            repo,
        )
    )


    documents = ingestor.ingest(
        owner=owner,
        repo=repo,
        branch=branch,
        max_files=100,
    )


    print(
        f"\nDocuments created: "
        f"{len(documents)}"
    )


    # ========================================================
    # BUILD GRAPH
    # ========================================================

    print(
        "\nBuilding code relationship graph..."
    )


    graph = CodeRelationshipGraph(
        documents
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        graph.summary()
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "GRAPH SUMMARY"
    )

    print(
        "=" * 80
    )


    print(
        f"Nodes: "
        f"{summary['nodes']}"
    )


    print(
        f"Relationships: "
        f"{summary['relationships']}"
    )


    print(
        "\nRelationship types:"
    )


    for (
        relationship_type,
        count,
    ) in summary[
        "relationship_types"
    ].items():

        print(
            f"  {relationship_type}: "
            f"{count}"
        )


    # ========================================================
    # SHOW RELATIONSHIPS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SAMPLE RELATIONSHIPS"
    )

    print(
        "=" * 80
    )


    for relationship in (
        graph.relationships[:30]
    ):

        print(
            f"\n"
            f"{relationship.source}"
        )

        print(
            f"   "
            f"--[{relationship.relationship_type}]-->"
        )

        print(
            f"   "
            f"{relationship.target}"
        )

        print(
            f"   confidence="
            f"{relationship.confidence:.2f}"
        )


    # ========================================================
    # TEST SPECIFIC SYMBOL
    # ========================================================

    target_symbol = (
        "authenticate_user"
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"RELATIONSHIP TEST: "
        f"{target_symbol}"
    )

    print(
        "=" * 80
    )


    symbols = graph.find_symbol(
        target_symbol
    )


    print(
        f"\nMatching symbols: "
        f"{len(symbols)}"
    )


    for node in symbols:

        print(
            f"\nSymbol:"
            f" {node.name}"
        )

        print(
            f"File:"
            f" {node.file_path}"
        )

        print(
            f"Lines:"
            f" {node.start_line}-"
            f"{node.end_line}"
        )


    # ========================================================
    # CALLERS
    # ========================================================

    callers = (
        graph.find_callers(
            target_symbol
        )
    )


    print(
        "\nCALLERS:"
    )


    if callers:

        for node in callers:

            print(
                f"  ← "
                f"{node.file_path}"
                f"::{node.name}"
            )

    else:

        print(
            "  No callers found."
        )


    # ========================================================
    # DEPENDENCIES
    # ========================================================

    dependencies = (
        graph.find_dependencies(
            target_symbol
        )
    )


    print(
        "\nDEPENDENCIES:"
    )


    if dependencies:

        for node in dependencies:

            print(
                f"  → "
                f"{node.file_path}"
                f"::{node.name}"
            )

    else:

        print(
            "  No dependencies found."
        )


    # ========================================================
    # RELATED
    # ========================================================

    related = (
        graph.find_related(
            target_symbol
        )
    )


    print(
        "\nRELATED SYMBOLS:"
    )


    if related:

        for node in related[:20]:

            print(
                f"  ↔ "
                f"{node.file_path}"
                f"::{node.name}"
            )

    else:

        print(
            "  No related symbols found."
        )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "RELATIONSHIP GRAPH TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
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


from app.graph.router import (
    RepoSenseRouter,
)


def main():

    print("=" * 80)

    print(
        "RepoSense - LangGraph Router Test"
    )

    print("=" * 80)


    # ========================================================
    # API KEY
    # ========================================================

    api_key = st.secrets[
        "GOOGLE_API_KEY"
    ]


    # ========================================================
    # ROUTER
    # ========================================================

    print(
        "\nInitializing router..."
    )

    router = RepoSenseRouter(
        api_key=api_key
    )


    print(
        "Router initialized."
    )


    # ========================================================
    # TEST QUESTIONS
    # ========================================================

    questions = [

        (
            "Where is authentication "
            "implemented?"
        ),

        (
            "Explain how the "
            "authentication flow works."
        ),

        (
            "Why does authenticate_user "
            "fail when the user does not exist?"
        ),

        (
            "Which file contains the "
            "database connection?"
        ),

        (
            "Explain the AgentRegistry class."
        ),

        (
            "Find the bug in the login function."
        ),

    ]


    # ========================================================
    # RUN TESTS
    # ========================================================

    for number, question in enumerate(
        questions,
        start=1,
    ):

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Test {number}"
        )

        print(
            f"Question: {question}"
        )


        try:

            intent = router.route(
                question
            )


            print(
                f"Intent: {intent}"
            )


            if intent == "search":

                print(
                    "✓ SEARCH"
                )

            elif intent == "explain":

                print(
                    "✓ EXPLAIN"
                )

            elif intent == "debug":

                print(
                    "✓ DEBUG"
                )

            else:

                print(
                    "✗ INVALID"
                )


        except Exception as exc:

            print(
                f"✗ ERROR: {exc}"
            )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "ROUTER TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
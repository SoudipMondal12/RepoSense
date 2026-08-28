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


def main():

    print("=" * 80)
    print(
        "RepoScense - Gemini Embedding Test"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Get API key from Streamlit secrets
    # --------------------------------------------------------

    try:

        api_key = st.secrets[
            "GOOGLE_API_KEY"
        ]

    except Exception:

        print(
            "\nERROR:"
        )

        print(
            "GOOGLE_API_KEY was not found "
            "in .streamlit/secrets.toml"
        )

        print(
            "\nExpected:"
        )

        print(
            'GOOGLE_API_KEY = "YOUR_KEY"'
        )

        return

    # --------------------------------------------------------
    # Create embedding service
    # --------------------------------------------------------

    print(
        "\nCreating Gemini embedding client..."
    )

    service = GeminiEmbeddingService(
        api_key=api_key
    )

    # --------------------------------------------------------
    # Test query embedding
    # --------------------------------------------------------

    text = (
        "Find the function responsible "
        "for user authentication."
    )

    print(
        "\nGenerating embedding..."
    )

    vector = service.embed_query(
        text
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        "\nEmbedding generated successfully!"
    )

    print(
        f"\nModel:"
        f" {service.MODEL_NAME}"
    )

    print(
        f"Dimension:"
        f" {len(vector)}"
    )

    print(
        "\nFirst 10 values:"
    )

    print(
        vector[:10]
    )

    print(
        "\nNo local embedding model was used."
    )

    print(
        "Embedding was generated through "
        "the Gemini API."
    )


if __name__ == "__main__":
    main()
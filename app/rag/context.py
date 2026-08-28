from __future__ import annotations

from langchain_core.documents import Document


def format_retrieved_context(
    documents: list[Document],
) -> str:

    if not documents:

        return (
            "No relevant repository code "
            "was retrieved."
        )

    sections = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = document.metadata

        file_path = metadata.get(
            "file_path",
            "Unknown file",
        )

        symbol_name = metadata.get(
            "symbol_name",
            "",
        )

        symbol_type = metadata.get(
            "symbol_type",
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

        language = metadata.get(
            "language",
            "text",
        )

        section = f"""
--- Retrieved Code {index} ---

File: {file_path}

Symbol: {symbol_name or "Unknown"}

Type: {symbol_type or "code"}

Lines: {start_line}-{end_line}

Language: {language}

Code:

{document.page_content}

--- End Retrieved Code {index} ---
"""

        sections.append(
            section.strip()
        )

    return "\n\n".join(
        sections
    )
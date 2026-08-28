from __future__ import annotations

from langchain_core.documents import Document

from app.core.models import CodeDocument


def code_document_to_langchain_document(
    code_document: CodeDocument,
) -> Document:
    """
    Convert our internal CodeDocument into
    a LangChain Document.
    """

    metadata = {
        "id": code_document.id,
        "repository": code_document.repository,
        "file_path": code_document.file_path,
        "language": code_document.language,
        "start_line": code_document.start_line,
        "end_line": code_document.end_line,
        "document_type": code_document.document_type,
        "symbol_name": (
            code_document.symbol_name
            or ""
        ),
        "symbol_type": (
            code_document.symbol_type
            or ""
        ),
        "parent_symbol": (
            code_document.parent_symbol
            or ""
        ),
    }

    return Document(
        page_content=code_document.content,
        metadata=metadata,
    )


def convert_code_documents(
    code_documents: list[CodeDocument],
) -> list[Document]:

    return [
        code_document_to_langchain_document(
            document
        )
        for document in code_documents
    ]
from __future__ import annotations

from app.core.models import CodeDocument


EMBEDDABLE_TYPES = {
    "class",
    "function",
    "method",
}


def get_embeddable_documents(
    documents: list[CodeDocument],
) -> list[CodeDocument]:

    result = []

    for document in documents:

        if document.document_type in (
            EMBEDDABLE_TYPES
        ):

            if document.content.strip():

                result.append(
                    document
                )

    return result
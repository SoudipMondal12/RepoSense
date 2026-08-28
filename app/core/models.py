from __future__ import annotations

from pydantic import BaseModel, Field


class RepositoryInfo(BaseModel):
    owner: str
    name: str
    full_name: str
    description: str | None = None
    default_branch: str
    language: str | None = None
    stars: int
    forks: int
    is_private: bool


class RepositoryFile(BaseModel):
    path: str
    size: int
    sha: str
    language: str | None = None
    content: str | None = None


class CodeSymbol(BaseModel):
    """
    Represents a function, class, method, or other
    identifiable code symbol.
    """

    name: str
    symbol_type: str

    file_path: str

    start_line: int
    end_line: int

    parent: str | None = None

    signature: str | None = None

    source_code: str = ""


class CodeDocument(BaseModel):
    """
    Structured representation of a piece of source code.

    This will later become the input to our embedding pipeline.
    """

    id: str

    repository: str

    file_path: str

    language: str

    content: str

    start_line: int = 1
    end_line: int = 1

    document_type: str = "file"

    symbol_name: str | None = None

    symbol_type: str | None = None

    parent_symbol: str | None = None

    imports: list[str] = Field(
        default_factory=list
    )

    metadata: dict = Field(
        default_factory=dict
    )
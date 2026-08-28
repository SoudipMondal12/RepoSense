from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from app.core.models import CodeDocument


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".vue": "vue",
    ".dart": "dart",
}


def detect_language(file_path: str) -> str:
    """
    Detect programming language from file extension.
    """

    extension = Path(file_path).suffix.lower()

    return LANGUAGE_MAP.get(
        extension,
        "unknown",
    )


def create_document_id(
    repository: str,
    file_path: str,
    symbol_name: str | None = None,
) -> str:
    """
    Create a deterministic ID for a code document.
    """

    raw = (
        f"{repository}:"
        f"{file_path}:"
        f"{symbol_name or 'file'}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def get_source_segment(
    source_lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    """
    Extract source code using 1-based line numbers.
    """

    start_index = max(
        start_line - 1,
        0,
    )

    end_index = min(
        end_line,
        len(source_lines),
    )

    return "".join(
        source_lines[start_index:end_index]
    )


def get_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """
    Generate a readable function signature.
    """

    arguments = []

    for arg in node.args.args:

        if arg.annotation:

            try:
                annotation = ast.unparse(
                    arg.annotation
                )
            except Exception:
                annotation = "..."
        else:
            annotation = None

        if annotation:
            arguments.append(
                f"{arg.arg}: {annotation}"
            )
        else:
            arguments.append(
                arg.arg
            )

    if node.args.vararg:
        arguments.append(
            f"*{node.args.vararg.arg}"
        )

    if node.args.kwarg:
        arguments.append(
            f"**{node.args.kwarg.arg}"
        )

    return_annotation = ""

    if node.returns:

        try:
            return_annotation = (
                " -> "
                + ast.unparse(
                    node.returns
                )
            )
        except Exception:
            pass

    prefix = (
        "async def"
        if isinstance(
            node,
            ast.AsyncFunctionDef,
        )
        else "def"
    )

    return (
        f"{prefix} "
        f"{node.name}"
        f"({', '.join(arguments)})"
        f"{return_annotation}"
    )


class PythonCodeParser:
    """
    AST-based Python source parser.

    Extracts:
    - imports
    - classes
    - functions
    - methods
    - source line ranges
    """

    def parse(
        self,
        repository: str,
        file_path: str,
        source: str,
    ) -> list[CodeDocument]:

        documents: list[CodeDocument] = []

        try:
            tree = ast.parse(
                source,
                filename=file_path,
            )

        except SyntaxError as exc:

            # If a file cannot be parsed,
            # preserve it as a normal document.
            documents.append(
                CodeDocument(
                    id=create_document_id(
                        repository,
                        file_path,
                    ),
                    repository=repository,
                    file_path=file_path,
                    language="python",
                    content=source,
                    start_line=1,
                    end_line=max(
                        len(source.splitlines()),
                        1,
                    ),
                    document_type="file",
                    metadata={
                        "parse_error": str(exc),
                    },
                )
            )

            return documents

        source_lines = source.splitlines(
            keepends=True
        )

        imports = self._extract_imports(
            tree
        )

        # Add file-level document.
        documents.append(
            CodeDocument(
                id=create_document_id(
                    repository,
                    file_path,
                ),
                repository=repository,
                file_path=file_path,
                language="python",
                content=source,
                start_line=1,
                end_line=max(
                    len(source_lines),
                    1,
                ),
                document_type="file",
                imports=imports,
                metadata={
                    "parser": "python-ast",
                },
            )
        )

        # Extract classes/functions/methods.
        for node in ast.walk(tree):

            if isinstance(
                node,
                ast.ClassDef,
            ):

                documents.append(
                    self._class_document(
                        repository,
                        file_path,
                        source_lines,
                        node,
                        imports,
                    )
                )

            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):

                parent = self._find_parent_class(
                    tree,
                    node,
                )

                documents.append(
                    self._function_document(
                        repository,
                        file_path,
                        source_lines,
                        node,
                        imports,
                        parent,
                    )
                )

        return documents

    def _extract_imports(
        self,
        tree: ast.AST,
    ) -> list[str]:

        imports = []

        for node in ast.walk(tree):

            if isinstance(
                node,
                ast.Import,
            ):

                for alias in node.names:
                    imports.append(
                        f"import {alias.name}"
                    )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):

                module = node.module or ""

                names = ", ".join(
                    alias.name
                    for alias in node.names
                )

                if node.level:
                    prefix = "." * node.level
                    imports.append(
                        f"from {prefix}{module} "
                        f"import {names}"
                    )
                else:
                    imports.append(
                        f"from {module} "
                        f"import {names}"
                    )

        return sorted(
            set(imports)
        )

    def _class_document(
        self,
        repository: str,
        file_path: str,
        source_lines: list[str],
        node: ast.ClassDef,
        imports: list[str],
    ) -> CodeDocument:

        start_line = node.lineno

        end_line = (
            node.end_lineno
            or node.lineno
        )

        source_code = get_source_segment(
            source_lines,
            start_line,
            end_line,
        )

        return CodeDocument(
            id=create_document_id(
                repository,
                file_path,
                node.name,
            ),
            repository=repository,
            file_path=file_path,
            language="python",
            content=source_code,
            start_line=start_line,
            end_line=end_line,
            document_type="class",
            symbol_name=node.name,
            symbol_type="class",
            imports=imports,
            metadata={
                "parser": "python-ast",
            },
        )

    def _function_document(
        self,
        repository: str,
        file_path: str,
        source_lines: list[str],
        node: ast.FunctionDef
        | ast.AsyncFunctionDef,
        imports: list[str],
        parent: str | None,
    ) -> CodeDocument:

        start_line = node.lineno

        end_line = (
            node.end_lineno
            or node.lineno
        )

        source_code = get_source_segment(
            source_lines,
            start_line,
            end_line,
        )

        signature = get_function_signature(
            node
        )

        symbol_name = node.name

        if parent:
            symbol_name = (
                f"{parent}.{node.name}"
            )

        return CodeDocument(
            id=create_document_id(
                repository,
                file_path,
                symbol_name,
            ),
            repository=repository,
            file_path=file_path,
            language="python",
            content=source_code,
            start_line=start_line,
            end_line=end_line,
            document_type="method"
            if parent
            else "function",
            symbol_name=symbol_name,
            symbol_type="method"
            if parent
            else "function",
            parent_symbol=parent,
            imports=imports,
            metadata={
                "parser": "python-ast",
                "signature": signature,
            },
        )

    def _find_parent_class(
        self,
        tree: ast.AST,
        target_node: ast.AST,
    ) -> str | None:

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.ClassDef,
            ):
                continue

            for child in node.body:

                if child is target_node:

                    return node.name

        return None


def parse_source_file(
    repository: str,
    file_path: str,
    source: str,
) -> list[CodeDocument]:
    """
    Parse source code according to language.
    """

    language = detect_language(
        file_path
    )

    if language == "python":

        parser = PythonCodeParser()

        return parser.parse(
            repository,
            file_path,
            source,
        )

    # For non-Python languages,
    # temporarily create a file-level document.
    #
    # We will add Tree-sitter based parsing later.
    return [
        CodeDocument(
            id=create_document_id(
                repository,
                file_path,
            ),
            repository=repository,
            file_path=file_path,
            language=language,
            content=source,
            start_line=1,
            end_line=max(
                len(source.splitlines()),
                1,
            ),
            document_type="file",
            metadata={
                "parser": "generic"
            },
        )
    ]
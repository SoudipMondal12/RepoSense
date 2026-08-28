from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field

from app.core.models import CodeDocument


@dataclass
class CodeRelationship:
    """
    Represents a relationship between two code symbols.
    """

    source: str
    target: str
    relationship_type: str

    source_file: str = ""
    target_file: str = ""

    confidence: float = 1.0


@dataclass
class CodeNode:
    """
    Represents a code symbol inside a repository.
    """

    key: str

    name: str

    file_path: str

    symbol_type: str

    start_line: int

    end_line: int

    parent: str | None = None

    document: CodeDocument | None = None


class CodeRelationshipGraph:
    """
    Lightweight repository code relationship graph.

    Current supported relationships:

    - calls
    - imports
    - inherits
    - contains

    The graph is built from the CodeDocument objects
    already produced by RepoSense's parser.
    """

    def __init__(
        self,
        documents: list[CodeDocument] | None = None,
    ):

        self.nodes: dict[str, CodeNode] = {}

        self.relationships: list[
            CodeRelationship
        ] = []

        self.outgoing: dict[
            str,
            list[CodeRelationship],
        ] = defaultdict(list)

        self.incoming: dict[
            str,
            list[CodeRelationship],
        ] = defaultdict(list)

        self.name_index: dict[
            str,
            list[str],
        ] = defaultdict(list)

        self.file_index: dict[
            str,
            list[str],
        ] = defaultdict(list)

        if documents:

            self.build(documents)

    # ========================================================
    # BUILD
    # ========================================================

    def build(
        self,
        documents: list[CodeDocument],
    ) -> None:

        self.clear()

        # ----------------------------------------------------
        # First pass:
        # register all symbols
        # ----------------------------------------------------

        for document in documents:

            symbol_name = (
                document.symbol_name
            )

            if not symbol_name:

                continue

            key = self.make_key(
                document.file_path,
                symbol_name,
            )

            node = CodeNode(
                key=key,
                name=symbol_name,
                file_path=document.file_path,
                symbol_type=(
                    document.symbol_type
                    or document.document_type
                ),
                start_line=document.start_line,
                end_line=document.end_line,
                parent=document.parent_symbol,
                document=document,
            )

            self.nodes[key] = node

            self.name_index[
                self.normalize_name(
                    symbol_name
                )
            ].append(key)

            self.file_index[
                document.file_path
            ].append(key)

        # ----------------------------------------------------
        # Second pass:
        # relationships
        # ----------------------------------------------------

        for node in self.nodes.values():

            document = node.document

            if document is None:

                continue

            self._add_contains_relationship(
                node
            )

            self._add_import_relationships(
                node
            )

            if document.language == "python":

                self._add_python_relationships(
                    node
                )

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:

        self.nodes.clear()

        self.relationships.clear()

        self.outgoing.clear()

        self.incoming.clear()

        self.name_index.clear()

        self.file_index.clear()

    # ========================================================
    # NODE KEY
    # ========================================================

    @staticmethod
    def make_key(
        file_path: str,
        symbol_name: str,
    ) -> str:

        return (
            f"{file_path}::"
            f"{symbol_name}"
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def normalize_name(
        name: str,
    ) -> str:

        name = name.strip()

        # Methods can arrive as:
        #
        # ClassName.method
        #
        # For call matching, also index the final name.

        if "." in name:

            name = name.split(
                "."
            )[-1]

        return name.lower()

    # ========================================================
    # ADD RELATIONSHIP
    # ========================================================

    def add_relationship(
        self,
        relationship: CodeRelationship,
    ) -> None:

        # ----------------------------------------------------
        # Prevent duplicates
        # ----------------------------------------------------

        for existing in self.relationships:

            if (
                existing.source
                == relationship.source
                and
                existing.target
                == relationship.target
                and
                existing.relationship_type
                == relationship.relationship_type
            ):

                return

        self.relationships.append(
            relationship
        )

        self.outgoing[
            relationship.source
        ].append(
            relationship
        )

        self.incoming[
            relationship.target
        ].append(
            relationship
        )

    # ========================================================
    # CONTAINS
    # ========================================================

    def _add_contains_relationship(
        self,
        node: CodeNode,
    ) -> None:

        if not node.parent:

            return

        parent_candidates = (
            self.name_index.get(
                self.normalize_name(
                    node.parent
                ),
                [],
            )
        )

        for parent_key in parent_candidates:

            parent_node = self.nodes.get(
                parent_key
            )

            if parent_node is None:

                continue

            if (
                parent_node.file_path
                != node.file_path
            ):

                continue

            self.add_relationship(
                CodeRelationship(
                    source=parent_key,
                    target=node.key,
                    relationship_type="contains",
                    source_file=parent_node.file_path,
                    target_file=node.file_path,
                )
            )

    # ========================================================
    # IMPORTS
    # ========================================================

    def _add_import_relationships(
        self,
        node: CodeNode,
    ) -> None:

        document = node.document

        if document is None:

            return

        for import_statement in (
            document.imports
        ):

            # ------------------------------------------------
            # We store imports as relationships from the
            # current symbol to the imported module/name.
            #
            # Example:
            #
            # from database.users import get_user
            #
            # becomes:
            #
            # current_symbol
            #       ↓ imports
            # get_user
            # ------------------------------------------------

            imported_names = (
                self.extract_import_names(
                    import_statement
                )
            )

            for imported_name in (
                imported_names
            ):

                candidates = (
                    self.name_index.get(
                        self.normalize_name(
                            imported_name
                        ),
                        [],
                    )
                )

                for target_key in candidates:

                    target_node = (
                        self.nodes.get(
                            target_key
                        )
                    )

                    if target_node is None:

                        continue

                    if (
                        target_node.file_path
                        == node.file_path
                    ):

                        continue

                    self.add_relationship(
                        CodeRelationship(
                            source=node.key,
                            target=target_key,
                            relationship_type="imports",
                            source_file=node.file_path,
                            target_file=target_node.file_path,
                            confidence=0.85,
                        )
                    )

    # ========================================================
    # PYTHON RELATIONSHIPS
    # ========================================================

    def _add_python_relationships(
        self,
        node: CodeNode,
    ) -> None:

        document = node.document

        if document is None:

            return

        source = document.content

        if not source.strip():

            return

        try:

            tree = ast.parse(
                source
            )

        except SyntaxError:

            return

        # ----------------------------------------------------
        # Inheritance
        # ----------------------------------------------------

        if (
            node.symbol_type
            == "class"
        ):

            self._extract_inheritance(
                node,
                tree,
            )

        # ----------------------------------------------------
        # Function/method calls
        # ----------------------------------------------------

        if node.symbol_type in {
            "function",
            "method",
        }:

            self._extract_calls(
                node,
                tree,
            )

    # ========================================================
    # INHERITANCE
    # ========================================================

    def _extract_inheritance(
        self,
        node: CodeNode,
        tree: ast.AST,
    ) -> None:

        class_nodes = [
            item
            for item in ast.walk(tree)
            if isinstance(
                item,
                ast.ClassDef,
            )
        ]

        for class_node in class_nodes:

            if (
                class_node.name
                != self.base_symbol_name(
                    node.name
                )
            ):

                continue

            for base in class_node.bases:

                base_name = (
                    self.ast_name(
                        base
                    )
                )

                if not base_name:

                    continue

                candidates = (
                    self.name_index.get(
                        self.normalize_name(
                            base_name
                        ),
                        [],
                    )
                )

                for target_key in candidates:

                    target_node = (
                        self.nodes.get(
                            target_key
                        )
                    )

                    if target_node is None:

                        continue

                    if (
                        target_node.file_path
                        == node.file_path
                        and
                        target_key
                        == node.key
                    ):

                        continue

                    self.add_relationship(
                        CodeRelationship(
                            source=node.key,
                            target=target_key,
                            relationship_type="inherits",
                            source_file=node.file_path,
                            target_file=target_node.file_path,
                            confidence=0.95,
                        )
                    )

    # ========================================================
    # CALL EXTRACTION
    # ========================================================

    def _extract_calls(
        self,
        node: CodeNode,
        tree: ast.AST,
    ) -> None:

        for item in ast.walk(tree):

            if not isinstance(
                item,
                ast.Call,
            ):

                continue

            called_name = (
                self.ast_name(
                    item.func
                )
            )

            if not called_name:

                continue

            simple_name = (
                called_name.split(
                    "."
                )[-1]
            )

            candidates = (
                self.name_index.get(
                    self.normalize_name(
                        simple_name
                    ),
                    [],
                )
            )

            if not candidates:

                continue

            for target_key in candidates:

                target_node = (
                    self.nodes.get(
                        target_key
                    )
                )

                if target_node is None:

                    continue

                # Don't link a function to itself.
                if (
                    target_key
                    == node.key
                ):

                    continue

                confidence = (
                    self.call_confidence(
                        called_name,
                        target_node,
                        node,
                    )
                )

                self.add_relationship(
                    CodeRelationship(
                        source=node.key,
                        target=target_key,
                        relationship_type="calls",
                        source_file=node.file_path,
                        target_file=target_node.file_path,
                        confidence=confidence,
                    )
                )

    # ========================================================
    # CALL CONFIDENCE
    # ========================================================

    @staticmethod
    def call_confidence(
        called_name: str,
        target_node: CodeNode,
        source_node: CodeNode,
    ) -> float:

        simple_name = (
            called_name.split(
                "."
            )[-1]
        )

        target_simple_name = (
            target_node.name.split(
                "."
            )[-1]
        )

        if (
            simple_name
            == target_simple_name
        ):

            # Same file is slightly stronger.
            if (
                target_node.file_path
                == source_node.file_path
            ):

                return 0.95

            return 0.80

        return 0.60

    # ========================================================
    # AST NAME
    # ========================================================

    @staticmethod
    def ast_name(
        node: ast.AST,
    ) -> str | None:

        if isinstance(
            node,
            ast.Name,
        ):

            return node.id

        if isinstance(
            node,
            ast.Attribute,
        ):

            parts = []

            current = node

            while isinstance(
                current,
                ast.Attribute,
            ):

                parts.append(
                    current.attr
                )

                current = current.value

            if isinstance(
                current,
                ast.Name,
            ):

                parts.append(
                    current.id
                )

            return ".".join(
                reversed(parts)
            )

        return None

    # ========================================================
    # IMPORT NAME EXTRACTION
    # ========================================================

    @staticmethod
    def extract_import_names(
        import_statement: str,
    ) -> list[str]:

        result = []

        text = (
            import_statement
            .strip()
        )

        if text.startswith(
            "import "
        ):

            value = text[
                len("import "):
            ]

            for part in value.split(
                ","
            ):

                part = part.strip()

                if not part:

                    continue

                result.append(
                    part.split(
                        " as "
                    )[0].strip()
                )

        elif text.startswith(
            "from "
        ):

            try:

                before, after = (
                    text.split(
                        " import ",
                        1,
                    )
                )

                module = before[
                    len("from "):
                ].strip()

                for name in after.split(
                    ","
                ):

                    name = (
                        name.strip()
                    )

                    if not name:

                        continue

                    name = (
                        name.split(
                            " as "
                        )[0]
                        .strip()
                    )

                    result.append(
                        name
                    )

            except ValueError:

                pass

        return result

    # ========================================================
    # BASE SYMBOL NAME
    # ========================================================

    @staticmethod
    def base_symbol_name(
        name: str,
    ) -> str:

        if "." in name:

            return name.split(
                "."
            )[0]

        return name

    # ========================================================
    # FIND NODE
    # ========================================================

    def find_symbol(
        self,
        name: str,
    ) -> list[CodeNode]:

        keys = self.name_index.get(
            self.normalize_name(
                name
            ),
            [],
        )

        return [
            self.nodes[key]
            for key in keys
            if key in self.nodes
        ]

    # ========================================================
    # CALLERS
    # ========================================================

    def find_callers(
        self,
        symbol_name: str,
    ) -> list[CodeNode]:

        targets = self.find_symbol(
            symbol_name
        )

        results = []

        seen = set()


        for target in targets:

            for relationship in (
                self.incoming.get(
                    target.key,
                    [],
                )
            ):

                if (
                    relationship.relationship_type
                    != "calls"
                ):

                    continue

                source = self.nodes.get(
                    relationship.source
                )

                if (
                    source is None
                    or source.key in seen
                ):

                    continue

                seen.add(
                    source.key
                )

                results.append(
                    source
                )


        return results

    # ========================================================
    # CALLEES
    # ========================================================

    def find_dependencies(
        self,
        symbol_name: str,
    ) -> list[CodeNode]:

        sources = self.find_symbol(
            symbol_name
        )

        results = []

        seen = set()


        for source in sources:

            for relationship in (
                self.outgoing.get(
                    source.key,
                    [],
                )
            ):

                if (
                    relationship.relationship_type
                    != "calls"
                ):

                    continue

                target = self.nodes.get(
                    relationship.target
                )

                if (
                    target is None
                    or target.key in seen
                ):

                    continue

                seen.add(
                    target.key
                )

                results.append(
                    target
                )


        return results

    # ========================================================
    # RELATED SYMBOLS
    # ========================================================

    def find_related(
        self,
        symbol_name: str,
    ) -> list[CodeNode]:

        nodes = self.find_symbol(
            symbol_name
        )

        results = []

        seen = set()


        for node in nodes:

            relationships = (
                self.outgoing.get(
                    node.key,
                    [],
                )
                +
                self.incoming.get(
                    node.key,
                    [],
                )
            )


            for relationship in (
                relationships
            ):

                related_key = (
                    relationship.target
                    if relationship.source
                    == node.key
                    else relationship.source
                )


                related = self.nodes.get(
                    related_key
                )


                if (
                    related is None
                    or related.key in seen
                ):

                    continue


                seen.add(
                    related.key
                )

                results.append(
                    related
                )


        return results

    # ========================================================
    # GRAPH SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> dict:

        counts = defaultdict(int)

        for relationship in (
            self.relationships
        ):

            counts[
                relationship.relationship_type
            ] += 1


        return {
            "nodes": len(
                self.nodes
            ),
            "relationships": len(
                self.relationships
            ),
            "relationship_types": dict(
                counts
            ),
        }
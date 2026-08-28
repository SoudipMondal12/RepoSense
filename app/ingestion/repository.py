from __future__ import annotations

from app.analysis.code_parser import (
    parse_source_file,
)
from app.core.models import CodeDocument
from app.github.client import (
    GitHubClient,
    RepositoryFileFilter,
)


class RepositoryIngestor:
    """
    Converts a GitHub repository into structured
    CodeDocument objects.
    """

    def __init__(
        self,
        github_client: GitHubClient,
    ):
        self.github = github_client

    def ingest(
        self,
        owner: str,
        repo: str,
        branch: str | None = None,
        max_files: int = 100,
    ) -> list[CodeDocument]:

        if branch is None:

            branch = (
                self.github.get_default_branch(
                    owner,
                    repo,
                )
            )

        tree = self.github.get_tree(
            owner,
            repo,
            branch,
        )

        repository_name = (
            f"{owner}/{repo}"
        )

        documents: list[CodeDocument] = []

        code_files = [
            item
            for item in tree
            if item.get("type") == "blob"
            and RepositoryFileFilter.should_include(
                item["path"],
                item.get("size", 0),
            )
        ]

        # Prevent extremely large repositories
        # from overwhelming the first version.
        code_files = code_files[
            :max_files
        ]

        for index, item in enumerate(
            code_files,
            start=1,
        ):

            path = item["path"]

            print(
                f"[{index}/{len(code_files)}] "
                f"Reading {path}"
            )

            try:

                content = self.github.get_file(
                    owner,
                    repo,
                    path,
                    branch,
                )

                file_documents = (
                    parse_source_file(
                        repository_name,
                        path,
                        content,
                    )
                )

                documents.extend(
                    file_documents
                )

            except Exception as exc:

                print(
                    f"WARNING: Could not process "
                    f"{path}: {exc}"
                )

        return documents
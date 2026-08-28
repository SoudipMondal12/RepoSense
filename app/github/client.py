from __future__ import annotations

import base64
import re
from typing import Any

import requests


class GitHubError(Exception):
    """GitHub API error."""
    pass


class RepositoryFileFilter:
    """Controls which repository files should be indexed."""

    ALLOWED_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sql",
        ".html",
        ".css",
        ".scss",
        ".vue",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".txt",
    }

    IGNORED_DIRECTORIES = {
        ".git",
        ".github",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        "coverage",
        ".pytest_cache",
    }

    MAX_FILE_SIZE = 500_000

    @classmethod
    def should_include(
        cls,
        path: str,
        size: int = 0,
    ) -> bool:

        if not path:
            return False

        parts = path.replace("\\", "/").split("/")

        for part in parts:

            if part in cls.IGNORED_DIRECTORIES:

                return False

        if size and size > cls.MAX_FILE_SIZE:

            return False

        extension = ""

        if "." in parts[-1]:

            extension = (
                "."
                + parts[-1].rsplit(".", 1)[-1].lower()
            )

        return extension in cls.ALLOWED_EXTENSIONS


class GitHubClient:
    """
    GitHub REST API client.

    Supports public repositories and authenticated
    requests through a GitHub Personal Access Token.
    """

    API_BASE = "https://api.github.com"

    def __init__(
        self,
        token: str | None = None,
        timeout: int = 30,
    ):

        self.token = token

        self.timeout = timeout

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Accept": (
                    "application/vnd.github+json"
                ),

                "X-GitHub-Api-Version":
                    "2022-11-28",

                "User-Agent":
                    "RepoSense",
            }
        )

        if token:

            self.session.headers.update(
                {
                    "Authorization":
                        f"Bearer {token}"
                }
            )


    # ========================================================
    # REQUEST
    # ========================================================

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Any:

        url = (
            endpoint
            if endpoint.startswith("http")
            else f"{self.API_BASE}{endpoint}"
        )

        try:

            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs,
            )

        except requests.RequestException as exc:

            raise GitHubError(
                f"Could not connect to GitHub: {exc}"
            ) from exc


        # ----------------------------------------------------
        # RATE LIMIT INFORMATION
        # ----------------------------------------------------

        remaining = response.headers.get(
            "X-RateLimit-Remaining"
        )

        limit = response.headers.get(
            "X-RateLimit-Limit"
        )

        reset = response.headers.get(
            "X-RateLimit-Reset"
        )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.ok:

            try:

                return response.json()

            except ValueError:

                return response.text


        # ----------------------------------------------------
        # ERROR BODY
        # ----------------------------------------------------

        try:

            error_data = response.json()

        except ValueError:

            error_data = {}


        message = error_data.get(
            "message",
            response.text,
        )


        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        if response.status_code == 403:

            if remaining == "0":

                raise GitHubError(
                    "GitHub API rate limit exceeded.\n\n"
                    f"Rate limit: {limit}\n"
                    f"Remaining: {remaining}\n"
                    f"Reset timestamp: {reset}\n\n"
                    "Add a valid GITHUB_TOKEN to "
                    ".streamlit/secrets.toml."
                )

            raise GitHubError(
                "GitHub API returned HTTP 403.\n\n"
                f"GitHub message: {message}\n"
                f"Rate limit: {limit}\n"
                f"Remaining: {remaining}\n\n"
                "This usually means the GitHub token "
                "is invalid, expired, revoked, or "
                "does not have the required access."
            )


        # ----------------------------------------------------
        # NOT FOUND
        # ----------------------------------------------------

        if response.status_code == 404:

            raise GitHubError(
                "GitHub repository not found.\n\n"
                f"GitHub message: {message}\n\n"
                "Check that the repository URL is "
                "correct and that the repository is public."
            )


        # ----------------------------------------------------
        # AUTHENTICATION
        # ----------------------------------------------------

        if response.status_code in {
            401,
            422,
        }:

            raise GitHubError(
                f"GitHub authentication/request error "
                f"(HTTP {response.status_code}).\n\n"
                f"GitHub message: {message}\n\n"
                "Check your GITHUB_TOKEN."
            )


        # ----------------------------------------------------
        # OTHER ERRORS
        # ----------------------------------------------------

        raise GitHubError(
            f"GitHub API error "
            f"(HTTP {response.status_code}).\n\n"
            f"GitHub message: {message}"
        )


    # ========================================================
    # PARSE URL
    # ========================================================

    @staticmethod
    def parse_repo_url(
        url: str,
    ) -> tuple[str, str]:

        url = url.strip()

        pattern = (
            r"https?://"
            r"(?:www\.)?"
            r"github\.com/"
            r"([^/]+)/"
            r"([^/#?]+)"
        )

        match = re.match(
            pattern,
            url,
        )

        if not match:

            raise GitHubError(
                "Invalid GitHub repository URL.\n\n"
                "Expected format:\n"
                "https://github.com/owner/repository"
            )

        owner = match.group(1)

        repo = match.group(2)

        repo = repo.removesuffix(".git")

        return owner, repo


    # ========================================================
    # REPOSITORY
    # ========================================================

    def get_repository(
        self,
        owner: str,
        repo: str,
    ) -> dict:

        return self._request(
            "GET",
            f"/repos/{owner}/{repo}",
        )


    # ========================================================
    # DEFAULT BRANCH
    # ========================================================

    def get_default_branch(
        self,
        owner: str,
        repo: str,
    ) -> str:

        data = self.get_repository(
            owner,
            repo,
        )

        return data.get(
            "default_branch",
            "main",
        )


    # ========================================================
    # TREE
    # ========================================================

    def get_tree(
        self,
        owner: str,
        repo: str,
        branch: str,
    ) -> list[dict]:

        data = self._request(
            "GET",
            f"/repos/{owner}/{repo}/git/trees/{branch}",
            params={
                "recursive": "1"
            },
        )

        if data.get("truncated"):

            print(
                "WARNING: GitHub tree response "
                "was truncated."
            )

        return data.get(
            "tree",
            [],
        )


    # ========================================================
    # FILE
    # ========================================================

    def get_file(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: str,
    ) -> str:

        data = self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{path}",
            params={
                "ref": branch
            },
        )

        if data.get("type") != "file":

            raise GitHubError(
                f"{path} is not a file."
            )

        content = data.get(
            "content",
            "",
        )

        encoding = data.get(
            "encoding",
            "base64",
        )

        if encoding == "base64":

            try:

                decoded = base64.b64decode(
                    content
                )

                return decoded.decode(
                    "utf-8",
                    errors="replace",
                )

            except Exception as exc:

                raise GitHubError(
                    f"Could not decode {path}: {exc}"
                ) from exc


        return str(content)
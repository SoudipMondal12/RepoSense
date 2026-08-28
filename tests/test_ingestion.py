import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.github.client import GitHubClient
from app.ingestion.repository import RepositoryIngestor


def main():

    if len(sys.argv) < 2:
        print(
            "Usage:"
        )
        print(
            "python tests/test_ingestion.py "
            "<github_repo_url>"
        )
        return

    url = sys.argv[1]

    client = GitHubClient()

    owner, repo = client.parse_repo_url(url)

    print("=" * 80)
    print("RepoSense - Code Ingestion Test")
    print("=" * 80)

    print(f"\nRepository: {owner}/{repo}")

    ingestor = RepositoryIngestor(client)

    documents = ingestor.ingest(
        owner,
        repo,
        max_files=100,
    )

    print(
        f"\nTotal Code Documents: "
        f"{len(documents)}"
    )

    for document in documents[:20]:

        print("\n" + "-" * 80)

        print(
            f"File   : {document.file_path}"
        )

        print(
            f"Type   : {document.document_type}"
        )

        print(
            f"Symbol : {document.symbol_name}"
        )

        print(
            f"Lines  : "
            f"{document.start_line}-"
            f"{document.end_line}"
        )


if __name__ == "__main__":
    main()
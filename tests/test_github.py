import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.github.client import (
    GitHubClient,
    GitHubError,
    RepositoryFileFilter,
)


def main():
    url = "https://github.com/psf/requests"

    client = GitHubClient()

    owner, repo = client.parse_repo_url(url)

    print("=" * 70)
    print("RepoSense - GitHub Repository Test")
    print("=" * 70)

    print(f"\nOwner       : {owner}")
    print(f"Repository  : {repo}")

    repository = client.get_repository(
        owner,
        repo,
    )

    print(
        f"\nFull name   : "
        f"{repository['full_name']}"
    )

    print(
        f"Description : "
        f"{repository.get('description')}"
    )

    print(
        f"Language    : "
        f"{repository.get('language')}"
    )

    print(
        f"Stars       : "
        f"{repository['stargazers_count']}"
    )

    print(
        f"Forks       : "
        f"{repository['forks_count']}"
    )

    branch = repository["default_branch"]

    print(
        f"Branch      : {branch}"
    )

    print("\nFetching repository tree...")

    tree = client.get_tree(
        owner,
        repo,
        branch,
    )

    files = [
        item
        for item in tree
        if item.get("type") == "blob"
    ]

    print(
        f"Total files : {len(files)}"
    )

    included_files = [
        item
        for item in files
        if RepositoryFileFilter.should_include(
            item["path"],
            item.get("size", 0),
        )
    ]

    print(
        f"Code files  : "
        f"{len(included_files)}"
    )

    print("\nFirst 20 code files:")
    print("-" * 70)

    for item in included_files[:20]:
        print(
            f"{item['path']:<60}"
            f"{item.get('size', 0):>10} bytes"
        )

    if included_files:

        first_file = included_files[0]

        print("\n" + "=" * 70)
        print(
            f"Reading: {first_file['path']}"
        )
        print("=" * 70)

        content = client.get_file(
            owner,
            repo,
            first_file["path"],
            branch,
        )

        print(
            content[:3000]
        )


if __name__ == "__main__":
    main()
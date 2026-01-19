"""
GitHub Repository Management Agent
A CrewAI agent that can clone repositories, commit/push changes, and create Pull Requests.

This agent provides tools for:
- Cloning GitHub repositories
- Creating and switching branches
- Staging and committing changes
- Pushing to remote
- Creating Pull Requests via GitHub API

Requirements:
- git (command line)
- GitHub Personal Access Token with repo permissions
"""

import os
import subprocess
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
DEFAULT_CLONE_DIR = os.path.join(os.path.dirname(__file__), "repos")

# Preset Repository Configuration
# Set these in your .env file or modify here
PRESET_REPO_URL = os.getenv("GITHUB_REPO_URL", "")  # e.g., https://github.com/owner/repo
PRESET_REPO_OWNER = os.getenv("GITHUB_REPO_OWNER", "")  # e.g., owner
PRESET_REPO_NAME = os.getenv("GITHUB_REPO_NAME", "")  # e.g., repo
PRESET_DEFAULT_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GitResult:
    """Result of a git operation."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class PRInfo:
    """Information about a Pull Request."""
    number: int
    title: str
    url: str
    state: str
    head_branch: str
    base_branch: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_git_command(args: List[str], cwd: Optional[str] = None) -> GitResult:
    """
    Execute a git command and return the result.

    Args:
        args: List of git command arguments (e.g., ['status', '-s'])
        cwd: Working directory for the command

    Returns:
        GitResult with success status and output/error message
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            return GitResult(
                success=True,
                message=result.stdout.strip() or "Command executed successfully"
            )
        else:
            return GitResult(
                success=False,
                message=f"Git error: {result.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        return GitResult(success=False, message="Git command timed out")
    except FileNotFoundError:
        return GitResult(success=False, message="Git is not installed or not in PATH")
    except Exception as e:
        return GitResult(success=False, message=f"Error executing git: {str(e)}")


def get_repo_path(repo_name: str) -> str:
    """Get the local path for a cloned repository."""
    return os.path.join(DEFAULT_CLONE_DIR, repo_name.split("/")[-1])


def ensure_clone_dir():
    """Ensure the clone directory exists."""
    os.makedirs(DEFAULT_CLONE_DIR, exist_ok=True)


# ============================================================================
# GIT TOOLS
# ============================================================================

@tool
def clone_repository(repo_url: str, branch: Optional[str] = None) -> str:
    """
    Clone a GitHub repository to local storage.

    Args:
        repo_url: Full GitHub repository URL (e.g., https://github.com/owner/repo)
        branch: Optional branch to checkout after cloning

    Returns:
        Status message with local path or error
    """
    ensure_clone_dir()

    # Extract repo name from URL
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = get_repo_path(repo_name)

    # Check if already cloned
    if os.path.exists(local_path):
        # Pull latest changes instead
        result = run_git_command(["pull"], cwd=local_path)
        if result.success:
            return f"Repository already exists at {local_path}. Pulled latest changes."
        return f"Repository exists at {local_path} but failed to pull: {result.message}"

    # Clone with token for authentication if available
    if GITHUB_TOKEN and "github.com" in repo_url:
        # Insert token into URL for authentication
        auth_url = repo_url.replace("https://", f"https://{GITHUB_TOKEN}@")
        args = ["clone", auth_url, local_path]
    else:
        args = ["clone", repo_url, local_path]

    result = run_git_command(args)

    if not result.success:
        return f"Failed to clone repository: {result.message}"

    # Checkout specific branch if requested
    if branch:
        branch_result = run_git_command(["checkout", branch], cwd=local_path)
        if not branch_result.success:
            return f"Cloned to {local_path} but failed to checkout branch '{branch}': {branch_result.message}"

    return f"Successfully cloned repository to {local_path}"


@tool
def create_branch(repo_name: str, branch_name: str, from_branch: str = "main") -> str:
    """
    Create a new branch in the repository.

    Args:
        repo_name: Name of the repository (local folder name)
        branch_name: Name for the new branch
        from_branch: Base branch to create from (default: main)

    Returns:
        Status message
    """
    local_path = get_repo_path(repo_name)

    if not os.path.exists(local_path):
        return f"Repository not found at {local_path}. Clone it first."

    # Fetch latest
    run_git_command(["fetch", "origin"], cwd=local_path)

    # Checkout base branch and pull
    checkout_result = run_git_command(["checkout", from_branch], cwd=local_path)
    if not checkout_result.success:
        return f"Failed to checkout base branch '{from_branch}': {checkout_result.message}"

    run_git_command(["pull", "origin", from_branch], cwd=local_path)

    # Create and switch to new branch
    result = run_git_command(["checkout", "-b", branch_name], cwd=local_path)

    if result.success:
        return f"Created and switched to new branch '{branch_name}'"
    return f"Failed to create branch: {result.message}"


@tool
def get_repo_status(repo_name: str) -> str:
    """
    Get the current status of a repository (branch, changes, etc.).

    Args:
        repo_name: Name of the repository (local folder name)

    Returns:
        Repository status information
    """
    local_path = get_repo_path(repo_name)

    if not os.path.exists(local_path):
        return f"Repository not found at {local_path}"

    # Get current branch
    branch_result = run_git_command(["branch", "--show-current"], cwd=local_path)
    current_branch = branch_result.message if branch_result.success else "unknown"

    # Get status
    status_result = run_git_command(["status", "--porcelain"], cwd=local_path)

    # Get recent commits
    log_result = run_git_command(
        ["log", "--oneline", "-5"],
        cwd=local_path
    )

    status_info = {
        "path": local_path,
        "current_branch": current_branch,
        "changes": status_result.message if status_result.success else "Error getting status",
        "recent_commits": log_result.message if log_result.success else "Error getting log"
    }

    return json.dumps(status_info, indent=2)


@tool
def stage_files(repo_name: str, file_paths: str = ".") -> str:
    """
    Stage files for commit.

    Args:
        repo_name: Name of the repository (local folder name)
        file_paths: Space-separated list of file paths to stage, or "." for all changes

    Returns:
        Status message
    """
    local_path = get_repo_path(repo_name)

    if not os.path.exists(local_path):
        return f"Repository not found at {local_path}"

    # Handle multiple files
    files = file_paths.split() if file_paths != "." else ["."]

    result = run_git_command(["add"] + files, cwd=local_path)

    if result.success:
        return f"Staged files: {file_paths}"
    return f"Failed to stage files: {result.message}"


@tool
def commit_changes(repo_name: str, commit_message: str) -> str:
    """
    Commit staged changes with a message.

    Args:
        repo_name: Name of the repository (local folder name)
        commit_message: Commit message describing the changes

    Returns:
        Status message with commit hash if successful
    """
    local_path = get_repo_path(repo_name)

    if not os.path.exists(local_path):
        return f"Repository not found at {local_path}"

    # Check if there are staged changes
    status_result = run_git_command(["diff", "--cached", "--stat"], cwd=local_path)
    if status_result.success and not status_result.message:
        return "No staged changes to commit. Use stage_files first."

    result = run_git_command(["commit", "-m", commit_message], cwd=local_path)

    if result.success:
        # Get the commit hash
        hash_result = run_git_command(["rev-parse", "--short", "HEAD"], cwd=local_path)
        commit_hash = hash_result.message if hash_result.success else "unknown"
        return f"Committed successfully. Hash: {commit_hash}"
    return f"Failed to commit: {result.message}"


@tool
def push_changes(repo_name: str, branch: Optional[str] = None, set_upstream: bool = True) -> str:
    """
    Push committed changes to the remote repository.

    Args:
        repo_name: Name of the repository (local folder name)
        branch: Branch to push (defaults to current branch)
        set_upstream: Whether to set upstream tracking (default True)

    Returns:
        Status message
    """
    local_path = get_repo_path(repo_name)

    if not os.path.exists(local_path):
        return f"Repository not found at {local_path}"

    # Get current branch if not specified
    if not branch:
        branch_result = run_git_command(["branch", "--show-current"], cwd=local_path)
        if branch_result.success:
            branch = branch_result.message
        else:
            return f"Failed to determine current branch: {branch_result.message}"

    # Build push command
    args = ["push"]
    if set_upstream:
        args.extend(["-u", "origin", branch])
    else:
        args.extend(["origin", branch])

    result = run_git_command(args, cwd=local_path)

    if result.success:
        return f"Successfully pushed '{branch}' to origin"
    return f"Failed to push: {result.message}"


@tool
def create_pull_request(
    repo_name: str,
    owner: str,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str = "main"
) -> str:
    """
    Create a Pull Request on GitHub using the GitHub API.

    Args:
        repo_name: Name of the repository
        owner: GitHub username or organization that owns the repo
        title: Title for the Pull Request
        body: Description/body of the Pull Request
        head_branch: Branch containing the changes
        base_branch: Branch to merge into (default: main)

    Returns:
        PR URL and number if successful, error message otherwise
    """
    if not GITHUB_TOKEN:
        return "GitHub token not configured. Set GITHUB_TOKEN environment variable."

    import urllib.request
    import urllib.error

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"

    payload = {
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch
    }

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(api_url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            return json.dumps({
                "success": True,
                "pr_number": result["number"],
                "pr_url": result["html_url"],
                "title": result["title"],
                "state": result["state"]
            }, indent=2)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_json = json.loads(error_body)
            error_msg = error_json.get("message", str(e))
            errors = error_json.get("errors", [])
            if errors:
                error_msg += f" - {errors[0].get('message', '')}"
        except:
            error_msg = str(e)
        return f"Failed to create PR: {error_msg}"
    except Exception as e:
        return f"Error creating PR: {str(e)}"


@tool
def list_pull_requests(owner: str, repo_name: str, state: str = "open") -> str:
    """
    List Pull Requests for a repository.

    Args:
        owner: GitHub username or organization that owns the repo
        repo_name: Name of the repository
        state: Filter by state - 'open', 'closed', or 'all'

    Returns:
        JSON list of PRs
    """
    if not GITHUB_TOKEN:
        return "GitHub token not configured. Set GITHUB_TOKEN environment variable."

    import urllib.request
    import urllib.error

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls?state={state}"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        request = urllib.request.Request(api_url, headers=headers)

        with urllib.request.urlopen(request, timeout=30) as response:
            prs = json.loads(response.read().decode("utf-8"))

            pr_list = []
            for pr in prs:
                pr_list.append({
                    "number": pr["number"],
                    "title": pr["title"],
                    "state": pr["state"],
                    "url": pr["html_url"],
                    "author": pr["user"]["login"],
                    "head_branch": pr["head"]["ref"],
                    "base_branch": pr["base"]["ref"],
                    "created_at": pr["created_at"]
                })

            return json.dumps(pr_list, indent=2)

    except urllib.error.HTTPError as e:
        return f"Failed to list PRs: {str(e)}"
    except Exception as e:
        return f"Error listing PRs: {str(e)}"


@tool
def write_file_to_repo(repo_name: str, file_path: str, content: str) -> str:
    """
    Write or update a file in the repository.

    Args:
        repo_name: Name of the repository (local folder name)
        file_path: Path to the file relative to repo root
        content: Content to write to the file

    Returns:
        Status message
    """
    local_path = get_repo_path(repo_name)

    if not os.path.exists(local_path):
        return f"Repository not found at {local_path}. Clone it first."

    full_path = os.path.join(local_path, file_path)

    # Create parent directories if needed
    parent_dir = os.path.dirname(full_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Failed to write file: {str(e)}"


@tool
def read_file_from_repo(repo_name: str, file_path: str) -> str:
    """
    Read a file from the repository.

    Args:
        repo_name: Name of the repository (local folder name)
        file_path: Path to the file relative to repo root

    Returns:
        File content or error message
    """
    local_path = get_repo_path(repo_name)

    if not os.path.exists(local_path):
        return f"Repository not found at {local_path}"

    full_path = os.path.join(local_path, file_path)

    if not os.path.exists(full_path):
        return f"File not found: {file_path}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {str(e)}"


# ============================================================================
# CREWAI AGENT SETUP
# ============================================================================

def create_github_agent(repo_url: str = "", repo_owner: str = "", repo_name: str = "", default_branch: str = "main") -> Agent:
    """Create the GitHub Repository Management agent with preset repository context."""

    # Use preset values if not provided
    repo_url = repo_url or PRESET_REPO_URL
    repo_owner = repo_owner or PRESET_REPO_OWNER
    repo_name = repo_name or PRESET_REPO_NAME
    default_branch = default_branch or PRESET_DEFAULT_BRANCH

    backstory = f"""You are a GitHub automation expert who manages repository workflows.
        You can clone repositories, create branches, make file changes, commit and push code,
        and create pull requests. You follow git best practices including meaningful commit
        messages and proper branching strategies. You always verify the state of the repository
        before making changes and handle errors gracefully.

        IMPORTANT - You are configured to work with a preset repository:
        - Repository URL: {repo_url}
        - Owner: {repo_owner}
        - Repository Name: {repo_name}
        - Default Branch: {default_branch}

        When the user asks you to perform actions, use this preset repository unless they
        explicitly specify a different one. Always clone/sync the repository first before
        making any changes."""

    return Agent(
        role="GitHub Repository Manager",
        goal=f"Manage the GitHub repository {repo_owner}/{repo_name} by making changes, committing, pushing, and creating pull requests based on user requests",
        backstory=backstory,
        tools=[
            clone_repository,
            create_branch,
            get_repo_status,
            stage_files,
            commit_changes,
            push_changes,
            create_pull_request,
            list_pull_requests,
            write_file_to_repo,
            read_file_from_repo
        ],
        verbose=True,
        allow_delegation=False
    )


def create_github_crew(
    task_description: str,
    repo_url: str = "",
    repo_owner: str = "",
    repo_name: str = "",
    default_branch: str = "main"
) -> Crew:
    """
    Create a crew for executing GitHub operations.

    Args:
        task_description: Description of what needs to be done
        repo_url: GitHub repository URL (uses preset if not provided)
        repo_owner: Repository owner (uses preset if not provided)
        repo_name: Repository name (uses preset if not provided)
        default_branch: Default branch name

    Returns:
        Configured Crew instance
    """
    # Use preset values
    repo_url = repo_url or PRESET_REPO_URL
    repo_owner = repo_owner or PRESET_REPO_OWNER
    repo_name = repo_name or PRESET_REPO_NAME

    agent = create_github_agent(repo_url, repo_owner, repo_name, default_branch)

    # Enhance task description with repository context
    enhanced_task = f"""
    Repository Context:
    - URL: {repo_url}
    - Owner: {repo_owner}
    - Name: {repo_name}
    - Default Branch: {default_branch}

    User Request:
    {task_description}

    Instructions:
    1. First, clone or sync the repository using clone_repository with URL: {repo_url}
    2. Then perform the requested operations
    3. If creating/modifying files, stage and commit the changes
    4. Push changes to the remote repository
    5. Create a PR if the user requests one or if changes are on a feature branch
    """

    task = Task(
        description=enhanced_task,
        expected_output="A detailed report of all git operations performed, including any errors encountered and the final state of the repository",
        agent=agent
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )


# ============================================================================
# HIGH-LEVEL OPERATIONS
# ============================================================================

class GitHubManager:
    """
    High-level manager for GitHub operations.
    Provides simple methods for common workflows with a preset repository.
    """

    def __init__(
        self,
        repo_url: str = "",
        repo_owner: str = "",
        repo_name: str = "",
        default_branch: str = "main"
    ):
        """
        Initialize the GitHub Manager.

        Args:
            repo_url: GitHub repository URL (uses GITHUB_REPO_URL env var if not provided)
            repo_owner: Repository owner (uses GITHUB_REPO_OWNER env var if not provided)
            repo_name: Repository name (uses GITHUB_REPO_NAME env var if not provided)
            default_branch: Default branch name (uses GITHUB_DEFAULT_BRANCH env var if not provided)
        """
        self.repo_url = repo_url or PRESET_REPO_URL
        self.repo_owner = repo_owner or PRESET_REPO_OWNER
        self.repo_name = repo_name or PRESET_REPO_NAME
        self.default_branch = default_branch or PRESET_DEFAULT_BRANCH
        self.crew = None
        self._initialized = False

    def _ensure_repo_cloned(self) -> str:
        """Ensure the repository is cloned locally."""
        if not self.repo_url:
            return "Error: No repository URL configured. Set GITHUB_REPO_URL in .env"

        result = clone_repository.run(repo_url=self.repo_url)
        self._initialized = True
        return result

    def execute_request(self, user_request: str) -> str:
        """
        Execute a natural language request against the preset repository.

        Args:
            user_request: Natural language description of what to do
                         Examples:
                         - "Create a new file test.py with a hello world function"
                         - "Update README.md to add installation instructions"
                         - "Create a branch feature/new-api and add api.py"

        Returns:
            Result of the operation
        """
        if not self.repo_url:
            return "Error: No repository configured. Set GITHUB_REPO_URL, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in your .env file."

        crew = create_github_crew(
            task_description=user_request,
            repo_url=self.repo_url,
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            default_branch=self.default_branch
        )

        result = crew.kickoff()
        return str(result)

    def make_changes_and_pr(
        self,
        repo_url: str,
        owner: str,
        repo_name: str,
        branch_name: str,
        changes: Dict[str, str],
        commit_message: str,
        pr_title: str,
        pr_body: str,
        base_branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Complete workflow: clone repo, create branch, make changes, commit, push, and create PR.

        Args:
            repo_url: Full GitHub repository URL
            owner: GitHub owner/organization
            repo_name: Repository name
            branch_name: Name for the new branch
            changes: Dict mapping file paths to their new content
            commit_message: Message for the commit
            pr_title: Title for the Pull Request
            pr_body: Body/description for the Pull Request
            base_branch: Branch to merge into

        Returns:
            Dict with operation results
        """
        results = {
            "clone": None,
            "branch": None,
            "changes": [],
            "commit": None,
            "push": None,
            "pr": None
        }

        # Clone repository
        clone_result = clone_repository.run(repo_url=repo_url)
        results["clone"] = clone_result
        if "Failed" in clone_result and "already exists" not in clone_result:
            return results

        # Create branch
        branch_result = create_branch.run(
            repo_name=repo_name,
            branch_name=branch_name,
            from_branch=base_branch
        )
        results["branch"] = branch_result
        if "Failed" in branch_result:
            return results

        # Make changes
        for file_path, content in changes.items():
            write_result = write_file_to_repo.run(
                repo_name=repo_name,
                file_path=file_path,
                content=content
            )
            results["changes"].append({"file": file_path, "result": write_result})

        # Stage all changes
        stage_files.run(repo_name=repo_name, file_paths=".")

        # Commit
        commit_result = commit_changes.run(
            repo_name=repo_name,
            commit_message=commit_message
        )
        results["commit"] = commit_result
        if "Failed" in commit_result:
            return results

        # Push
        push_result = push_changes.run(
            repo_name=repo_name,
            branch=branch_name,
            set_upstream=True
        )
        results["push"] = push_result
        if "Failed" in push_result:
            return results

        # Create PR
        pr_result = create_pull_request.run(
            repo_name=repo_name,
            owner=owner,
            title=pr_title,
            body=pr_body,
            head_branch=branch_name,
            base_branch=base_branch
        )
        results["pr"] = pr_result

        return results

    def run_with_crew(self, task_description: str) -> str:
        """
        Run a GitHub task using the CrewAI agent.

        Args:
            task_description: Natural language description of what to do

        Returns:
            Result from the crew execution
        """
        crew = create_github_crew(
            task_description=task_description,
            repo_url=self.repo_url,
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            default_branch=self.default_branch
        )
        result = crew.kickoff()
        return str(result)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("  GitHub Repository Management Agent")
    print("  Powered by CrewAI")
    print("=" * 60)


def print_config_status():
    """Print the current configuration status."""
    print("\n[Configuration Status]")

    if GITHUB_TOKEN:
        print(f"  GitHub Token: Configured (***{GITHUB_TOKEN[-4:]})")
    else:
        print("  GitHub Token: NOT SET (PR creation will fail)")

    if PRESET_REPO_URL:
        print(f"  Repository URL: {PRESET_REPO_URL}")
        print(f"  Owner: {PRESET_REPO_OWNER}")
        print(f"  Repo Name: {PRESET_REPO_NAME}")
        print(f"  Default Branch: {PRESET_DEFAULT_BRANCH}")
    else:
        print("  Repository: NOT CONFIGURED")
        print("\n  To configure, add to your .env file:")
        print("    GITHUB_REPO_URL=https://github.com/owner/repo")
        print("    GITHUB_REPO_OWNER=owner")
        print("    GITHUB_REPO_NAME=repo")


def interactive_mode():
    """Run the agent in interactive mode."""
    print_banner()
    print_config_status()

    if not PRESET_REPO_URL:
        print("\nPlease configure the repository in .env file first.")
        return

    manager = GitHubManager()

    print("\n[Interactive Mode]")
    print("Type your requests in natural language. Examples:")
    print('  - "Create a new file test.py with print hello world"')
    print('  - "Add a README.md with project description"')
    print('  - "Create branch feature/api and add api.py with a Flask app"')
    print('  - "List all open pull requests"')
    print('  - "Show repository status"')
    print("\nType 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("\nYour request > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'status':
                # Quick status check
                result = get_repo_status.run(repo_name=PRESET_REPO_NAME)
                print(f"\n{result}")
                continue

            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  status  - Show repository status")
                print("  help    - Show this help")
                print("  quit    - Exit the program")
                print("\nOr type any natural language request like:")
                print('  "Create a file called utils.py with helper functions"')
                print('  "Push changes to main branch"')
                print('  "Create a PR for the current branch"')
                continue

            print(f"\nProcessing: {user_input}")
            print("-" * 40)

            result = manager.execute_request(user_input)

            print("\n" + "=" * 40)
            print("RESULT:")
            print("=" * 40)
            print(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def run_single_request(request: str) -> str:
    """
    Run a single request and return the result.

    Args:
        request: Natural language request

    Returns:
        Result string
    """
    if not PRESET_REPO_URL:
        return "Error: Repository not configured. Set GITHUB_REPO_URL in .env"

    manager = GitHubManager()
    return manager.execute_request(request)


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        # Command line argument mode
        request = " ".join(sys.argv[1:])
        print_banner()
        print(f"\nExecuting: {request}")
        result = run_single_request(request)
        print(f"\nResult:\n{result}")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()

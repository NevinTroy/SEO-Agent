"""
SEO Remediation Agent
An agent that processes SEO audit results and uses the GitHub agent to implement fixes.

This agent:
1. Parses SEO audit JSON from the seo_orchestrator
2. Analyzes each remediation task
3. Generates appropriate code changes
4. Uses the GitHub agent to commit and push changes

Supported remediation actions:
- UPDATE_CANONICAL: Add/update canonical tags
- UPDATE_TAG: Update title/meta description tags
- RESTRUCTURE_HEADINGS: Fix heading hierarchy
- REVIEW_LINKS: Fix malformed links
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Import GitHub agent tools
from github_agent import (
    GitHubManager,
    clone_repository,
    create_branch,
    get_repo_status,
    stage_files,
    commit_changes,
    push_changes,
    create_pull_request,
    write_file_to_repo,
    read_file_from_repo,
    PRESET_REPO_URL,
    PRESET_REPO_OWNER,
    PRESET_REPO_NAME,
    PRESET_DEFAULT_BRANCH
)


# ============================================================================
# DATA CLASSES
# ============================================================================

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ActionType(Enum):
    UPDATE_CANONICAL = "UPDATE_CANONICAL"
    UPDATE_TAG = "UPDATE_TAG"
    RESTRUCTURE_HEADINGS = "RESTRUCTURE_HEADINGS"
    REVIEW_LINKS = "REVIEW_LINKS"


@dataclass
class RemediationTask:
    """A single SEO remediation task."""
    category: str
    issue: str
    severity: str
    reasoning: str
    action: str
    target: str
    page_url: str
    current_value: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class PageFix:
    """Accumulated fixes for a single page."""
    page_url: str
    page_path: str  # Path in the codebase
    fixes: List[RemediationTask] = field(default_factory=list)
    canonical_url: Optional[str] = None
    new_title: Optional[str] = None
    new_meta_description: Optional[str] = None
    heading_changes: Optional[Dict] = None


# ============================================================================
# AUDIT PARSER
# ============================================================================

class AuditParser:
    """Parses SEO audit JSON and extracts remediation tasks."""

    def __init__(self, audit_data: Dict[str, Any]):
        self.audit_data = audit_data
        self.tasks: List[RemediationTask] = []
        self.pages: Dict[str, PageFix] = {}

    def parse(self) -> List[RemediationTask]:
        """Parse the audit data and extract all tasks."""
        # Get prioritized tasks from audit (top-level)
        prioritized = self.audit_data.get("prioritized_tasks", [])

        # Also get tasks from individual page audits (top-level)
        page_audits = self.audit_data.get("page_audits", [])

        # Process prioritized tasks first
        for task_data in prioritized:
            task = self._parse_task(task_data)
            if task:
                self.tasks.append(task)
                self._add_to_page(task)

        # Process tasks inside each page audit
        for page in page_audits:
            page_url = page.get("analyzed_url", "")
            for task_data in page.get("tasks", []):
                # Ensure page_url present
                task_data.setdefault("page_url", page_url)
                task = self._parse_task(task_data)
                if task:
                    self.tasks.append(task)
                    self._add_to_page(task)

        return self.tasks

    def _parse_task(self, task_data: Dict) -> Optional[RemediationTask]:
        """Parse a single task from the audit data."""
        try:
            remediation = task_data.get("remediation", {})

            return RemediationTask(
                category=task_data.get("category", ""),
                issue=task_data.get("issue", ""),
                severity=task_data.get("severity", "LOW"),
                reasoning=task_data.get("reasoning", ""),
                action=remediation.get("action", ""),
                target=remediation.get("target", ""),
                page_url=task_data.get("page_url", ""),
                current_value=remediation.get("current_value"),
                details=remediation.get("current_structure") or remediation.get("examples")
            )
        except Exception as e:
            print(f"Error parsing task: {e}")
            return None

    def _add_to_page(self, task: RemediationTask):
        """Add a task to its page's fix list."""
        if task.page_url not in self.pages:
            page_path = self._url_to_path(task.page_url)
            self.pages[task.page_url] = PageFix(
                page_url=task.page_url,
                page_path=page_path
            )

        self.pages[task.page_url].fixes.append(task)

        # Extract specific fix types
        if task.action == "UPDATE_CANONICAL":
            self.pages[task.page_url].canonical_url = task.page_url

        elif task.action == "UPDATE_TAG" and task.target == "title":
            # Will need to generate a better title
            self.pages[task.page_url].new_title = None  # To be generated

        elif task.action == "UPDATE_TAG" and "description" in task.target:
            self.pages[task.page_url].new_meta_description = None  # To be generated

        elif task.action == "RESTRUCTURE_HEADINGS":
            self.pages[task.page_url].heading_changes = task.details

    def _url_to_path(self, url: str) -> str:
        """Convert a URL to a likely file path in a Next.js project."""
        # Extract path from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return "src/app/page.tsx"  # Root page

        # Common Next.js patterns with src directory
        # Could be: src/app/[route]/page.tsx
        return f"src/app/{path}/page.tsx"

    def get_tasks_by_severity(self, severity: str) -> List[RemediationTask]:
        """Get all tasks of a specific severity."""
        return [t for t in self.tasks if t.severity == severity]

    def get_tasks_by_action(self, action: str) -> List[RemediationTask]:
        """Get all tasks of a specific action type."""
        return [t for t in self.tasks if t.action == action]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tasks."""
        return {
            "total_tasks": len(self.tasks),
            "by_severity": {
                "CRITICAL": len(self.get_tasks_by_severity("CRITICAL")),
                "HIGH": len(self.get_tasks_by_severity("HIGH")),
                "MEDIUM": len(self.get_tasks_by_severity("MEDIUM")),
                "LOW": len(self.get_tasks_by_severity("LOW"))
            },
            "by_action": {
                "UPDATE_CANONICAL": len(self.get_tasks_by_action("UPDATE_CANONICAL")),
                "UPDATE_TAG": len(self.get_tasks_by_action("UPDATE_TAG")),
                "RESTRUCTURE_HEADINGS": len(self.get_tasks_by_action("RESTRUCTURE_HEADINGS")),
                "REVIEW_LINKS": len(self.get_tasks_by_action("REVIEW_LINKS"))
            },
            "pages_affected": len(self.pages)
        }


def _fallback_title(page_path: str, h1_content: str) -> str:
    """Generate an SEO-friendly title based on page path and H1."""
    path_titles = {
        "": "Award-Winning Landscaping Company Toronto & GTA | Avanti Landscaping",
        "get-a-quote": "Get a Free Landscaping Quote | Avanti Landscaping Toronto",
        "services/landscaping": "Professional Landscaping Services Toronto | Avanti Landscaping",
        "services/stonework": "Expert Stonework Services Toronto & GTA | Avanti Landscaping",
        "services/woodwork": "Custom Woodwork & Decking Services | Avanti Landscaping",
        "services/swimming-pools": "Swimming Pool Installation Toronto | Avanti Landscaping",
        "services/other-services": "Additional Outdoor Services | Avanti Landscaping",
        "gallery": "Landscaping Project Gallery | Avanti Landscaping Toronto",
        "about-us": "About Avanti Landscaping | Award-Winning GTA Landscapers",
        "blog": "Landscaping Tips & Insights | Avanti Landscaping Blog",
        "contacts": "Contact Avanti Landscaping | Toronto & GTA Service",
        "privacy-policy": "Privacy Policy | Avanti Landscaping",
        "sitemap": "Site Map | Avanti Landscaping"
    }
    return path_titles.get(page_path, f"{h1_content} | Avanti Landscaping" if h1_content else "Avanti Landscaping")


def _fallback_description(page_path: str, h1_content: str) -> str:
    """Generate an SEO-friendly meta description with call-to-action."""
    path_descriptions = {
        "": "Transform your outdoor space with Toronto's award-winning landscaping company. Offering landscaping, stonework, pools & more. Get your free quote today!",
        "get-a-quote": "Request your free landscaping quote from Avanti Landscaping. Professional outdoor transformations in Toronto & GTA. Contact us today!",
        "services/landscaping": "Professional landscaping services in Toronto & GTA. From design to installation, we create stunning outdoor spaces. Get your free estimate!",
        "services/stonework": "Expert stonework services including patios, walkways & retaining walls in Toronto. Quality craftsmanship guaranteed. Request a quote!",
        "services/woodwork": "Custom decks, pergolas & woodwork in Toronto & GTA. Premium materials & expert installation. Get your free consultation today!",
        "services/swimming-pools": "Swimming pool installation & design in Toronto. Inground pools, pool landscaping & more. Start your pool project today!",
        "services/other-services": "Comprehensive outdoor services including lighting, irrigation & maintenance in Toronto. Contact Avanti Landscaping for details!",
        "gallery": "Browse our portfolio of stunning landscaping transformations in Toronto & GTA. See our work and get inspired for your project!",
        "about-us": "Learn about Avanti Landscaping - Toronto's trusted landscaping experts since 2010. Award-winning service & certified professionals.",
        "blog": "Expert landscaping tips, outdoor design ideas & industry insights from Avanti Landscaping. Stay informed about your outdoor space!",
        "contacts": "Contact Avanti Landscaping for your Toronto & GTA outdoor project. Call (647) 870-8337 or request a free quote online!",
        "privacy-policy": "Avanti Landscaping privacy policy. Learn how we protect your personal information and data.",
        "sitemap": "Navigate Avanti Landscaping website. Find landscaping services, gallery, contact information and more."
    }

    if page_path in path_descriptions:
        return path_descriptions[page_path]
    elif h1_content:
        return f"{h1_content}. Professional landscaping services in Toronto & GTA. Contact Avanti Landscaping today!"
    else:
        return "Professional landscaping services in Toronto & GTA. Contact Avanti Landscaping today!"


def _fallback_metadata(page: PageFix, base_domain: str) -> str:
    """Generate a metadata.ts file content from page info."""
    page_path = page.page_url.replace(f"https://{base_domain}", "").strip("/")
    if not page_path:
        page_path = ""

    h1_content = ""
    if page.heading_changes and "h1" in page.heading_changes:
        h1_list = page.heading_changes.get("h1", [])
        if h1_list:
            h1_content = h1_list[0]

    title = _fallback_title(page_path, h1_content)
    description = _fallback_description(page_path, h1_content)

    return f'''import type {{ Metadata }} from 'next'

export const metadata: Metadata = {{
  title: '{title}',
  description: '{description}',
  alternates: {{
    canonical: '{page.page_url}',
  }},
}}
'''


# ============================================================================
# CREWAI TOOLS
# ============================================================================

@tool
def load_audit_file(file_path: str) -> str:
    """
    Load and parse an SEO audit JSON file.

    Args:
        file_path: Path to the audit JSON file

    Returns:
        Summary of the audit tasks
    """
    try:
        with open(file_path, 'r') as f:
            audit_data = json.load(f)

        parser = AuditParser(audit_data)
        parser.parse()
        summary = parser.get_summary()

        return json.dumps({
            "status": "success",
            "file": file_path,
            "summary": summary,
            "base_domain": audit_data.get("crawl_summary", {}).get("base_domain", "")
        }, indent=2)
    except Exception as e:
        return f"Error loading audit file: {str(e)}"


@tool
def get_tasks_for_page(file_path: str, page_url: str) -> str:
    """
    Get all remediation tasks for a specific page.

    Args:
        file_path: Path to the audit JSON file
        page_url: URL of the page to get tasks for

    Returns:
        List of tasks for the page
    """
    try:
        with open(file_path, 'r') as f:
            audit_data = json.load(f)

        parser = AuditParser(audit_data)
        parser.parse()

        if page_url in parser.pages:
            page = parser.pages[page_url]
            return json.dumps({
                "page_url": page_url,
                "page_path": page.page_path,
                "tasks": [
                    {
                        "action": t.action,
                        "issue": t.issue,
                        "severity": t.severity,
                        "target": t.target,
                        "current_value": t.current_value
                    }
                    for t in page.fixes
                ]
            }, indent=2)
        else:
            return f"No tasks found for page: {page_url}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def generate_page_changes(file_path: str, page_url: str, action: str = "") -> str:
    """
    Generate actionable code-change suggestions for a page based on SEO audit tasks.

    Args:
        file_path: Path to the audit JSON file
        page_url: URL of the page to generate changes for
        action: Optional filter (e.g., UPDATE_CANONICAL, UPDATE_TAG, RESTRUCTURE_HEADINGS, REVIEW_LINKS)

    Returns:
        JSON with suggested changes per task (issue -> file path -> proposed change)
    """
    try:
        with open(file_path, 'r') as f:
            audit_data = json.load(f)

        parser = AuditParser(audit_data)
        parser.parse()

        base_domain = (
            audit_data.get("crawl_summary", {}).get("base_domain")
            or audit_data.get("base_domain", "")
        )

        if page_url not in parser.pages:
            return f"No page found: {page_url}"

        page = parser.pages[page_url]
        suggestions = []

        for task in page.fixes:
            if action and task.action != action:
                continue

            suggestion: Dict[str, Any] = {
                "issue": task.issue,
                "action": task.action,
                "target": task.target,
                "severity": task.severity,
                "page_url": page_url,
                "suggested_path": page.page_path,
            }

            if task.action == "UPDATE_CANONICAL":
                suggestion["proposed_change"] = {
                    "type": "canonical",
                    "tag": f'<link rel="canonical" href="{page.page_url}" />',
                    "suggested_path": page.page_path.replace("page.tsx", "metadata.ts"),
                }

            elif task.action == "UPDATE_TAG":
                if task.target == "title":
                    suggestion["proposed_change"] = {
                        "type": "title",
                        "note": "Rewrite title to align with H1/intent (<60 chars).",
                        "current_value": task.current_value,
                    }
                elif "description" in task.target:
                    desc = _fallback_description(
                        page.page_url.replace(f"https://{base_domain}", "").strip("/"),
                        "",
                    )
                    suggestion["proposed_change"] = {
                        "type": "meta_description",
                        "proposed_value": desc,
                        "current_value": task.current_value,
                    }

            elif task.action == "RESTRUCTURE_HEADINGS":
                suggestion["proposed_change"] = {
                    "type": "headings",
                    "current_structure": task.details,
                    "note": "Insert appropriate H2s or re-tier H3s under H2s.",
                }

            elif task.action == "REVIEW_LINKS":
                examples = task.details or []
                normalized = []
                for link in examples:
                    if link.startswith("tel:"):
                        digits = re.sub(r"[^\d+]", "", link.replace("tel:", ""))
                        if not digits.startswith("+"):
                            digits = "+1" + digits
                        normalized.append(f"tel:{digits}")
                    elif link.startswith("mailto:"):
                        normalized.append(link.lower())
                    else:
                        normalized.append(link)
                suggestion["proposed_change"] = {
                    "type": "links",
                    "examples": examples,
                    "normalized_examples": normalized,
                }

            suggestions.append(suggestion)

        return json.dumps({
            "page_url": page_url,
            "suggestions": suggestions
        }, indent=2)

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def list_all_pages(file_path: str) -> str:
    """
    List all pages in the audit that need fixes.

    Args:
        file_path: Path to the audit JSON file

    Returns:
        List of all pages with their fix counts
    """
    try:
        with open(file_path, 'r') as f:
            audit_data = json.load(f)

        parser = AuditParser(audit_data)
        parser.parse()

        pages_list = []
        for url, page in parser.pages.items():
            pages_list.append({
                "url": url,
                "path": page.page_path,
                "fix_count": len(page.fixes),
                "actions": list(set(t.action for t in page.fixes))
            })

        return json.dumps({
            "total_pages": len(pages_list),
            "pages": sorted(pages_list, key=lambda x: x["fix_count"], reverse=True)
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# SEO REMEDIATION AGENT
# ============================================================================

def create_seo_remediation_agent() -> Agent:
    """Create the SEO Remediation agent."""
    return Agent(
        role="SEO Remediation Specialist",
        goal="Process SEO audit results and implement fixes by modifying code in the repository",
        backstory="""You are an expert SEO developer who specializes in implementing technical SEO fixes.
        You understand Next.js, React, and modern web development practices. You can read SEO audit
        reports and translate them into concrete code changes. You work systematically through each
        issue, generating appropriate fixes and committing them to the repository. You apply fixes
        automatically (no HITL prompts) and log every code change (issue -> file path -> action).

        CRITICAL - Before making ANY changes:
        1. ALWAYS use read_file_from_repo to examine the existing codebase structure
        2. Check if pages are in 'src/app/' or 'app/' directory by reading existing files
        3. Look at existing page.tsx and layout.tsx files to understand current patterns
        4. Write new files to the SAME directory structure as existing files

        Your workflow:
        1. Clone/sync the repository
        2. READ existing files to understand the project structure (src/app vs app)
        3. Load and analyze the SEO audit file
        4. List all pages that need fixes
        5. For each task from SEO_Analyst, generate and APPLY the code change in the CORRECT directory
        6. Log every change: issue -> file path -> action taken (append to a change log)
        7. Stage, commit, and push; optionally create a PR

        You prioritize fixes by severity (CRITICAL > HIGH > MEDIUM > LOW) and group related
        changes together for cleaner commits.

        For Next.js App Router: Metadata files (metadata.ts) should be placed alongside
        their corresponding page.tsx files in the same directory.""",
        tools=[
            # SEO Tools
            load_audit_file,
            get_tasks_for_page,
            generate_page_changes,
            list_all_pages,
            # GitHub Tools
            clone_repository,
            create_branch,
            get_repo_status,
            stage_files,
            commit_changes,
            push_changes,
            create_pull_request,
            write_file_to_repo,
            read_file_from_repo
        ],
        verbose=True,
        allow_delegation=False
    )


def create_code_generator_agent() -> Agent:
    """
    Create the code generator agent that applies fixes directly to the local repo
    and uses GitHub tools to push/PR.
    """
    return Agent(
        role="SEO Code Generator",
        goal="Apply SEO_Analyst remediation tasks by editing the repo and pushing changes",
        backstory="""You are a code-focused SEO engineer. You read remediation tasks and
        implement them directly in the repository. You work in the preset repo (see tools) unless instructed otherwise. You:
        - Clone/sync the repo
        - Create a branch
        - Apply file edits (metadata, titles, descriptions, headings, links, canonicals)
        - Stage, commit, push, and optionally create a PR
        - Log each change: issue -> file path -> action taken
        """,
        tools=[
            clone_repository,
            create_branch,
            get_repo_status,
            write_file_to_repo,
            read_file_from_repo,
            stage_files,
            commit_changes,
            push_changes,
            create_pull_request,
            list_all_pages,
            get_tasks_for_page,
            generate_page_changes
        ],
        verbose=True,
        allow_delegation=False
    )


def create_remediation_crew(audit_file: str, create_pr: bool = True) -> Crew:
    """
    Create a crew for processing SEO remediation.

    Args:
        audit_file: Path to the SEO audit JSON file
        create_pr: Whether to create a PR after making changes

    Returns:
        Configured Crew instance
    """
    code_agent = create_code_generator_agent()

    task_description = f"""
    Process the SEO audit file and implement all recommended fixes automatically (no HITL).

    Audit File: {audit_file}
    Repository: {PRESET_REPO_URL}
    Owner: {PRESET_REPO_OWNER}
    Repo Name: {PRESET_REPO_NAME}

    Steps to follow:
    1. Clone/sync the repository using clone_repository with URL: {PRESET_REPO_URL}
    2. IMPORTANT: First, examine the existing codebase structure to understand where pages are located.
       - Check if pages are in 'src/app/' or 'app/' directory
       - Read existing page.tsx and layout.tsx files to understand the current pattern
    3. Create a new branch called 'seo/audit-fixes-{datetime.now().strftime("%Y%m%d")}'
    4. Load the audit file and list all pages needing fixes
    5. For each task from SEO_Analyst:
       a. Generate the appropriate code fix
       b. APPLY the fix to the CORRECT directory (typically src/app/ for projects using src directory)
       c. Log the change: issue -> file path -> action (append to change log file, e.g., SEO_Audit_Report.txt in repo root)
       d. Stage the changes
    6. Commit all changes with a descriptive message
    7. Push the branch to remote
    {"8. Create a Pull Request with a summary of all fixes" if create_pr else ""}

    Important:
    - ALWAYS check existing codebase structure before writing files
    - If the project uses 'src/app/', write metadata files to 'src/app/[route]/metadata.ts'
    - Do NOT create a separate 'app/' directory if one doesn't exist
    - Group related changes into logical commits
    - Use clear commit messages that reference the SEO issues being fixed
    - For Next.js App Router, metadata.ts exports must be re-exported from page.tsx or layout.tsx to work
    - Append a change log entry for every applied task (issue, file path, action)
    - Prioritize CRITICAL and HIGH severity issues first
    """

    task = Task(
        description=task_description,
        expected_output="A detailed report of all SEO fixes implemented, including per-task change log (issue -> file path -> action), files modified, commits made, and PR link if created",
        agent=code_agent
    )

    return Crew(
        agents=[code_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )


# ============================================================================
# HIGH-LEVEL MANAGER
# ============================================================================

class SEORemediationManager:
    """
    High-level manager for SEO remediation operations.
    Coordinates between audit parsing and GitHub operations.
    """

    def __init__(self, audit_file: str):
        self.audit_file = audit_file
        self.audit_data = None
        self.parser = None
        self.generator = None
        self.github = GitHubManager()

    def load_audit(self) -> Dict[str, Any]:
        """Load and parse the audit file."""
        with open(self.audit_file, 'r') as f:
            self.audit_data = json.load(f)

        self.parser = AuditParser(self.audit_data)
        self.parser.parse()

        return self.parser.get_summary()

    def process_all_fixes(self, create_pr: bool = True) -> Dict[str, Any]:
        """
        Process all SEO fixes using the CrewAI agent.

        Args:
            create_pr: Whether to create a PR after making changes

        Returns:
            Results of the remediation process
        """
        crew = create_remediation_crew(self.audit_file, create_pr)
        result = crew.kickoff()
        return {"status": "completed", "result": str(result)}

    def process_single_page(self, page_url: str) -> Dict[str, Any]:
        """
        Process fixes for a single page.

        Args:
            page_url: URL of the page to fix

        Returns:
            Results of the fix
        """
        code_agent = create_code_generator_agent()
        instruction = f"Apply all remediation tasks for page: {page_url}"

        task = Task(
            description=instruction,
            expected_output="Applied fixes for the page with change log, files modified, and commit/push details",
            agent=code_agent
        )

        crew = Crew(
            agents=[code_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        return {"status": "completed", "result": str(result)}

    def generate_all_metadata_files(self) -> List[Dict[str, Any]]:
        """
        Generate metadata files for all pages needing fixes.

        Returns:
            List of generated files with their content
        """
        return []

    def execute_with_agent(self, instruction: str) -> str:
        """
        Execute a custom instruction using the CrewAI agent.

        Args:
            instruction: Natural language instruction for the agent

        Returns:
            Agent's response
        """
        agent = create_seo_remediation_agent()

        task = Task(
            description=f"""
            Audit File: {self.audit_file}
            Repository: {PRESET_REPO_URL}

            User Instruction:
            {instruction}
            """,
            expected_output="Detailed report of actions taken",
            agent=agent
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        return str(result)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("  SEO Remediation Agent")
    print("  Automated SEO Fix Implementation")
    print("=" * 60)


def interactive_mode(audit_file: str):
    """Run the agent in interactive mode."""
    print_banner()

    print(f"\n[Audit File]: {audit_file}")
    print(f"[Repository]: {PRESET_REPO_URL}")

    manager = SEORemediationManager(audit_file)

    try:
        summary = manager.load_audit()
        print("\n[Audit Summary]")
        print(f"  Total Tasks: {summary['total_tasks']}")
        print(f"  Pages Affected: {summary['pages_affected']}")
        print(f"  By Severity: {summary['by_severity']}")
        print(f"  By Action: {summary['by_action']}")
    except Exception as e:
        print(f"\nError loading audit: {e}")
        return

    print("\n[Commands]")
    print("  list        - List all pages needing fixes")
    print("  show <url>  - Show fixes for a specific page")
    print("  fix all     - Process all fixes with the agent")
    print("  fix <url>   - Generate fix for a specific page")
    print("  preview     - Preview all generated metadata files")
    print("  help        - Show this help")
    print("  quit        - Exit")

    while True:
        try:
            user_input = input("\nCommand > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  list        - List all pages needing fixes")
                print("  show <url>  - Show fixes for a specific page")
                print("  fix all     - Process all fixes with the agent")
                print("  fix <url>   - Generate fix for a specific page")
                print("  preview     - Preview all generated metadata files")
                continue

            if user_input.lower() == 'list':
                for url, page in manager.parser.pages.items():
                    print(f"\n  {url}")
                    print(f"    Path: {page.page_path}")
                    print(f"    Fixes: {len(page.fixes)}")
                    actions = list(set(t.action for t in page.fixes))
                    print(f"    Actions: {', '.join(actions)}")
                continue

            if user_input.lower().startswith('show '):
                url = user_input[5:].strip()
                if url in manager.parser.pages:
                    page = manager.parser.pages[url]
                    print(f"\n[{url}]")
                    for task in page.fixes:
                        print(f"\n  - {task.issue}")
                        print(f"    Severity: {task.severity}")
                        print(f"    Action: {task.action}")
                        if task.current_value:
                            print(f"    Current: {task.current_value[:60]}...")
                else:
                    print(f"Page not found: {url}")
                continue

            if user_input.lower() == 'fix all':
                print("\nStarting full remediation process...")
                print("This will create a branch, make changes, and create a PR.")
                confirm = input("Continue? (y/n) > ").strip().lower()
                if confirm == 'y':
                    result = manager.process_all_fixes(create_pr=True)
                    print(f"\nResult: {result}")
                continue

            if user_input.lower().startswith('fix '):
                url = user_input[4:].strip()
                result = manager.process_single_page(url)
                if "error" in result:
                    print(f"\nError: {result['error']}")
                else:
                    print(f"\n[Generated Fix for {url}]")
                    print(f"File: {result['file_path']}")
                    print(f"Fixes: {', '.join(result['fixes_applied'])}")
                    print(f"\nCode:\n{result['code']}")
                continue

            if user_input.lower() == 'preview':
                files = manager.generate_all_metadata_files()
                for f in files:
                    print(f"\n{'='*40}")
                    print(f"File: {f['file_path']}")
                    print(f"URL: {f['page_url']}")
                    print(f"{'='*40}")
                    print(f"{f['code']}")
                continue

            # Custom instruction
            print(f"\nExecuting: {user_input}")
            result = manager.execute_with_agent(user_input)
            print(f"\nResult:\n{result}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    """Main entry point."""
    import sys

    print_banner()

    # Check for audit file argument
    if len(sys.argv) < 2:
        # Look for recent audit files
        import glob
        audit_files = glob.glob("**/seo_pipeline_*.json", recursive=True)
        audit_files += glob.glob("**/audit_result_*.json", recursive=True)

        if audit_files:
            print("\nFound audit files:")
            for i, f in enumerate(audit_files[:5], 1):
                print(f"  {i}. {f}")

            choice = input("\nSelect a file (number) or enter path: ").strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(audit_files):
                    audit_file = audit_files[idx]
                else:
                    audit_file = choice
            except ValueError:
                audit_file = choice
        else:
            print("\nNo audit files found.")
            print("Usage: python seo_remediation_agent.py <audit_file.json>")
            return
    else:
        audit_file = sys.argv[1]

    if not os.path.exists(audit_file):
        print(f"\nError: File not found: {audit_file}")
        return

    interactive_mode(audit_file)


if __name__ == "__main__":
    main()

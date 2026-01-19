"""
SEO Multi-Agent Orchestrator
Coordinates the SEO_Scout (Crawler), SEO_Analyst (Analysis),
SEO_Remediation (HITL), and GitHub agents into a unified pipeline.

Pipeline Flow:
  User Input (URL) -> SEO_Scout -> Crawl Data -> SEO_Analyst -> Audit Report

This orchestrator manages the handoff between agents and produces comprehensive reports.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from crewai import Agent, Task, Crew, Process

from dotenv import load_dotenv

load_dotenv()
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Import agent components
from seo_scout_agent import (
    SiteCrawler,
    crawl_website,
    check_robots_txt,
    fetch_page_data,
    extract_seo_elements,
    map_links,
    MAX_DEPTH,
    MAX_PAGES
)

from seo_analyst_agent import (
    SiteAnalyzer,
    analyze_crawl_data,
    validate_json_ld,
    check_keyword_intent,
    print_detailed_report
)

from seo_remediation_agent import SEORemediationManager


# ============================================================================
# ORCHESTRATOR CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Configuration for the SEO pipeline."""

    def __init__(
        self,
        max_depth: int = MAX_DEPTH,
        max_pages: int = MAX_PAGES,
        verbose: bool = True,
        save_intermediate: bool = True,
        output_dir: str = ".",
        apply_remediation: bool = False,
        auto_approve_remediation: bool = False,
        create_pr: bool = True
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.output_dir = output_dir
        self.apply_remediation = apply_remediation
        self.auto_approve_remediation = auto_approve_remediation
        self.create_pr = create_pr


# ============================================================================
# CREWAI MULTI-AGENT SETUP
# ============================================================================

def create_scout_agent() -> Agent:
    """Create the SEO Scout (Crawler) agent."""
    return Agent(
        role="SEO Web Crawler",
        goal="Traverse target websites to harvest comprehensive technical SEO data",
        backstory="""You are SEO_Scout, the "Eyes" of the SEO analysis system. You are a precise
        and autonomous Web Crawler Agent. Your job is to gather data without judgment or analysis.
        You report the technical reality of a website exactly as it is. You respect robots.txt,
        implement rate limiting, and collect all SEO-relevant data from every reachable page.""",
        tools=[check_robots_txt, fetch_page_data, extract_seo_elements, map_links],
        verbose=True,
        allow_delegation=False
    )


def create_analyst_agent() -> Agent:
    """Create the SEO Analyst agent."""
    return Agent(
        role="Senior SEO Strategist",
        goal="Analyze crawl data to identify SEO issues and generate prioritized remediation tasks",
        backstory="""You are SEO_Analyst, the "Brain" of the SEO analysis system. You are a Senior
        SEO Strategist who evaluates context, relevance, and user intent - not just tag presence.
        You transform raw technical data into actionable remediation tasks. You never give generic
        advice - you specify exactly which elements need to change and why.""",
        tools=[validate_json_ld, check_keyword_intent],
        verbose=True,
        allow_delegation=False
    )


def create_crawl_task(agent: Agent, url: str, config: PipelineConfig) -> Task:
    """Create the crawling task."""
    return Task(
        description=f"""
        Perform a comprehensive crawl of the website: {url}

        Configuration:
        - Maximum crawl depth: {config.max_depth}
        - Maximum pages to crawl: {config.max_pages}

        Steps:
        1. Check robots.txt compliance before crawling
        2. Crawl the seed URL and extract all SEO data
        3. Follow all internal links up to the configured depth
        4. For each page, extract: title, meta description, headings, images, links
        5. Compile all data into a structured JSON format

        Return the complete crawl data as JSON.
        """,
        expected_output="Complete crawl data JSON with all pages and their SEO elements",
        agent=agent
    )


def create_analysis_task(agent: Agent, context_task: Task) -> Task:
    """Create the analysis task that depends on crawl results."""
    return Task(
        description="""
        Analyze the crawl data provided and produce a comprehensive SEO audit.

        For the entire site, give site wide recommendations.
        
        For each page in the crawl data:
        1. Evaluate title and meta description quality and intent alignment
        2. Check heading structure and hierarchy (H1 -> H2 -> H3)
        3. Assess image alt text coverage and accessibility
        4. Review internal linking patterns
        5. Verify canonical tags and redirect status
        6. Identify structured data opportunities (Product, Article, FAQ, etc.)

        Generate specific, actionable remediation tasks with severity ratings:
        - CRITICAL: Technical breakers (404/500 errors)
        - HIGH: Missed intent alignment, missing schema
        - MEDIUM: Optimization opportunities
        - LOW: Minor improvements

        Do NOT give generic advice. Be specific about what to change.
        """,
        expected_output="""Comprehensive SEO audit JSON containing:
        - Overall site score and summary statistics
        - Site-wide pattern issues
        - Per-page audit results with component scores
        - Prioritized list of remediation tasks""",
        agent=agent,
        context=[context_task]
    )


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class SEOPipeline:
    """
    Orchestrates the multi-agent SEO analysis pipeline.

    Flow: URL -> SEO_Scout (Crawl) -> SEO_Analyst (Analyze) -> Report
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        self.start_time = None
        self.crawl_data = None
        self.audit_data = None
        self.audit_file_path = None
        self.remediation_result = None
        self.remediation_approved = False

    def run(self, seed_url: str) -> Dict[str, Any]:
        """
        Execute the complete SEO analysis pipeline.

        Args:
            seed_url: The starting URL to crawl and analyze

        Returns:
            Dictionary containing complete pipeline results
        """
        self.start_time = datetime.utcnow()

        if self.config.verbose:
            self._print_header(seed_url)

        # Phase 1: Crawling
        if self.config.verbose:
            self._print_phase("PHASE 1: CRAWLING", "SEO_Scout is gathering data...")

        self.crawl_data = self._run_crawler(seed_url)

        if self.crawl_data.get("status") == "BLOCKED":
            return self._compile_results(status="BLOCKED_BY_ROBOTS")

        if self.config.save_intermediate:
            self._save_intermediate("crawl", self.crawl_data)

        # Phase 2: Analysis
        if self.config.verbose:
            self._print_phase("PHASE 2: ANALYSIS", "SEO_Analyst is evaluating data...")

        self.audit_data = self._run_analyzer(self.crawl_data)

        if self.config.save_intermediate:
            self.audit_file_path = self._save_intermediate("audit", self.audit_data)

        # Phase 3: Remediation (optional, HITL)
        if self.config.apply_remediation:
            if self.config.verbose:
                self._print_phase("PHASE 3: REMEDIATION", "Waiting for approval to apply fixes...")
            self.remediation_approved = self.config.auto_approve_remediation or self._prompt_remediation_approval()
            if self.remediation_approved:
                if self.config.verbose:
                    print("  Approval granted. Running remediation + GitHub agent...\n")
                self.remediation_result = self._run_remediation(
                    self.audit_data,
                    create_pr=self.config.create_pr
                )
                if self.config.save_intermediate:
                    self._save_intermediate("remediation", self.remediation_result)
            else:
                self.remediation_result = {
                    "status": "SKIPPED_BY_USER",
                    "approved": False,
                    "message": "User declined remediation"
                }

        # Compile final results
        return self._compile_results(status="COMPLETED")

    def run_with_crew(self, seed_url: str) -> Dict[str, Any]:
        """
        Execute the pipeline using CrewAI's native task orchestration.

        This method uses CrewAI's Process.sequential to chain the agents.

        Args:
            seed_url: The starting URL to crawl and analyze

        Returns:
            Dictionary containing complete pipeline results
        """
        self.start_time = datetime.utcnow()

        if self.config.verbose:
            self._print_header(seed_url)
            print("\n  Using CrewAI sequential process for agent coordination...")

        # Create agents
        scout = create_scout_agent()
        analyst = create_analyst_agent()

        # Create tasks with dependencies
        crawl_task = create_crawl_task(scout, seed_url, self.config)
        analysis_task = create_analysis_task(analyst, crawl_task)

        # Create and run the crew
        crew = Crew(
            agents=[scout, analyst],
            tasks=[crawl_task, analysis_task],
            process=Process.sequential,
            verbose=self.config.verbose
        )

        # Execute the crew
        result = crew.kickoff()

        return {
            "pipeline_id": self.pipeline_id,
            "seed_url": seed_url,
            "crew_result": str(result),
            "status": "COMPLETED"
        }

    def _run_crawler(self, seed_url: str) -> Dict[str, Any]:
        """Execute the crawler phase."""
        crawler = SiteCrawler(
            seed_url,
            max_depth=self.config.max_depth,
            max_pages=self.config.max_pages
        )
        return crawler.crawl()

    def _run_analyzer(self, crawl_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis phase."""
        analyzer = SiteAnalyzer(crawl_data)
        return analyzer.analyze()

    def _run_remediation(self, audit_data: Dict[str, Any], create_pr: bool) -> Dict[str, Any]:
        """Execute remediation via the SEO Remediation agent and GitHub agent."""
        audit_path = self.audit_file_path or self._save_intermediate("audit", audit_data)
        manager = SEORemediationManager(audit_path)

        try:
            summary = manager.load_audit()
        except Exception as e:
            return {
                "status": "FAILED_TO_LOAD_AUDIT",
                "approved": True,
                "error": str(e)
            }

        try:
            result = manager.process_all_fixes(create_pr=create_pr)
            return {
                "status": "COMPLETED",
                "approved": True,
                "summary": summary,
                "result": result
            }
        except Exception as e:
            return {
                "status": "FAILED",
                "approved": True,
                "error": str(e)
            }

    def _compile_results(self, status: str) -> Dict[str, Any]:
        """Compile the final pipeline results."""
        end_time = datetime.utcnow()
        duration_ms = int((end_time - self.start_time).total_seconds() * 1000)

        results = {
            "pipeline_id": self.pipeline_id,
            "status": status,
            "timestamp_start": self.start_time.isoformat() + "Z",
            "timestamp_end": end_time.isoformat() + "Z",
            "duration_ms": duration_ms,
            "config": {
                "max_depth": self.config.max_depth,
                "max_pages": self.config.max_pages
            }
        }

        if self.crawl_data:
            results["crawl_summary"] = {
                "seed_url": self.crawl_data.get("seed_url"),
                "base_domain": self.crawl_data.get("base_domain"),
                "pages_crawled": self.crawl_data.get("summary", {}).get("pages_crawled", 0),
                "total_internal_links": self.crawl_data.get("summary", {}).get("total_internal_links_found", 0),
                "total_external_links": self.crawl_data.get("summary", {}).get("total_external_links_found", 0)
            }
            results["crawl_data"] = self.crawl_data

        if self.audit_data:
            results["audit_summary"] = {
                "overall_score": self.audit_data.get("summary", {}).get("overall_score", 0),
                "total_issues": self.audit_data.get("summary", {}).get("total_issues", 0),
                "issues_by_severity": self.audit_data.get("summary", {}).get("issues_by_severity", {})
            }
            results["audit_data"] = self.audit_data

        if self.remediation_result is not None:
            results["remediation_summary"] = {
                "status": self.remediation_result.get("status"),
                "approved": self.remediation_result.get("approved", False),
                "message": self.remediation_result.get("message"),
            }
            results["remediation_result"] = self.remediation_result

        return results

    def _save_intermediate(self, phase: str, data: Dict[str, Any]):
        """Save intermediate results to file."""
        filename = f"{self.config.output_dir}/{self.pipeline_id}_{phase}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        if self.config.verbose:
            print(f"  Saved {phase} results to: {filename}")
        return filename

    def _print_header(self, seed_url: str):
        """Print pipeline header."""
        print("\n" + "=" * 80)
        print("           SEO MULTI-AGENT PIPELINE")
        print("=" * 80)
        print(f"\n  Pipeline ID:  {self.pipeline_id}")
        print(f"  Target URL:   {seed_url}")
        print(f"  Max Depth:    {self.config.max_depth}")
        print(f"  Max Pages:    {self.config.max_pages}")
        print(f"  Started:      {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    def _print_phase(self, title: str, description: str):
        """Print phase header."""
        print(f"\n{'─' * 80}")
        print(f"  {title}")
        print(f"{'─' * 80}")
        print(f"  {description}\n")

    def _prompt_remediation_approval(self) -> bool:
        """HITL prompt to approve remediation step."""
        try:
            response = input("Approve remediation and GitHub changes? (y/N): ").strip().lower()
            return response == "y"
        except EOFError:
            return False
        except Exception:
            return False


# ============================================================================
# REPORTING
# ============================================================================

def print_pipeline_report(results: Dict[str, Any]):
    """Print comprehensive pipeline report."""

    print("\n" + "=" * 80)
    print("              SEO PIPELINE EXECUTION REPORT")
    print("=" * 80)

    # Pipeline Info
    print(f"\n{'─' * 80}")
    print("  PIPELINE EXECUTION")
    print(f"{'─' * 80}")
    print(f"  Pipeline ID:    {results.get('pipeline_id', 'N/A')}")
    print(f"  Status:         {results.get('status', 'N/A')}")
    print(f"  Duration:       {results.get('duration_ms', 0) / 1000:.2f} seconds")

    # Crawl Summary
    crawl_summary = results.get("crawl_summary", {})
    if crawl_summary:
        print(f"\n{'─' * 80}")
        print("  CRAWL PHASE (SEO_Scout)")
        print(f"{'─' * 80}")
        print(f"  Domain:              {crawl_summary.get('base_domain', 'N/A')}")
        print(f"  Pages Crawled:       {crawl_summary.get('pages_crawled', 0)}")
        print(f"  Internal Links:      {crawl_summary.get('total_internal_links', 0)}")
        print(f"  External Links:      {crawl_summary.get('total_external_links', 0)}")

    # Audit Summary
    audit_summary = results.get("audit_summary", {})
    if audit_summary:
        print(f"\n{'─' * 80}")
        print("  ANALYSIS PHASE (SEO_Analyst)")
        print(f"{'─' * 80}")
        print(f"  Overall Score:       {audit_summary.get('overall_score', 0)}/100")
        print(f"  Total Issues:        {audit_summary.get('total_issues', 0)}")
        print(f"\n  Issues by Severity:")
        severity_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        for sev, count in audit_summary.get("issues_by_severity", {}).items():
            icon = severity_icons.get(sev, "⚪")
            print(f"    {icon} {sev}: {count}")

    remediation_summary = results.get("remediation_summary", {})
    if remediation_summary:
        print(f"\n{'─' * 80}")
        print("  REMEDIATION PHASE (HITL + GitHub)")
        print(f"{'─' * 80}")
        print(f"  Approved:           {remediation_summary.get('approved', False)}")
        print(f"  Status:             {remediation_summary.get('status', 'N/A')}")
        message = remediation_summary.get("message")
        if message:
            print(f"  Note:               {message}")

    # Print detailed audit report if available
    audit_data = results.get("audit_data")
    if audit_data:
        print_detailed_report(audit_data)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_seo_pipeline(
    url: str,
    max_depth: int = MAX_DEPTH,
    max_pages: int = MAX_PAGES,
    verbose: bool = True,
    use_crew: bool = False,
    apply_fixes: bool = False,
    auto_approve_fixes: bool = False,
    create_pr: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for running the SEO multi-agent pipeline.

    Args:
        url: The seed URL to crawl and analyze
        max_depth: Maximum crawl depth (default: 3)
        max_pages: Maximum pages to crawl (default: 100)
        verbose: Print progress output (default: True)
        use_crew: Use CrewAI's native orchestration (default: False)

    Returns:
        Dictionary containing complete pipeline results
    """
    config = PipelineConfig(
        max_depth=max_depth,
        max_pages=max_pages,
        verbose=verbose,
        output_dir=output_dir or ".",
        apply_remediation=apply_fixes,
        auto_approve_remediation=auto_approve_fixes,
        create_pr=create_pr
    )

    pipeline = SEOPipeline(config)

    if use_crew:
        return pipeline.run_with_crew(url)
    else:
        return pipeline.run(url)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="SEO Multi-Agent Pipeline - Crawl and analyze websites for SEO issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python seo_orchestrator.py https://example.com
  python seo_orchestrator.py https://example.com --depth 2 --pages 50
  python seo_orchestrator.py https://example.com --use-crew

Pipeline Flow:
  URL → SEO_Scout (Crawler) → Crawl Data → SEO_Analyst → Audit Report
      → (HITL Approval) → SEO_Remediation → GitHub (branch/PR)
        """
    )

    parser.add_argument("url", help="The seed URL to crawl and analyze")
    parser.add_argument("--depth", type=int, default=MAX_DEPTH,
                        help=f"Maximum crawl depth (default: {MAX_DEPTH})")
    parser.add_argument("--pages", type=int, default=MAX_PAGES,
                        help=f"Maximum pages to crawl (default: {MAX_PAGES})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--use-crew", action="store_true",
                        help="Use CrewAI's native sequential process")
    parser.add_argument("--apply-fixes", action="store_true",
                        help="Run remediation (HITL) after analysis")
    parser.add_argument("--auto-approve-fixes", action="store_true",
                        help="Skip HITL prompt and auto-approve remediation")
    parser.add_argument("--no-pr", action="store_true",
                        help="Do not create a Pull Request after remediation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (default: auto-generated)")

    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                    SEO MULTI-AGENT SYSTEM                                 ║
    ║                                                                           ║
    ║   Agents:                                                                 ║
    ║   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   ┌──────────┐  ║
    ║   │  SEO_Scout  │ → │ SEO_Analyst │ → │ SEO_Remediation │ → │ GitHub   │  ║
    ║   │  (Crawler)  │   │ (Analyzer)  │   │ (HITL + fixes)  │   │ (PR/CI)  │  ║
    ║   └─────────────┘   └─────────────┘   └─────────────────┘   └──────────┘  ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Run the pipeline
    results = run_seo_pipeline(
        url=args.url,
        max_depth=args.depth,
        max_pages=args.pages,
        verbose=not args.quiet,
        use_crew=args.use_crew,
        apply_fixes=args.apply_fixes,
        auto_approve_fixes=args.auto_approve_fixes,
        create_pr=not args.no_pr,
        output_dir=args.output or "."
    )

    # Print the report
    print_pipeline_report(results)

    # Save results
    output_file = args.output or f"seo_pipeline_{results['pipeline_id']}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"  Full results saved to: {output_file}")
    print(f"{'=' * 80}\n")

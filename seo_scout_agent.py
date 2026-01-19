"""
SEO Scout Agent - A CrewAI-based web crawler for SEO data extraction.
Based on the SEO_Scout agent blueprint.
Crawls entire websites by following internal links.
"""

import time
import random
import uuid
import json
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Optional, Set, List, Dict

import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

from crewai import Agent, Task, Crew
from crewai.tools import tool

from dotenv import load_dotenv

load_dotenv()
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ============================================================================
# CONFIGURATION / GUARDRAILS
# ============================================================================

MAX_DEPTH = 3  # Maximum recursion depth for crawling
MAX_PAGES = 100  # Maximum number of pages to crawl
RATE_LIMIT_MIN = 1  # Minimum seconds between requests
RATE_LIMIT_MAX = 3  # Maximum seconds between requests
# Increase or override via env if pages are large; 0 means no truncation.
USER_AGENT = "SEO_Agent_Bot/1.0"


# ============================================================================
# STANDALONE UTILITY FUNCTIONS (for recursive crawling)
# ============================================================================

def _check_robots_txt(url: str, user_agent: str = USER_AGENT) -> bool:
    """Check if crawling is allowed by robots.txt."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def _get_robots_txt_content(url: str, user_agent: str = USER_AGENT) -> dict:
    """Fetch robots.txt content; return content or None if missing/unreachable."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=10)
        if resp.status_code == 200:
            return {
                "url": robots_url,
                "status_code": resp.status_code,
                "content": resp.text
            }
        else:
            return {
                "url": robots_url,
                "status_code": resp.status_code,
                "content": None
            }
    except Exception as e:
        return {
            "url": robots_url if 'robots_url' in locals() else "",
            "status_code": 0,
            "content": None,
            "error": str(e)
        }


def _fetch_page(url: str, user_agent: str = USER_AGENT) -> dict:
    """Fetch a single page and return its data."""
    headers = {"User-Agent": user_agent}

    try:
        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response_ms = int((time.time() - start_time) * 1000)


        raw_html = response.text

        return {
            "status_code": response.status_code,
            "raw_html": raw_html,
            "response_ms": response_ms,
            "final_url": response.url,
            "headers": dict(response.headers),
            "error": None
        }
    except requests.Timeout:
        return {
            "status_code": 0,
            "raw_html": "",
            "response_ms": -1,
            "final_url": url,
            "headers": {},
            "error": "TIMEOUT"
        }
    except requests.RequestException as e:
        return {
            "status_code": 0,
            "raw_html": "",
            "response_ms": -1,
            "final_url": url,
            "headers": {},
            "error": str(e)
        }


def _extract_seo_elements(raw_html: str) -> dict:
    """Extract SEO elements from HTML."""
    if not raw_html or raw_html.strip() == "":
        return {"error": "FAILED_PARSE", "message": "Empty or corrupted HTML"}

    try:
        soup = BeautifulSoup(raw_html, 'html.parser')

        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else None

        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_desc.get('content', None) if meta_desc else None

        canonical = soup.find('link', rel='canonical')
        canonical_tag = canonical.get('href', None) if canonical else None

        robots = soup.find('meta', attrs={'name': 'robots'})
        robots_meta = robots.get('content', None) if robots else None

        h1_tags = [h.get_text(strip=True) for h in soup.find_all('h1')]
        h2_tags = [h.get_text(strip=True) for h in soup.find_all('h2')]
        h3_tags = [h.get_text(strip=True) for h in soup.find_all('h3')]

        images = soup.find_all('img')
        image_alts = [
            {"src": img.get('src', ''), "alt": img.get('alt', '')}
            for img in images
        ]

        return {
            "title": title,
            "meta_description": meta_description,
            "h1_tags": h1_tags,
            "h2_tags": h2_tags,
            "h3_tags": h3_tags,
            "canonical_tag": canonical_tag,
            "robots_meta": robots_meta,
            "image_alts": image_alts
        }
    except Exception as e:
        return {"error": "FAILED_PARSE", "message": str(e)}


def _extract_links(raw_html: str, base_url: str) -> dict:
    """Extract and categorize links from HTML."""
    if not raw_html:
        return {"internal_links": [], "external_links": [], "malformed_links": []}

    try:
        soup = BeautifulSoup(raw_html, 'html.parser')
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc

        internal_links = []
        external_links = []
        malformed_links = []

        for anchor in soup.find_all('a', href=True):
            href = anchor.get('href', '').strip()

            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue

            # Skip non-HTML resources
            if any(href.lower().endswith(ext) for ext in
                   ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.zip', '.mp4', '.mp3']):
                continue

            try:
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)

                # Normalize URL (remove fragments)
                normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                if parsed_url.query:
                    normalized_url += f"?{parsed_url.query}"

                if not parsed_url.scheme or not parsed_url.netloc:
                    malformed_links.append(href)
                elif parsed_url.netloc == base_domain:
                    internal_links.append(normalized_url)
                else:
                    external_links.append(normalized_url)
            except Exception:
                malformed_links.append(href)

        # Remove duplicates
        internal_links = list(dict.fromkeys(internal_links))
        external_links = list(dict.fromkeys(external_links))
        malformed_links = list(dict.fromkeys(malformed_links))

        return {
            "internal_links": internal_links,
            "external_links": external_links,
            "malformed_links": malformed_links
        }
    except Exception as e:
        return {"internal_links": [], "external_links": [], "malformed_links": [], "error": str(e)}


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if normalized.endswith('/'):
        normalized = normalized[:-1]
    return normalized.lower()


# ============================================================================
# CREWAI TOOLS (wrap the utility functions)
# ============================================================================

@tool("Check Robots.txt")
def check_robots_txt(url: str, user_agent: str = USER_AGENT) -> bool:
    """
    Parses the domain's robots.txt file to ensure the agent is allowed to crawl.
    Returns True if allowed, False if disallowed.
    """
    return _check_robots_txt(url, user_agent)


@tool("Fetch Page Data")
def fetch_page_data(url: str, user_agent: str = USER_AGENT) -> dict:
    """
    Executes a GET request to the target URL and returns page data including
    status_code, raw_html, response_ms, final_url, and headers.
    """
    return _fetch_page(url, user_agent)


@tool("Extract SEO Elements")
def extract_seo_elements(raw_html: str) -> dict:
    """
    Parses HTML to extract SEO-relevant elements: title, meta_description,
    headings (h1, h2, h3), canonical_tag, robots_meta, and image_alts.
    """
    return _extract_seo_elements(raw_html)


@tool("Map Links")
def map_links(raw_html: str, base_url: str) -> dict:
    """
    Identifies and categorizes all hyperlinks on the page into internal_links,
    external_links, and malformed_links.
    """
    return _extract_links(raw_html, base_url)


# ============================================================================
# RECURSIVE SITE CRAWLER
# ============================================================================

class SiteCrawler:
    """Recursive website crawler that collects SEO data from all internal pages."""

    def __init__(self, seed_url: str, max_depth: int = MAX_DEPTH, max_pages: int = MAX_PAGES):
        self.seed_url = seed_url
        self.max_depth = max_depth
        self.max_pages = max_pages

        parsed = urlparse(seed_url)
        self.base_domain = parsed.netloc
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"

        self.visited_urls: Set[str] = set()
        self.pages_data: List[dict] = []
        self.all_internal_links: Set[str] = set()
        self.all_external_links: Set[str] = set()
        self.blocked_urls: List[str] = []
        self.failed_urls: List[dict] = []
        self.robots_txt: Dict[str, any] = {}

        self.scan_id = f"crawl_{uuid.uuid4().hex[:8]}"
        self.start_time = None

    def crawl(self) -> dict:
        """Start the crawl and return comprehensive site data."""
        self.start_time = datetime.utcnow()

        print(f"[SEO_Scout] Starting full site crawl for: {self.seed_url}")
        print(f"[SEO_Scout] Max depth: {self.max_depth}, Max pages: {self.max_pages}")

        # Fetch robots.txt content up front (store even if missing)
        self.robots_txt = _get_robots_txt_content(self.seed_url)

        # Check robots.txt for the seed URL first
        if not _check_robots_txt(self.seed_url):
            print(f"[SEO_Scout] BLOCKED by robots.txt: {self.seed_url}")
            return self._compile_final_result(status="BLOCKED")

        # Start recursive crawl
        self._crawl_page(self.seed_url, depth=0)

        return self._compile_final_result(status="COMPLETED")

    def _crawl_page(self, url: str, depth: int):
        """Recursively crawl a page and its internal links."""
        # Normalize URL for tracking
        normalized = _normalize_url(url)

        # Check stopping conditions
        if normalized in self.visited_urls:
            return
        if len(self.pages_data) >= self.max_pages:
            print(f"[SEO_Scout] Reached max pages limit ({self.max_pages})")
            return
        if depth > self.max_depth:
            return

        # Domain confinement guardrail
        parsed = urlparse(url)
        if parsed.netloc != self.base_domain:
            return

        # Mark as visited
        self.visited_urls.add(normalized)

        # Rate limiting guardrail
        time.sleep(random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX))

        print(f"[SEO_Scout] Crawling (depth={depth}): {url}")

        # Check robots.txt for this specific URL
        if not _check_robots_txt(url):
            print(f"[SEO_Scout] BLOCKED by robots.txt: {url}")
            self.blocked_urls.append(url)
            return

        # Fetch page
        page_result = _fetch_page(url)

        if page_result.get("error"):
            self.failed_urls.append({"url": url, "error": page_result["error"]})
            return

        if page_result["status_code"] != 200:
            self.failed_urls.append({
                "url": url,
                "error": f"HTTP_{page_result['status_code']}"
            })
            return

        # Extract SEO elements
        seo_elements = _extract_seo_elements(page_result["raw_html"])

        # Extract links
        links = _extract_links(page_result["raw_html"], url)

        # Track all links
        self.all_internal_links.update(links["internal_links"])
        self.all_external_links.update(links["external_links"])

        # Compile page data
        page_data = {
            "url": url,
            "final_url": page_result["final_url"],
            "is_redirect": url != page_result["final_url"],
            "http_status": page_result["status_code"],
            "load_time_ms": page_result["response_ms"],
            "crawl_depth": depth,
            "page_data": {
                "title": seo_elements.get("title"),
                "meta_description": seo_elements.get("meta_description"),
                "canonical": seo_elements.get("canonical_tag"),
                "robots_tag": seo_elements.get("robots_meta"),
                "headings": {
                    "h1": seo_elements.get("h1_tags", []),
                    "h2": seo_elements.get("h2_tags", []),
                    "h3": seo_elements.get("h3_tags", [])
                },
                "images": seo_elements.get("image_alts", [])
            },
            "links": {
                "internal_count": len(links["internal_links"]),
                "external_count": len(links["external_links"]),
                "internal_links": links["internal_links"],
                "external_links": links["external_links"],
                "malformed_links": links["malformed_links"]
            }
        }

        self.pages_data.append(page_data)

        # Recursively crawl internal links
        for internal_link in links["internal_links"]:
            if len(self.pages_data) >= self.max_pages:
                break
            self._crawl_page(internal_link, depth + 1)

    def _compile_final_result(self, status: str) -> dict:
        """Compile the final comprehensive JSON result."""
        end_time = datetime.utcnow()
        duration_ms = int((end_time - self.start_time).total_seconds() * 1000)

        # Calculate site-wide statistics
        total_images = sum(len(p["page_data"]["images"]) for p in self.pages_data)
        images_without_alt = sum(
            1 for p in self.pages_data
            for img in p["page_data"]["images"]
            if not img.get("alt")
        )
        pages_without_title = sum(
            1 for p in self.pages_data
            if not p["page_data"]["title"]
        )
        pages_without_meta_desc = sum(
            1 for p in self.pages_data
            if not p["page_data"]["meta_description"]
        )
        pages_without_h1 = sum(
            1 for p in self.pages_data
            if not p["page_data"]["headings"]["h1"]
        )

        return {
            "scan_id": self.scan_id,
            "status": status,
            "seed_url": self.seed_url,
            "base_domain": self.base_domain,
            "robots_txt": {
                "url": self.robots_txt.get("url"),
                "status_code": self.robots_txt.get("status_code"),
                "content": self.robots_txt.get("content"),
                "error": self.robots_txt.get("error")
            },
            "timestamp_start": self.start_time.isoformat() + "Z",
            "timestamp_end": end_time.isoformat() + "Z",
            "duration_ms": duration_ms,
            "crawl_config": {
                "max_depth": self.max_depth,
                "max_pages": self.max_pages,
                "user_agent": USER_AGENT
            },
            "summary": {
                "pages_crawled": len(self.pages_data),
                "pages_blocked": len(self.blocked_urls),
                "pages_failed": len(self.failed_urls),
                "total_internal_links_found": len(self.all_internal_links),
                "total_external_links_found": len(self.all_external_links),
                "total_images": total_images,
                "images_without_alt": images_without_alt,
                "pages_without_title": pages_without_title,
                "pages_without_meta_description": pages_without_meta_desc,
                "pages_without_h1": pages_without_h1
            },
            "pages": self.pages_data,
            "site_graph": {
                "all_internal_links": list(self.all_internal_links),
                "all_external_links": list(self.all_external_links)
            },
            "blocked_urls": self.blocked_urls,
            "failed_urls": self.failed_urls
        }


# ============================================================================
# CREWAI AGENT DEFINITION
# ============================================================================

SEO_SCOUT_SYSTEM_PROMPT = """
You are "SEO_Scout," a precise and autonomous Web Crawler Agent. Your only job is to gather data.
You do not analyze, you do not fix, and you do not judge. You report the technical reality of a
website exactly as it is.

OPERATIONAL WORKFLOW:
1) Compliance: ALWAYS call `check_robots_txt` first. If false, terminate immediately with a "BLOCKED" status.
2) Fetch: Call `fetch_page_data`.
   - If status != 200: Stop and report the error code.
   - If status == 200: Proceed.
3) Extraction: Call `extract_seo_elements` and `map_links` on the raw HTML.
4) Output: Compile all gathered data into the required JSON schema.

CRITICAL RULES:
- Zero Hallucination: If a tag (like Meta Description) is missing, value must be null or "". Do not invent text.
- Redirects: If the `final_url` differs from the input URL, explicitly flag this as a redirect.
- Strict Structure: Do not add conversational text to your output. Return ONLY the JSON object.
- Heading Hierarchy: Strictly differentiate between H1, H2, and H3 tags.

ERROR HANDLING:
- If the HTML is empty or corrupted, return status: "FAILED_PARSE".
- If the request times out, return status: "TIMEOUT".
"""


def create_seo_scout_agent() -> Agent:
    """Creates and returns the SEO Scout agent."""
    return Agent(
        role="SEO Web Crawler",
        goal="Traverse target URLs to harvest high-fidelity technical SEO data without judgment or analysis",
        backstory="""You are SEO_Scout, the "Eyes" of an SEO analysis system. You are a precise
        and autonomous Web Crawler Agent whose only job is to gather data. You do not analyze,
        you do not fix, and you do not judge. You report the technical reality of a website
        exactly as it is. Your output feeds into downstream analyzer agents.""",
        tools=[check_robots_txt, fetch_page_data, extract_seo_elements, map_links],
        verbose=True,
        allow_delegation=False,
        system_prompt=SEO_SCOUT_SYSTEM_PROMPT
    )


def create_site_crawl_task(agent: Agent, target_url: str) -> Task:
    """Creates a task for crawling an entire website."""
    return Task(
        description=f"""
        Perform a comprehensive crawl of the website starting from: {target_url}

        The crawl should:
        1. Start by checking robots.txt compliance
        2. Crawl the seed URL and extract all SEO data
        3. Follow all internal links up to depth {MAX_DEPTH}
        4. Collect SEO data from every reachable page
        5. Track all internal and external links found

        For each page, extract:
        - Title, meta description, canonical URL, robots tag
        - All headings (H1, H2, H3)
        - All images with their alt text
        - Internal and external links

        Return a comprehensive JSON report of the entire website.
        """,
        expected_output="""A comprehensive JSON object containing:
        - scan_id, status, seed_url, timestamps
        - summary statistics (pages crawled, links found, SEO issues)
        - detailed page_data for each crawled page
        - complete site_graph with all internal and external links
        - lists of blocked and failed URLs""",
        agent=agent
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def crawl_website(url: str, max_depth: int = MAX_DEPTH, max_pages: int = MAX_PAGES) -> dict:
    """
    Main function to crawl an entire website and return comprehensive SEO data.

    Args:
        url: The seed URL to start crawling from
        max_depth: Maximum recursion depth (default: 3)
        max_pages: Maximum number of pages to crawl (default: 100)

    Returns:
        Dictionary containing comprehensive crawl results for the entire site
    """
    crawler = SiteCrawler(url, max_depth=max_depth, max_pages=max_pages)
    return crawler.crawl()


def crawl_single_page(url: str) -> dict:
    """
    Crawl a single page using the CrewAI agent.

    Args:
        url: The target URL to crawl

    Returns:
        Dictionary containing the crawl results for a single page
    """
    time.sleep(random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX))

    agent = create_seo_scout_agent()
    task = create_site_crawl_task(agent, url)
    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    return crew.kickoff()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python seo_scout_agent.py <url> [max_depth] [max_pages]")
        print("Example: python seo_scout_agent.py https://example.com 3 50")
        sys.exit(1)

    target_url = sys.argv[1]
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_DEPTH
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_PAGES

    print(f"Starting full site crawl for: {target_url}")
    print(f"Configuration: max_depth={max_depth}, max_pages={max_pages}")
    print("=" * 60)

    result = crawl_website(target_url, max_depth=max_depth, max_pages=max_pages)

    print("\n" + "=" * 60)
    print("CRAWL COMPLETED")
    print("=" * 60)

    # Pretty print the result
    print(json.dumps(result, indent=2))

    # Also save to file
    output_file = f"crawl_result_{result['scan_id']}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")

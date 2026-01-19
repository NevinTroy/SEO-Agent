"""
SEO Analyst Agent - A CrewAI-based SEO analysis agent.
Based on the SEO_Analyst agent blueprint.
Analyzes crawl data and produces actionable remediation tasks.
"""

import json
import uuid
import re
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

from crewai import Agent, Task, Crew
from crewai.tools import tool

from dotenv import load_dotenv

load_dotenv()
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool("Validate JSON-LD")
def validate_json_ld(json_string: str) -> dict:
    """
    Validates the syntax of JSON-LD structured data markup.
    Uses strict JSON parsing to ensure the markup is valid.

    Args:
        json_string: The JSON-LD string to validate

    Returns:
        Dictionary with 'valid' boolean and 'error_msg' if invalid
    """
    try:
        # Parse the JSON
        parsed = json.loads(json_string)

        # Basic JSON-LD structure checks
        if not isinstance(parsed, dict):
            return {"valid": False, "error_msg": "JSON-LD must be an object"}

        if "@context" not in parsed:
            return {"valid": False, "error_msg": "Missing @context property"}

        if "@type" not in parsed:
            return {"valid": False, "error_msg": "Missing @type property"}

        # Validate @context
        context = parsed.get("@context")
        if isinstance(context, str):
            if "schema.org" not in context:
                return {"valid": False, "error_msg": "@context should reference schema.org"}
        elif isinstance(context, list):
            if not any("schema.org" in str(c) for c in context):
                return {"valid": False, "error_msg": "@context should reference schema.org"}

        return {"valid": True, "error_msg": None}

    except json.JSONDecodeError as e:
        return {"valid": False, "error_msg": f"Invalid JSON syntax: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error_msg": f"Validation error: {str(e)}"}


@tool("Check Keyword Intent")
def check_keyword_intent(keyword: str) -> str:
    """
    Analyzes the search intent behind a keyword by classifying it into
    Informational, Navigational, Transactional, or Commercial categories.

    Args:
        keyword: The keyword or phrase to analyze

    Returns:
        A summary of the keyword intent and typical SERP characteristics
    """
    keyword_lower = keyword.lower().strip()

    # Intent classification based on keyword patterns
    transactional_signals = [
        "buy", "price", "cheap", "deal", "discount", "coupon", "order",
        "purchase", "shop", "store", "sale", "cost", "affordable"
    ]
    informational_signals = [
        "how to", "what is", "why", "guide", "tutorial", "tips",
        "learn", "examples", "definition", "meaning", "explained"
    ]
    navigational_signals = [
        "login", "sign in", "website", "official", "homepage", "contact"
    ]
    commercial_signals = [
        "best", "top", "review", "reviews", "vs", "versus", "comparison",
        "compare", "alternative", "alternatives"
    ]

    intent = "Informational"  # Default
    confidence = "Medium"
    serp_features = []

    # Check for transactional intent
    if any(signal in keyword_lower for signal in transactional_signals):
        intent = "Transactional"
        confidence = "High"
        serp_features = ["Shopping ads", "Product listings", "Price comparisons"]

    # Check for informational intent
    elif any(signal in keyword_lower for signal in informational_signals):
        intent = "Informational"
        confidence = "High"
        serp_features = ["Featured snippets", "Knowledge panels", "People Also Ask"]

    # Check for navigational intent
    elif any(signal in keyword_lower for signal in navigational_signals):
        intent = "Navigational"
        confidence = "High"
        serp_features = ["Site links", "Knowledge panel", "Direct answer"]

    # Check for commercial investigation
    elif any(signal in keyword_lower for signal in commercial_signals):
        intent = "Commercial Investigation"
        confidence = "High"
        serp_features = ["Review snippets", "Comparison tables", "Listicles"]

    return f"""
Keyword: "{keyword}"
Intent Classification: {intent}
Confidence: {confidence}
Expected SERP Features: {', '.join(serp_features) if serp_features else 'Standard organic results'}

Optimization Recommendations:
- For {intent} intent, focus on {'product pages with clear CTAs and pricing' if intent == 'Transactional' else 'comprehensive, educational content' if intent == 'Informational' else 'brand authority and trust signals' if intent == 'Navigational' else 'detailed comparisons and reviews'}
"""


@tool("Execute Python Code")
def execute_python(code: str) -> dict:
    """
    Execute a short Python snippet in a restricted subprocess (timeout 10s).

    Use this for quick diagnostics, e.g., checking HTTP headers for
    X-Robots-Tag via requests or curl.

    Args:
        code: Python source to execute.

    Returns:
        Dict with stdout, stderr, exit_code, timed_out flag, and optional error.
    """
    try:
        proc = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return {
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
            "exit_code": proc.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": (e.stdout or "")[-4000:],
            "stderr": (e.stderr or "")[-4000:],
            "exit_code": None,
            "timed_out": True,
            "error": "Execution timed out",
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timed_out": False,
            "error": f"Execution failed: {e}",
        }


# ============================================================================
# ANALYSIS UTILITY FUNCTIONS
# ============================================================================

def analyze_title_quality(title: str, h1_tags: List[str], url: str) -> dict:
    """Analyze the quality and intent alignment of a page title."""
    issues = []
    severity = "LOW"
    score = 100

    if not title:
        return {
            "score": 0,
            "issues": [{"issue": "Missing page title", "severity": "CRITICAL"}],
            "severity": "CRITICAL"
        }

    # Check length
    if len(title) > 60:
        issues.append({
            "issue": f"Title too long ({len(title)} chars, recommended < 60)",
            "severity": "MEDIUM"
        })
        score -= 15
        severity = "MEDIUM"
    elif len(title) < 30:
        issues.append({
            "issue": f"Title too short ({len(title)} chars, recommended 30-60)",
            "severity": "LOW"
        })
        score -= 5

    # Check for generic titles
    generic_titles = ["home", "welcome", "untitled", "page", "new page", "document"]
    if title.lower().strip() in generic_titles:
        issues.append({
            "issue": f"Generic title '{title}' provides no SEO value",
            "severity": "HIGH"
        })
        score -= 40
        severity = "HIGH"

    # Check H1 alignment
    if h1_tags:
        h1_text = h1_tags[0].lower()
        title_lower = title.lower()

        # Check if there's keyword overlap
        h1_words = set(h1_text.split())
        title_words = set(title_lower.split())
        overlap = h1_words.intersection(title_words)

        if len(overlap) < 2 and len(h1_words) > 2:
            issues.append({
                "issue": "Title and H1 have low keyword alignment",
                "severity": "MEDIUM"
            })
            score -= 20
            if severity == "LOW":
                severity = "MEDIUM"

    return {"score": score, "issues": issues, "severity": severity}


def analyze_meta_description(meta_desc: str, title: str) -> dict:
    """Analyze meta description quality."""
    issues = []
    severity = "LOW"
    score = 100

    if not meta_desc:
        return {
            "score": 0,
            "issues": [{"issue": "Missing meta description", "severity": "HIGH"}],
            "severity": "HIGH"
        }

    # Check length
    if len(meta_desc) > 160:
        issues.append({
            "issue": f"Meta description too long ({len(meta_desc)} chars, will be truncated)",
            "severity": "MEDIUM"
        })
        score -= 15
        severity = "MEDIUM"
    elif len(meta_desc) < 70:
        issues.append({
            "issue": f"Meta description too short ({len(meta_desc)} chars, opportunity missed)",
            "severity": "LOW"
        })
        score -= 10

    # Check for call-to-action
    cta_words = ["learn", "discover", "find", "get", "start", "try", "see", "read", "click"]
    has_cta = any(word in meta_desc.lower() for word in cta_words)
    if not has_cta:
        issues.append({
            "issue": "Meta description lacks a call-to-action",
            "severity": "LOW"
        })
        score -= 5

    return {"score": score, "issues": issues, "severity": severity}


def analyze_heading_structure(headings: dict) -> dict:
    """Analyze the heading hierarchy and structure."""
    issues = []
    severity = "LOW"
    score = 100

    h1_tags = headings.get("h1", [])
    h2_tags = headings.get("h2", [])
    h3_tags = headings.get("h3", [])

    # Check H1 presence
    if not h1_tags:
        issues.append({
            "issue": "Missing H1 tag - critical for SEO and accessibility",
            "severity": "HIGH"
        })
        score -= 30
        severity = "HIGH"
    elif len(h1_tags) > 1:
        issues.append({
            "issue": f"Multiple H1 tags found ({len(h1_tags)}) - should have exactly one",
            "severity": "MEDIUM"
        })
        score -= 15
        severity = "MEDIUM"

    # Check heading hierarchy
    if h3_tags and not h2_tags:
        issues.append({
            "issue": "H3 tags used without H2 tags - broken hierarchy",
            "severity": "MEDIUM"
        })
        score -= 15
        if severity == "LOW":
            severity = "MEDIUM"

    # Check for empty headings
    all_headings = h1_tags + h2_tags + h3_tags
    empty_headings = [h for h in all_headings if not h.strip()]
    if empty_headings:
        issues.append({
            "issue": f"Found {len(empty_headings)} empty heading tag(s)",
            "severity": "MEDIUM"
        })
        score -= 10

    return {"score": score, "issues": issues, "severity": severity}


def analyze_images(images: List[dict]) -> dict:
    """Analyze image alt text and SEO optimization."""
    issues = []
    severity = "LOW"
    score = 100

    if not images:
        return {"score": 100, "issues": [], "severity": "LOW"}

    missing_alt = [img for img in images if not img.get("alt", "").strip()]
    missing_alt_count = len(missing_alt)

    if missing_alt_count > 0:
        percentage = (missing_alt_count / len(images)) * 100
        if percentage > 50:
            severity = "HIGH"
            score -= 30
        elif percentage > 20:
            severity = "MEDIUM"
            score -= 15
        else:
            score -= 5

        issues.append({
            "issue": f"{missing_alt_count} of {len(images)} images missing alt text ({percentage:.0f}%)",
            "severity": severity,
            "affected_images": [img.get("src", "unknown") for img in missing_alt[:5]]
        })

    return {"score": score, "issues": issues, "severity": severity}


def analyze_links(links: dict) -> dict:
    """Analyze internal and external link structure."""
    issues = []
    severity = "LOW"
    score = 100

    internal_count = links.get("internal_count", 0)
    external_count = links.get("external_count", 0)
    malformed = links.get("malformed_links", [])

    # Check for orphan page (no internal links)
    if internal_count == 0:
        issues.append({
            "issue": "No internal links found - page may be orphaned",
            "severity": "MEDIUM"
        })
        score -= 20
        severity = "MEDIUM"

    # Check for malformed links
    if malformed:
        issues.append({
            "issue": f"Found {len(malformed)} malformed links",
            "severity": "LOW",
            "examples": malformed[:3]
        })
        score -= 5

    # Check link ratio
    total_links = internal_count + external_count
    if total_links > 100:
        issues.append({
            "issue": f"Excessive links on page ({total_links}) - may dilute link equity",
            "severity": "LOW"
        })
        score -= 5

    return {"score": score, "issues": issues, "severity": severity}


def analyze_canonical(canonical: str, final_url: str, is_redirect: bool) -> dict:
    """Analyze canonical tag and redirect status."""
    issues = []
    severity = "LOW"
    score = 100

    if not canonical:
        issues.append({
            "issue": "Missing canonical tag - risk of duplicate content issues",
            "severity": "MEDIUM"
        })
        score -= 20
        severity = "MEDIUM"
    elif canonical != final_url:
        # Check if it's pointing to a different page
        issues.append({
            "issue": f"Canonical URL differs from page URL",
            "severity": "LOW",
            "details": f"Canonical: {canonical}, Page: {final_url}"
        })
        score -= 5

    if is_redirect:
        issues.append({
            "issue": "Page accessed via redirect - consider updating internal links",
            "severity": "LOW"
        })
        score -= 5

    return {"score": score, "issues": issues, "severity": severity}


def infer_page_type(page_data: dict) -> str:
    """Infer the type of page based on content signals."""
    title = (page_data.get("title") or "").lower()
    h1_tags = page_data.get("headings", {}).get("h1", [])
    h1_text = " ".join(h1_tags).lower() if h1_tags else ""

    combined_text = f"{title} {h1_text}"

    # Product page signals
    if any(word in combined_text for word in ["buy", "price", "cart", "shop", "product", "$", "order"]):
        return "Product"

    # Article/Blog signals
    if any(word in combined_text for word in ["blog", "article", "post", "news", "guide", "how to"]):
        return "Article"

    # Recipe signals
    if any(word in combined_text for word in ["recipe", "ingredients", "cook", "bake", "minutes"]):
        return "Recipe"

    # FAQ signals
    if any(word in combined_text for word in ["faq", "questions", "answers", "help"]):
        return "FAQPage"

    # Contact page
    if any(word in combined_text for word in ["contact", "reach us", "get in touch"]):
        return "ContactPage"

    # About page
    if any(word in combined_text for word in ["about", "our story", "who we are", "our team"]):
        return "AboutPage"

    return "WebPage"


def generate_schema_suggestion(page_type: str, page_data: dict) -> Optional[dict]:
    """Generate appropriate JSON-LD schema based on page type."""
    url = page_data.get("url", "")
    title = page_data.get("title", "")
    description = page_data.get("meta_description", "")

    if page_type == "Product":
        return {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": title,
            "description": description,
            "url": url,
            "offers": {
                "@type": "Offer",
                "url": url,
                "priceCurrency": "USD",
                "price": "TODO: Add price",
                "availability": "https://schema.org/InStock"
            }
        }

    elif page_type == "Article":
        return {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": title,
            "description": description,
            "url": url,
            "author": {
                "@type": "Organization",
                "name": "TODO: Add author"
            },
            "datePublished": "TODO: Add date",
            "dateModified": "TODO: Add date"
        }

    elif page_type == "Recipe":
        return {
            "@context": "https://schema.org",
            "@type": "Recipe",
            "name": title,
            "description": description,
            "url": url,
            "recipeIngredient": ["TODO: Add ingredients"],
            "recipeInstructions": ["TODO: Add instructions"]
        }

    elif page_type == "FAQPage":
        return {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": "TODO: Add question",
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": "TODO: Add answer"
                    }
                }
            ]
        }

    return None


def generate_improved_title(current_title: str, h1_tags: List[str], page_type: str) -> str:
    """Generate an improved title based on H1 and page type."""
    if not h1_tags:
        return current_title

    h1 = h1_tags[0]

    # Clean up the H1 for use in title
    base_title = h1.strip()[:50]

    # Add page type specific suffixes
    suffixes = {
        "Product": " | Buy Online",
        "Article": " | Complete Guide",
        "Recipe": " | Easy Recipe",
        "FAQPage": " | FAQ & Answers",
        "ContactPage": " | Contact Us",
        "AboutPage": " | Learn More"
    }

    suffix = suffixes.get(page_type, "")

    new_title = f"{base_title}{suffix}"

    # Ensure it's under 60 characters
    if len(new_title) > 60:
        new_title = new_title[:57] + "..."

    return new_title


# ============================================================================
# PAGE ANALYZER
# ============================================================================

class PageAnalyzer:
    """Analyzes a single page's SEO data and generates remediation tasks."""

    def __init__(self, page_data: dict):
        self.page_data = page_data
        self.url = page_data.get("url", "")
        self.audit_id = f"audit_{uuid.uuid4().hex[:8]}"
        self.tasks = []
        self.overall_score = 100

    def analyze(self) -> dict:
        """Perform complete SEO analysis on the page."""
        pd = self.page_data.get("page_data", {})

        # Extract data
        title = pd.get("title")
        meta_desc = pd.get("meta_description")
        canonical = pd.get("canonical")
        headings = pd.get("headings", {})
        images = pd.get("images", [])
        links = self.page_data.get("links", {})
        is_redirect = self.page_data.get("is_redirect", False)
        final_url = self.page_data.get("final_url", self.url)

        # Infer page type
        page_type = infer_page_type(pd)
        h1_tags = headings.get("h1", [])

        # Analyze each aspect
        title_analysis = analyze_title_quality(title, h1_tags, self.url)
        meta_analysis = analyze_meta_description(meta_desc, title)
        heading_analysis = analyze_heading_structure(headings)
        image_analysis = analyze_images(images)
        link_analysis = analyze_links(links)
        canonical_analysis = analyze_canonical(canonical, final_url, is_redirect)

        # Generate tasks from title issues
        for issue in title_analysis["issues"]:
            task = {
                "category": "METADATA",
                "issue": issue["issue"],
                "severity": issue["severity"],
                "reasoning": f"Title tag analysis for {self.url}",
                "remediation": {
                    "action": "UPDATE_TAG",
                    "target": "title",
                    "current_value": title
                }
            }
            if issue["severity"] in ["HIGH", "CRITICAL"] and h1_tags:
                task["remediation"]["proposed_value"] = generate_improved_title(
                    title or "", h1_tags, page_type
                )
            self.tasks.append(task)

        # Generate tasks from meta description issues
        for issue in meta_analysis["issues"]:
            self.tasks.append({
                "category": "METADATA",
                "issue": issue["issue"],
                "severity": issue["severity"],
                "reasoning": f"Meta description analysis for {self.url}",
                "remediation": {
                    "action": "UPDATE_TAG",
                    "target": "meta[name='description']",
                    "current_value": meta_desc
                }
            })

        # Generate tasks from heading issues
        for issue in heading_analysis["issues"]:
            self.tasks.append({
                "category": "CONTENT_STRUCTURE",
                "issue": issue["issue"],
                "severity": issue["severity"],
                "reasoning": "Heading hierarchy analysis",
                "remediation": {
                    "action": "RESTRUCTURE_HEADINGS",
                    "target": "headings",
                    "current_structure": headings
                }
            })

        # Generate tasks from image issues
        for issue in image_analysis["issues"]:
            self.tasks.append({
                "category": "ACCESSIBILITY",
                "issue": issue["issue"],
                "severity": issue["severity"],
                "reasoning": "Image accessibility and SEO analysis",
                "remediation": {
                    "action": "ADD_ALT_TEXT",
                    "target": "img",
                    "affected_images": issue.get("affected_images", [])
                }
            })

        # Generate tasks from link issues
        for issue in link_analysis["issues"]:
            self.tasks.append({
                "category": "LINKS",
                "issue": issue["issue"],
                "severity": issue["severity"],
                "reasoning": "Internal linking and link health analysis",
                "remediation": {
                    "action": "REVIEW_LINKS",
                    "target": "a",
                    "examples": issue.get("examples", [])
                }
            })

        # Generate tasks from canonical issues
        for issue in canonical_analysis["issues"]:
            self.tasks.append({
                "category": "TECHNICAL",
                "issue": issue["issue"],
                "severity": issue["severity"],
                "reasoning": "Canonical tag and redirect analysis",
                "remediation": {
                    "action": "UPDATE_CANONICAL",
                    "target": "link[rel='canonical']",
                    "details": issue.get("details", "")
                }
            })

        # Check for structured data opportunity
        schema_suggestion = generate_schema_suggestion(page_type, pd)
        if schema_suggestion and page_type != "WebPage":
            self.tasks.append({
                "category": "STRUCTURED_DATA",
                "issue": f"Missing {page_type} schema markup",
                "severity": "MEDIUM",
                "reasoning": f"Page appears to be a {page_type} based on content analysis but lacks structured data",
                "remediation": {
                    "action": "INJECT_CODE",
                    "target": "head",
                    "proposed_value": json.dumps(schema_suggestion, indent=2)
                }
            })

        # Calculate overall score
        all_scores = [
            title_analysis["score"],
            meta_analysis["score"],
            heading_analysis["score"],
            image_analysis["score"],
            link_analysis["score"],
            canonical_analysis["score"]
        ]
        self.overall_score = sum(all_scores) // len(all_scores)

        # Determine intent
        intent = self._determine_intent(title, h1_tags)

        return {
            "audit_id": self.audit_id,
            "analyzed_url": self.url,
            "page_type": page_type,
            "intent_assessment": intent,
            "overall_score": self.overall_score,
            "component_scores": {
                "title": title_analysis["score"],
                "meta_description": meta_analysis["score"],
                "headings": heading_analysis["score"],
                "images": image_analysis["score"],
                "links": link_analysis["score"],
                "technical": canonical_analysis["score"]
            },
            "tasks": sorted(self.tasks, key=lambda x: {
                "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3
            }.get(x["severity"], 4))
        }

    def _determine_intent(self, title: str, h1_tags: List[str]) -> str:
        """Determine the search intent of the page."""
        text = f"{title or ''} {' '.join(h1_tags)}".lower()

        if any(word in text for word in ["buy", "price", "shop", "order", "cart"]):
            return "Transactional (User wants to buy)"
        elif any(word in text for word in ["how", "what", "why", "guide", "learn", "tutorial"]):
            return "Informational (User wants to learn)"
        elif any(word in text for word in ["best", "top", "review", "compare", "vs"]):
            return "Commercial Investigation (User is researching)"
        elif any(word in text for word in ["login", "contact", "about", "home"]):
            return "Navigational (User seeking specific page)"
        else:
            return "Mixed/Unclear Intent"


# ============================================================================
# SITE ANALYZER (Analyzes full crawl data)
# ============================================================================

class SiteAnalyzer:
    """Analyzes complete site crawl data and generates comprehensive audit."""

    def __init__(self, crawl_data: dict):
        self.crawl_data = crawl_data
        self.audit_id = f"site_audit_{uuid.uuid4().hex[:8]}"
        self.page_audits = []
        self.site_wide_issues = []

    def analyze(self) -> dict:
        """Perform complete site-wide SEO analysis."""
        pages = self.crawl_data.get("pages", [])

        # Analyze each page
        for page in pages:
            analyzer = PageAnalyzer(page)
            audit = analyzer.analyze()
            self.page_audits.append(audit)

        # Identify site-wide patterns
        self._analyze_site_patterns()

        # Calculate site-wide score
        if self.page_audits:
            avg_score = sum(a["overall_score"] for a in self.page_audits) // len(self.page_audits)
        else:
            avg_score = 0

        # Aggregate all tasks by severity
        all_tasks = []
        for audit in self.page_audits:
            for task in audit["tasks"]:
                task["page_url"] = audit["analyzed_url"]
                all_tasks.append(task)

        severity_counts = {
            "CRITICAL": len([t for t in all_tasks if t["severity"] == "CRITICAL"]),
            "HIGH": len([t for t in all_tasks if t["severity"] == "HIGH"]),
            "MEDIUM": len([t for t in all_tasks if t["severity"] == "MEDIUM"]),
            "LOW": len([t for t in all_tasks if t["severity"] == "LOW"])
        }

        return {
            "audit_id": self.audit_id,
            "crawl_id": self.crawl_data.get("scan_id"),
            "base_domain": self.crawl_data.get("base_domain"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "pages_analyzed": len(self.page_audits),
                "overall_score": avg_score,
                "total_issues": len(all_tasks),
                "issues_by_severity": severity_counts
            },
            "site_wide_issues": self.site_wide_issues,
            "page_audits": self.page_audits,
            "prioritized_tasks": sorted(all_tasks, key=lambda x: {
                "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3
            }.get(x["severity"], 4))[:50]  # Top 50 most important tasks
        }

    def _analyze_site_patterns(self):
        """Identify patterns across all pages."""
        if not self.page_audits:
            return

        # Check for duplicate titles
        titles = [a.get("analyzed_url") for a in self.page_audits]
        # (This is a simplified check - would need page_data access for real title comparison)

        # Check for pages without H1
        pages_without_h1 = [
            a["analyzed_url"] for a in self.page_audits
            if a["component_scores"]["headings"] < 70
        ]
        if len(pages_without_h1) > 2:
            self.site_wide_issues.append({
                "category": "CONTENT_STRUCTURE",
                "issue": f"{len(pages_without_h1)} pages have heading structure issues",
                "severity": "HIGH",
                "affected_pages": pages_without_h1[:10]
            })

        # Check for low-scoring pages
        low_score_pages = [
            a["analyzed_url"] for a in self.page_audits
            if a["overall_score"] < 50
        ]
        if low_score_pages:
            self.site_wide_issues.append({
                "category": "OVERALL_QUALITY",
                "issue": f"{len(low_score_pages)} pages have low SEO scores (< 50)",
                "severity": "HIGH",
                "affected_pages": low_score_pages[:10]
            })


# ============================================================================
# CREWAI AGENT DEFINITION
# ============================================================================

SEO_ANALYST_SYSTEM_PROMPT = """
#Technical SEO Audit Specialist

You are an expert Technical SEO Auditor with deep expertise in website optimization for search engines and AI systems. 
Your mission is to conduct comprehensive technical SEO audits and deliver actionable improvement reports that drive measurable results. 
You will be given a JSON containing the crawl data of a website.

Your Core Responsibilities

- Actionable Reporting: Deliver structured, prioritized reports with specific recommendations

- Best Practice Guidance: Provide expert advice based on current SEO standards and search engine guidelines

- Performance Optimization: Focus on improvements that enhance both search visibility and user experience

##Technical SEO Foundation

- Technical SEO ensures websites are crawlable, indexable, and understandable by search engines and AI systems. Your expertise covers these critical areas:

- Site Structure: Use a flat, logical structure with clear navigation and internal linking.

- Canonicalization: Prevent duplicate content, especially for paginated and filtered pages.

- Schema Markup: Use rich, appropriate schema (Product, ItemList, Organization, BreadcrumbList, FAQPage, etc.) to help search engines  understand your content and relationships.

- Meta Tags & Headings: Every page needs unique, descriptive meta tags and a logical heading structure.

- Pagination: Each paginated page should be indexable.

- Internal Linking: Ensure all important pages are linked and avoid orphaned content.

- Sitemaps & Robots.txt: Keep sitemaps up to date and robots.txt permissive for important content.

##Audit Methodology

###A. Indexability Checks

Check robots.txt for disallow rules that might block important pages.
Make sure the robots.txt is accessible and not blocked.

B. Canonicalization & Duplicate Content

Check for canonical tags on all pages, especially for:

Agent listings (should self-canonicalize)

Category/search/paginated pages (should canonicalize to the main version or themselves if unique)

Test multiple URL variants (e.g., with/without trailing slash, with/without query params, paginated URLs) to ensure canonicalization is consistent and correct.

Check for duplicate content across agent listings, categories, and paginated pages.

D. Schema Markup

Analyze website pages for opportunities for comprehensive, accurate, and page-specific schema markup implementation to improve search engine understanding,

E. Pagination

Paginated pages should use:

Canonical tags pointing to the main or current page as appropriate.

Check that paginated pages are not causing duplicate content issues.

F. Internal Linking

Ensure all agent listings are linked from at least one category or search page.

Ensure builder pages link to their agents and vice versa.

Avoid orphaned pages.

G. Meta Tags & Headings

Every page should have a unique, descriptive <title> and <meta name="description">.

Use clear, hierarchical heading structure (H1, H2, H3, etc.).

H. Sitemaps & Robots.txt

Ensure all important pages are included in the XML sitemap.

Robots.txt should not block important pages.

"""


def create_seo_analyst_agent() -> Agent:
    """Creates and returns the SEO Analyst agent."""
    return Agent(
        role="SEO Analyst",
        goal="Analyze crawl data to identify SEO issues and generate prioritized remediation tasks",
        backstory="""You are SEO_Analyst, the "Brain" of an SEO analysis system. You are a Senior
        SEO Strategist who evaluates context, relevance, and user intent - not just tag presence.
        You transform raw technical data into actionable remediation tasks that developers can
        implement to improve search performance.""",
        tools=[validate_json_ld, check_keyword_intent, execute_python],
        verbose=True,
        allow_delegation=False,
        system_prompt=SEO_ANALYST_SYSTEM_PROMPT
    )


def create_analysis_task(agent: Agent, crawl_data: dict) -> Task:
    """Creates an analysis task for the crawl data."""
    return Task(
        description=f"""
        Analyze the following crawl data and produce a comprehensive SEO audit:

        Crawl Summary:
        - Pages crawled: {crawl_data.get('summary', {}).get('pages_crawled', 0)}
        - Domain: {crawl_data.get('base_domain', 'unknown')}


        Give a site wide analysis and recommendations.

        For each page, analyze:
        1. Title and meta description quality and intent alignment
        2. Heading structure and hierarchy
        3. Image alt text coverage
        4. Internal linking patterns
        5. Canonical tag and redirect status
        6. Structured data opportunities

        Generate specific, actionable remediation tasks with severity ratings.
        Do NOT give generic advice - be specific about what to change.
        """,
        expected_output="""A comprehensive JSON audit containing:
        - Overall site score and summary statistics
        - Site-wide pattern issues
        - Per-page audit results with component scores
        - Prioritized list of remediation tasks with severity ratings""",
        agent=agent
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def analyze_crawl_data(crawl_data: dict) -> dict:
    """
    Main function to analyze crawl data and return SEO audit.

    Args:
        crawl_data: The crawl data JSON from SEO_Scout

    Returns:
        Dictionary containing comprehensive SEO audit results
    """
    analyzer = SiteAnalyzer(crawl_data)
    return analyzer.analyze()


def analyze_single_page(page_data: dict) -> dict:
    """
    Analyze a single page's SEO data.

    Args:
        page_data: Single page data from crawl

    Returns:
        Dictionary containing page audit results
    """
    analyzer = PageAnalyzer(page_data)
    return analyzer.analyze()


def print_detailed_report(result: dict):
    """Print a detailed, formatted report of the analysis results."""
    summary = result.get("summary", {})

    # Header
    print("\n" + "=" * 80)
    print("                        SEO ANALYSIS REPORT")
    print("=" * 80)

    # Summary Section
    print(f"\n{'─' * 80}")
    print("  SUMMARY")
    print(f"{'─' * 80}")
    print(f"  Audit ID:        {result.get('audit_id', 'N/A')}")
    print(f"  Domain:          {result.get('base_domain', 'N/A')}")
    print(f"  Pages Analyzed:  {summary.get('pages_analyzed', 0)}")
    print(f"  Overall Score:   {summary.get('overall_score', 0)}/100")
    print(f"  Total Issues:    {summary.get('total_issues', 0)}")

    # Severity Breakdown
    print(f"\n  Issues by Severity:")
    severity_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    for sev, count in summary.get("issues_by_severity", {}).items():
        icon = severity_icons.get(sev, "⚪")
        print(f"    {icon} {sev}: {count}")

    # Site-Wide Issues
    site_issues = result.get("site_wide_issues", [])
    if site_issues:
        print(f"\n{'─' * 80}")
        print("  SITE-WIDE ISSUES")
        print(f"{'─' * 80}")
        for i, issue in enumerate(site_issues, 1):
            print(f"\n  [{issue.get('severity', 'N/A')}] {issue.get('issue', 'N/A')}")
            print(f"  Category: {issue.get('category', 'N/A')}")
            affected = issue.get('affected_pages', [])
            if affected:
                print(f"  Affected Pages ({len(affected)}):")
                for page in affected[:5]:
                    print(f"    - {page}")
                if len(affected) > 5:
                    print(f"    ... and {len(affected) - 5} more")

    # Detailed Issues by Severity
    print(f"\n{'─' * 80}")
    print("  ISSUES BY SEVERITY")
    print(f"{'─' * 80}")

    tasks = result.get("prioritized_tasks", [])

    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        severity_tasks = [t for t in tasks if t.get("severity") == severity]
        if severity_tasks:
            icon = severity_icons.get(severity, "⚪")
            print(f"\n  {icon} {severity} ({len(severity_tasks)} issues)")
            print(f"  {'─' * 40}")

            for i, task in enumerate(severity_tasks[:10], 1):  # Show top 10 per severity
                print(f"\n  {i}. [{task.get('category', 'N/A')}] {task.get('issue', 'N/A')}")
                print(f"     Page: {task.get('page_url', 'N/A')}")
                print(f"     Reasoning: {task.get('reasoning', 'N/A')}")

            if len(severity_tasks) > 10:
                print(f"\n  ... and {len(severity_tasks) - 10} more {severity} issues")

    # Structured Tasks (Remediation)
    print(f"\n{'─' * 80}")
    print("  STRUCTURED REMEDIATION TASKS")
    print(f"{'─' * 80}")

    # Group tasks by category
    categories = {}
    for task in tasks:
        cat = task.get("category", "OTHER")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(task)

    task_num = 1
    for category, cat_tasks in categories.items():
        print(f"\n  📁 {category} ({len(cat_tasks)} tasks)")
        print(f"  {'─' * 40}")

        for task in cat_tasks[:5]:  # Show top 5 per category
            severity = task.get("severity", "N/A")
            icon = severity_icons.get(severity, "⚪")

            print(f"\n  TASK-{task_num:03d} {icon} [{severity}]")
            print(f"  ├── Issue: {task.get('issue', 'N/A')}")
            print(f"  ├── Page: {task.get('page_url', 'N/A')}")

            remediation = task.get("remediation", {})
            action = remediation.get("action", "N/A")
            target = remediation.get("target", "N/A")

            print(f"  ├── Action: {action}")
            print(f"  ├── Target: {target}")

            if remediation.get("current_value"):
                current = str(remediation.get("current_value", ""))[:60]
                print(f"  ├── Current: {current}{'...' if len(str(remediation.get('current_value', ''))) > 60 else ''}")

            if remediation.get("proposed_value"):
                proposed = str(remediation.get("proposed_value", ""))
                if len(proposed) > 100:
                    print(f"  └── Proposed: [JSON-LD Schema - see full report]")
                else:
                    print(f"  └── Proposed: {proposed}")
            else:
                print(f"  └── Proposed: [Requires manual review]")

            task_num += 1

        if len(cat_tasks) > 5:
            print(f"\n  ... and {len(cat_tasks) - 5} more {category} tasks")

    # Page-by-Page Scores
    print(f"\n{'─' * 80}")
    print("  PAGE SCORES")
    print(f"{'─' * 80}")

    page_audits = result.get("page_audits", [])
    sorted_pages = sorted(page_audits, key=lambda x: x.get("overall_score", 0))

    print(f"\n  {'URL':<50} {'Score':>6} {'Issues':>7}")
    print(f"  {'─' * 50} {'─' * 6} {'─' * 7}")

    for audit in sorted_pages[:15]:  # Show top 15 pages
        url = audit.get("analyzed_url", "N/A")
        if len(url) > 48:
            url = url[:45] + "..."
        score = audit.get("overall_score", 0)
        issues = len(audit.get("tasks", []))

        # Color code by score
        if score >= 80:
            score_str = f"{score:>6}"
        elif score >= 60:
            score_str = f"{score:>6}"
        else:
            score_str = f"{score:>6}"

        print(f"  {url:<50} {score_str} {issues:>7}")

    if len(page_audits) > 15:
        print(f"\n  ... and {len(page_audits) - 15} more pages")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python seo_analyst_agent.py <crawl_result.json>")
        print("Example: python seo_analyst_agent.py crawl_result_abc123.json")
        sys.exit(1)

    input_file = sys.argv[1]

    print(f"Loading crawl data from: {input_file}")

    try:
        with open(input_file, 'r') as f:
            crawl_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        sys.exit(1)

    print(f"Analyzing {crawl_data.get('summary', {}).get('pages_crawled', 0)} pages...")

    result = analyze_crawl_data(crawl_data)

    # Print detailed report
    print_detailed_report(result)

    # Save full results
    output_file = f"audit_result_{result['audit_id']}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nFull results saved to: {output_file}")

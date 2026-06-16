import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Incident Insight Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are an Incident Insight Agent. Your job is to analyze issues and incidents from Jira and related documentation from Confluence, then produce a structured root cause analysis with actionable solutions.

## Your Capabilities
- Aggregate and categorize incidents by theme, component, and severity
- Identify patterns and recurring issues across tickets
- Perform root cause analysis based on evidence from tickets and documentation
- Propose concrete solutions with references to source tickets

## Rules
- NEVER fabricate ticket IDs or links. Only reference tickets that exist in the data provided.
- ALWAYS cite source tickets (key + summary) when making claims.
- If there is insufficient data to determine a root cause, say so explicitly — do NOT guess.
- Group related incidents together to identify patterns.
- Prioritize by severity and frequency.

## Output Format
Structure your response as:

### 📊 Summary
- Total incidents found
- Time period analyzed
- Top affected components/projects

### 🔍 Incident Categories
Group incidents by theme/component. For each group:
- Number of incidents
- List of ticket keys with summaries
- Common patterns observed

### 🔬 Root Cause Analysis
For each identified root cause:
- Description of the root cause
- Supporting evidence (ticket references)
- Confidence level (High/Medium/Low)
- Affected components

### 💡 Proposed Solutions
For each root cause:
- Recommended actions (short-term and long-term)
- Priority (Critical/High/Medium/Low)
- Expected impact

### 📚 Related Documentation
List any relevant Confluence pages found, with titles and links.
"""


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    metadata: dict = {}


def get_llm_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("AI_PLATFORM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1")
    model = os.getenv("LLM_MODEL", "google/gemma-4-31b-it")
    if not api_key:
        raise RuntimeError("AI_PLATFORM_API_KEY is not set")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def parse_user_intent(message: str) -> dict:
    """Use LLM to extract time range and filters from user message."""
    client, model = get_llm_client()
    today = datetime.now().strftime("%Y-%m-%d")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""Today is {today}. Extract the search parameters from the user's message.
Return a JSON object with:
- "days_back": integer, how many days to look back (default 7)
- "jira_project": string or null, Jira project key if mentioned
- "component": string or null, component/system name if mentioned
- "severity": string or null, severity/priority if mentioned
- "keywords": list of strings, any specific keywords to search for
- "confluence_query": string, a search query for Confluence docs related to the incidents

Return ONLY valid JSON, no markdown fences.""",
            },
            {"role": "user", "content": message},
        ],
        max_tokens=500,
        temperature=0,
    )

    text = response.choices[0].message.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"days_back": 7, "jira_project": None, "component": None, "severity": None, "keywords": [], "confluence_query": "incident post-mortem"}


def build_jql(params: dict) -> str:
    days = params.get("days_back", 7)
    clauses = [f"created >= -{days}d"]

    project = params.get("jira_project")
    if project:
        clauses.append(f'project = "{project}"')

    component = params.get("component")
    if component:
        clauses.append(f'component = "{component}"')

    severity = params.get("severity")
    if severity:
        clauses.append(f'priority = "{severity}"')

    keywords = params.get("keywords", [])
    if keywords:
        keyword_str = " OR ".join(f'text ~ "{k}"' for k in keywords)
        clauses.append(f"({keyword_str})")

    return " AND ".join(clauses) + " ORDER BY created DESC"


def query_jira(jql: str, max_results: int = 100) -> list[dict]:
    base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
    token = os.getenv("JIRA_PAT", "")
    username = os.getenv("JIRA_USERNAME", "")
    password = os.getenv("JIRA_PASSWORD", "")

    if not base_url:
        logger.warning("JIRA_BASE_URL not set, skipping Jira query")
        return []

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username and password:
        import base64
        creds = base64.b64encode(f"{username}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {creds}"
    else:
        logger.warning("No Jira credentials configured")
        return []

    url = f"{base_url}/rest/api/2/search"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": "summary,status,priority,components,created,updated,description,comment,labels,assignee,reporter,issuetype",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30, verify=os.getenv("JIRA_VERIFY_SSL", "true").lower() == "true")
        resp.raise_for_status()
        data = resp.json()
        issues = data.get("issues", [])
        logger.info(f"Jira returned {len(issues)} issues for JQL: {jql}")
        return issues
    except requests.RequestException as e:
        logger.error(f"Jira query failed: {e}")
        return []


def format_jira_issues(issues: list[dict]) -> str:
    if not issues:
        return "No Jira issues found for the given criteria."

    lines = [f"## Jira Issues ({len(issues)} found)\n"]
    for issue in issues:
        key = issue.get("key", "???")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "No summary")
        status = fields.get("status", {}).get("name", "Unknown")
        priority = fields.get("priority", {}).get("name", "Unknown") if fields.get("priority") else "None"
        issue_type = fields.get("issuetype", {}).get("name", "Unknown")
        created = fields.get("created", "")[:10]
        components = ", ".join(c.get("name", "") for c in fields.get("components", []))
        labels = ", ".join(fields.get("labels", []))
        assignee = fields.get("assignee", {})
        assignee_name = assignee.get("displayName", "Unassigned") if assignee else "Unassigned"
        description = fields.get("description", "") or ""
        if len(description) > 500:
            description = description[:500] + "..."

        comments = fields.get("comment", {}).get("comments", [])
        comment_text = ""
        if comments:
            last_comments = comments[-3:]
            comment_text = "\n".join(
                f"  - {c.get('author', {}).get('displayName', '?')} ({c.get('created', '')[:10]}): {c.get('body', '')[:200]}"
                for c in last_comments
            )

        lines.append(f"### {key}: {summary}")
        lines.append(f"- **Type**: {issue_type} | **Status**: {status} | **Priority**: {priority}")
        lines.append(f"- **Created**: {created} | **Assignee**: {assignee_name}")
        if components:
            lines.append(f"- **Components**: {components}")
        if labels:
            lines.append(f"- **Labels**: {labels}")
        if description:
            lines.append(f"- **Description**: {description}")
        if comment_text:
            lines.append(f"- **Recent comments**:\n{comment_text}")
        lines.append("")

    return "\n".join(lines)


def query_confluence(query: str, max_results: int = 20) -> list[dict]:
    base_url = os.getenv("CONFLUENCE_BASE_URL", "").rstrip("/")
    token = os.getenv("CONFLUENCE_PAT", os.getenv("JIRA_PAT", ""))
    username = os.getenv("CONFLUENCE_USERNAME", os.getenv("JIRA_USERNAME", ""))
    password = os.getenv("CONFLUENCE_PASSWORD", os.getenv("JIRA_PASSWORD", ""))

    if not base_url:
        logger.warning("CONFLUENCE_BASE_URL not set, skipping Confluence query")
        return []

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username and password:
        import base64
        creds = base64.b64encode(f"{username}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {creds}"
    else:
        logger.warning("No Confluence credentials configured")
        return []

    url = f"{base_url}/rest/api/search"
    params = {
        "cql": query,
        "limit": max_results,
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30, verify=os.getenv("CONFLUENCE_VERIFY_SSL", "true").lower() == "true")
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        logger.info(f"Confluence returned {len(results)} pages for query: {query}")
        return results
    except requests.RequestException as e:
        logger.error(f"Confluence query failed: {e}")
        return []


def format_confluence_pages(pages: list[dict]) -> str:
    if not pages:
        return "No related Confluence documentation found."

    lines = [f"## Related Confluence Documentation ({len(pages)} found)\n"]
    base_url = os.getenv("CONFLUENCE_BASE_URL", "").rstrip("/")

    for result in pages:
        title = result.get("title", "Untitled")
        excerpt = result.get("excerpt", "")
        url_path = result.get("url", "")
        last_modified = result.get("friendlyLastModified", "")
        container = result.get("resultGlobalContainer", {})
        space_name = container.get("title", "Unknown space")

        content = result.get("content", {})
        page_id = content.get("id", "")

        link = f"{base_url}{url_path}" if base_url and url_path else ""

        if len(excerpt) > 800:
            excerpt = excerpt[:800] + "..."

        lines.append(f"### {title}")
        lines.append(f"- **Space**: {space_name} | **Last updated**: {last_modified}")
        if link:
            lines.append(f"- **Link**: {link}")
        if excerpt:
            lines.append(f"- **Excerpt**: {excerpt}")
        lines.append("")

    return "\n".join(lines)


def analyze_with_llm(jira_data: str, confluence_data: str, user_message: str) -> str:
    client, model = get_llm_client()

    analysis_prompt = f"""The user asked: "{user_message}"

Here is the data collected from Jira and Confluence:

{jira_data}

---

{confluence_data}

---

Based on the above data, provide a comprehensive incident analysis following your output format guidelines. Focus on identifying patterns, root causes, and actionable solutions. Always reference specific ticket keys when making claims."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": analysis_prompt},
        ],
        max_tokens=4000,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        params = parse_user_intent(req.message)
        logger.info(f"Parsed intent: {params}")

        jql = build_jql(params)
        logger.info(f"JQL: {jql}")

        jira_issues = query_jira(jql)
        jira_data = format_jira_issues(jira_issues)

        days = params.get("days_back", 7)
        cq = params.get("confluence_query", "incident")
        confluence_cql = f'type = "page" AND text ~ "{cq}" ORDER BY lastModified DESC'

        confluence_pages = query_confluence(confluence_cql)
        confluence_data = format_confluence_pages(confluence_pages)

        analysis = analyze_with_llm(jira_data, confluence_data, req.message)

        metadata = {
            "jira_issues_count": len(jira_issues),
            "confluence_pages_count": len(confluence_pages),
            "jql_used": jql,
            "time_range_days": params.get("days_back", 7),
        }

        return ChatResponse(response=analysis, metadata=metadata)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during chat")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/invocations")
async def invocations(req: ChatRequest):
    """SDK-convention endpoint — same logic as /chat."""
    return await chat(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

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


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Incident Insight Agent</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;flex-direction:column}
.header{background:linear-gradient(135deg,#1e293b,#334155);padding:20px 32px;border-bottom:1px solid #334155;display:flex;align-items:center;gap:16px}
.header h1{font-size:24px;font-weight:700;color:#f8fafc}
.header .badge{background:#22d3ee;color:#0f172a;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:600}
.main{flex:1;display:flex;flex-direction:column;max-width:960px;width:100%;margin:0 auto;padding:24px}
.messages{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:16px;padding-bottom:24px}
.msg{padding:16px 20px;border-radius:12px;max-width:100%;line-height:1.7;font-size:14px;white-space:pre-wrap;word-wrap:break-word}
.msg.user{background:#1e40af;color:#e0f2fe;align-self:flex-end;max-width:70%;border-bottom-right-radius:4px}
.msg.bot{background:#1e293b;border:1px solid #334155;align-self:flex-start;border-bottom-left-radius:4px}
.msg.bot h3{color:#22d3ee;margin:12px 0 6px;font-size:15px}
.msg.bot h4{color:#94a3b8;margin:8px 0 4px;font-size:13px;text-transform:uppercase;letter-spacing:.5px}
.msg.bot strong{color:#f8fafc}
.msg.bot ul,.msg.bot ol{margin-left:20px;margin-top:4px}
.msg.bot table{border-collapse:collapse;margin:8px 0;width:100%}
.msg.bot th,.msg.bot td{border:1px solid #475569;padding:6px 10px;text-align:left;font-size:13px}
.msg.bot th{background:#334155;color:#cbd5e1}
.msg.bot a{color:#38bdf8;text-decoration:none}
.msg.bot a:hover{text-decoration:underline}
.meta{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}
.meta span{background:#334155;padding:4px 10px;border-radius:6px;font-size:11px;color:#94a3b8}
.input-area{background:#1e293b;border-top:1px solid #334155;padding:16px 24px}
.input-wrap{max-width:960px;margin:0 auto;display:flex;gap:12px}
.input-wrap input{flex:1;background:#0f172a;border:1px solid #475569;border-radius:10px;padding:12px 16px;color:#f8fafc;font-size:15px;outline:none;transition:border-color .2s}
.input-wrap input:focus{border-color:#22d3ee}
.input-wrap button{background:#22d3ee;color:#0f172a;border:none;border-radius:10px;padding:12px 24px;font-size:15px;font-weight:600;cursor:pointer;transition:background .2s}
.input-wrap button:hover{background:#06b6d4}
.input-wrap button:disabled{background:#475569;cursor:not-allowed}
.loading{display:inline-block;width:20px;height:20px;border:3px solid #475569;border-top-color:#22d3ee;border-radius:50%;animation:spin .8s linear infinite;margin:8px 0}
@keyframes spin{to{transform:rotate(360deg)}}
.suggestions{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.suggestions button{background:#1e293b;border:1px solid #475569;color:#94a3b8;padding:8px 14px;border-radius:8px;font-size:13px;cursor:pointer;transition:all .2s}
.suggestions button:hover{border-color:#22d3ee;color:#22d3ee}
</style>
</head>
<body>
<div class="header">
<h1>Incident Insight Agent</h1>
<span class="badge">Jira + Confluence + AI</span>
</div>
<div class="main">
<div class="messages" id="messages">
<div class="msg bot">Welcome! I analyze incidents from Jira and Confluence to identify patterns, root causes, and solutions. Try one of the suggestions below or type your own query.</div>
<div class="suggestions" id="suggestions">
<button onclick="ask(this.textContent)">Summarize all incidents from the last 7 days</button>
<button onclick="ask(this.textContent)">Analyze payment-related issues in the last 14 days</button>
<button onclick="ask(this.textContent)">What are the top critical bugs this week?</button>
</div>
</div>
</div>
<div class="input-area">
<div class="input-wrap">
<input type="text" id="input" placeholder="Ask about incidents... (e.g., 'Summarize incidents in the last 7 days')" onkeydown="if(event.key==='Enter')send()">
<button id="btn" onclick="send()">Analyze</button>
</div>
</div>
<script>
function md(s){
  s=s.replace(/^### (.*$)/gm,'<h3>$1</h3>');
  s=s.replace(/^## (.*$)/gm,'<h3>$1</h3>');
  s=s.replace(/^#### (.*$)/gm,'<h4>$1</h4>');
  s=s.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>');
  s=s.replace(/\\*(.+?)\\*/g,'<em>$1</em>');
  s=s.replace(/`([^`]+)`/g,'<code style="background:#334155;padding:2px 6px;border-radius:4px;font-size:13px">$1</code>');
  s=s.replace(/^[-*] (.+)/gm,'<li>$1</li>');
  s=s.replace(/(<li>.*<\\/li>)/s,function(m){return '<ul>'+m+'</ul>'});
  s=s.replace(/\\n---\\n/g,'<hr style="border:none;border-top:1px solid #475569;margin:12px 0">');
  s=s.replace(/\\|(.+)\\|/g,function(m){
    var cells=m.split('|').filter(c=>c.trim());
    if(cells.every(c=>/^[-:\\s]+$/.test(c)))return '';
    var tag=cells.some(c=>/\\*\\*/.test(c))?'th':'td';
    return '<tr>'+cells.map(c=>'<'+tag+'>'+c.trim()+'</'+tag+'>').join('')+'</tr>';
  });
  s=s.replace(/(<tr>.*<\\/tr>)/s,function(m){return '<table>'+m+'</table>'});
  return s;
}
function ask(text){document.getElementById('input').value=text;send();}
function send(){
  var input=document.getElementById('input'),btn=document.getElementById('btn'),msgs=document.getElementById('messages');
  var q=input.value.trim();if(!q)return;
  var sug=document.getElementById('suggestions');if(sug)sug.remove();
  msgs.innerHTML+='<div class="msg user">'+q.replace(/</g,'&lt;')+'</div>';
  msgs.innerHTML+='<div class="msg bot" id="loading"><div class="loading"></div> Analyzing incidents from Jira & Confluence... This may take 30-60 seconds.</div>';
  input.value='';btn.disabled=true;input.disabled=true;
  msgs.scrollTop=msgs.scrollHeight;
  fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q})})
  .then(r=>r.json()).then(d=>{
    document.getElementById('loading').remove();
    var html='<div class="msg bot">'+md(d.response||d.detail||JSON.stringify(d));
    if(d.metadata){
      html+='<div class="meta">';
      html+='<span>Jira: '+d.metadata.jira_issues_count+' issues</span>';
      html+='<span>Confluence: '+d.metadata.confluence_pages_count+' pages</span>';
      html+='<span>Period: '+d.metadata.time_range_days+' days</span>';
      html+='</div>';
    }
    html+='</div>';
    msgs.innerHTML+=html;
    msgs.scrollTop=msgs.scrollHeight;
  }).catch(e=>{
    document.getElementById('loading').remove();
    msgs.innerHTML+='<div class="msg bot" style="border-color:#ef4444">Error: '+e.message+'</div>';
  }).finally(()=>{btn.disabled=false;input.disabled=false;input.focus();});
}
</script>
</body>
</html>"""


@app.get("/")
async def index():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(HTML_PAGE)


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

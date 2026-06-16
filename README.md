# Incident Insight Agent

AI-powered agent that aggregates issues/incidents from Jira and Confluence, performs root cause analysis, and proposes solutions.

## What it does

- Queries Jira for recent issues/incidents within a configurable time range
- Searches Confluence for related documentation and post-mortems
- Categorizes incidents by theme, component, and severity
- Identifies patterns and recurring issues
- Performs root cause analysis with confidence levels
- Proposes short-term and long-term solutions

## Usage

Send a message describing what you want to analyze:

```
"Summarize all incidents from the last 7 days"
"What issues happened in the PAYMENTS project this week?"
"Analyze critical bugs in the last 30 days related to authentication"
```

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials
2. Build: `docker build -t incident-agent .`
3. Run: `docker run -p 8080:8080 --env-file .env incident-agent`
4. Test: `curl http://localhost:8080/health`

## API

- `GET /health` — Health check
- `POST /chat` — Send analysis request (`{"message": "..."}`)

## Tech Stack

- Python + FastAPI
- Jira REST API (Data Center)
- Confluence REST API (Data Center)
- GreenNode AI Platform (LLM)

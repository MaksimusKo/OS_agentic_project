# ScholarAgent AI

A local scholarship recommendation platform with a plain HTML/CSS/JS frontend and a FastAPI backend.

The app serves a browser dashboard from `app/static` and exposes `/api/chat` for the scholarship agent loop.

---

## What changed

- Frontend is now vanilla **HTML + CSS + JavaScript**
- No `npm` or React build is required for the app
- FastAPI serves both the frontend and the `/api/chat` endpoint
- The UI renders agent answers, scholarship matches, and diagnostics

---

## Project structure

```
prjct/
├── app/
│   ├── agents/
│   │   ├── supervisor.py      ← LangGraph orchestration and tool loop
│   │   ├── tools.py           ← search and evaluation tool implementations
│   │   └── __init__.py
│   ├── database/
│   │   ├── vector_db.py       ← scholarship ranking and scoring logic
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── models.py          ← Pydantic request and response models
│   │   └── __init__.py
│   ├── services/
│   │   ├── tavily_search.py   ← Tavily integration
│   │   └── ...
│   ├── static/
│   │   ├── index.html         ← frontend page served by FastAPI
│   │   ├── style.css          ← UI styles
│   │   └── app.js             ← vanilla JavaScript frontend logic
│   └── main.py                ← FastAPI app, routes, and static mounting
├── tests/
│   └── test_scholar_agent.py  ← backend tests
├── requirements.txt
└── README.md
```

---

## Quick start

### Prerequisites

- Python 3.11+
- `uvicorn`
- `fastapi`

(Optional) Ollama if you want local LangGraph model execution.

### Install dependencies

```bash
cd C:\Users\max\Downloads\prjct
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the app

```bash
uvicorn app.main:app --reload --port 8000
```

Open the UI at:

```text
http://localhost:8000
```

### Important

- This repo does not currently include `package.json`
- `npm run dev` is not used for this frontend
- The frontend is served directly by FastAPI from `app/static`

---

## Frontend

The browser UI is implemented with plain JavaScript in `app/static/app.js`.

Key frontend behavior:

- submits student profile + prompt to `/api/chat`
- highlights the agent pipeline progress
- displays the agent summary
- renders scholarship match cards
- shows diagnostics like tool cycles and latency

---

## API

### `POST /api/chat`

Example request:

```json
{
  "prompt": "Find scholarships for a CS student in Seoul with a 3.7 GPA.",
  "student_profile": {
    "gpa": 3.7,
    "major": "Computer Science",
    "residency": "South Korea",
    "interests": ["machine learning", "open-source"]
  },
  "search_mode": "general"
}
```

Example response:

```json
{
  "answer": "Based on your profile, these scholarships are the best match...",
  "matches": [
    {
      "scholarship_id": "SCH-001",
      "name": "Global STEM Scholarship",
      "provider": "Scholarship Fund",
      "match_score": 85.7,
      "min_gpa": 3.5,
      "required_residency": "Any",
      "eligible": true,
      "summary": "Strong fit for CS and AI interests",
      "evidence": "Matches your residency and GPA requirements"
    }
  ],
  "tool_calls_made": 2,
  "metadata": {
    "total_messages": 12,
    "latency_ms": 2150.3
  }
}
```

### `GET /health`

Returns a simple health check.

---

## Testing

Run backend tests with:

```bash
pytest tests/ -v
```

---

## Development notes

- Edit frontend markup in `app/static/index.html`
- Edit frontend behavior in `app/static/app.js`
- Backend logic is in `app/main.py`, `app/agents/*`, `app/database/*`, and `app/schemas/*`

---

## License

MIT

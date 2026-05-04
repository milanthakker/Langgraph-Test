# LangGraph Agent with Web Search

A simple LangGraph agent that can search the web using DuckDuckGo, exposed via a FastAPI server.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- OpenAI API key

## Setup

1. Install dependencies:

```bash
poetry install
```

2. Create a `.env` file from the example:

```bash
cp .env.example .env
```

3. Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_actual_api_key
```

## Running the API

Start Phoenix and the agent together:

```bash
./start.sh
```

Or start them individually:

```bash
# Terminal 1 — Phoenix UI at http://localhost:6006
poetry run phoenix serve

# Terminal 2 — Agent API at http://localhost:8000
poetry run uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.

### Interactive CLI chat

To chat with the agent directly in your terminal (without the API):

```bash
poetry run python chat.py
```

Type your message and press Enter. Type `quit` or `exit` to end the session.

## API Endpoints

### POST /chat

Send a message to the agent and receive a response.

**Request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest news about AI?"}'
```

**Response:**

```json
{
  "response": "Based on my search, here are the latest developments in AI..."
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```


## Observability

Traces are captured automatically via Arize Phoenix whenever the agent or chat script is running. The Phoenix UI is available at `http://localhost:6006`.

### Export spans to CSV

```bash
poetry run python export_spans.py
```

Writes all spans to a timestamped CSV file: `spanexport_<project>_<datetime>.csv`.

### Run frustration evaluations

```bash
poetry run python evaluate_frustration.py
```

Runs two LLM-as-judge evaluations across all captured spans:

- **Itinerary frustration** — detects when a user asked for an itinerary but the agent asked clarifying questions instead of building one.
- **Search frustration** — detects when a user asked for real-time information but the agent responded from its own knowledge instead of searching.

Results are printed to the terminal and logged back to Phoenix as named evaluations on each span.

## Project Structure

```
se-interview/
├── pyproject.toml   # Poetry dependencies
├── .env.example     # Environment variable template
├── README.md        # This file
├── agent.py         # LangGraph agent implementation
└── api.py           # FastAPI server
```

## How It Works

1. The agent receives a user message via the `/chat` endpoint
2. It calls GPT-4o with the message and available tools (DuckDuckGo search)
3. If the LLM decides to search, it executes the search and feeds results back
4. The loop continues until the LLM provides a final response
5. The response is returned to the user

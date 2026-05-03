#!/usr/bin/env bash
set -e

echo "Starting Phoenix..."
python3 -m phoenix.server.main serve &
PHOENIX_PID=$!

# Give Phoenix a moment to bind its port before the agent tries to export traces
sleep 2

echo "Starting agent..."
poetry run uvicorn api:app --reload &
AGENT_PID=$!

trap "echo 'Shutting down...'; kill $PHOENIX_PID $AGENT_PID 2>/dev/null" INT TERM

echo ""
echo "Phoenix UI: http://localhost:6006"
echo "Agent API:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both."

wait

import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from phoenix.otel import using_session, using_user

load_dotenv()

from agent import build_agent
from database import clear_all_itineraries, init_db

init_db()
agent = build_agent()
history = []
session_id = str(uuid.uuid4())

print("Travel Assistant ready. Type 'quit' to exit.\n")

try:
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        history.append(HumanMessage(content=user_input))
        with using_session(session_id=session_id), using_user("cli-user"):
            result = agent.invoke({"messages": history})
        response = result["messages"][-1]
        history.append(AIMessage(content=response.content))

        print(f"\nAssistant: {response.content}\n")
finally:
    clear_all_itineraries()
    print("Session ended. All itineraries cleared.")

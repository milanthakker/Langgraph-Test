from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

from agent import build_agent
from database import clear_all_itineraries, init_db

init_db()
agent = build_agent()
history = []

print("Travel Assistant ready. Type 'quit' to exit.\n")

try:
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": history})
        response = result["messages"][-1]
        history.append(AIMessage(content=response.content))

        print(f"\nAssistant: {response.content}\n")
finally:
    clear_all_itineraries()
    print("Session ended. All itineraries cleared.")

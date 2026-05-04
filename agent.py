import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from typing_extensions import TypedDict

from tools import build_itinerary, get_past_itineraries

load_dotenv()

tracer_provider = register(
    project_name=os.getenv("PHOENIX_PROJECT_NAME", "travel-agent"),
    endpoint="http://localhost:6006/v1/traces",
    batch=True,
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, build_itinerary, get_past_itineraries]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the current messages and available tools."""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a precise and helpful travel assistant. "
                            "When a user asks to plan a trip or vacation, do NOT call build_itinerary immediately. "
                            "First, ask for any missing details in a single message: destination, number of days, "
                            "and their interests or what they want to focus on each day (e.g. food, history, art, adventure, nature, architecture). "
                            "If they give vague interests, ask a follow-up to get specifics — for example, if they say 'culture', "
                            "ask whether they prefer museums, architecture, live music, or street art. "
                            "Once you have destination, duration, and clear interests, call build_itinerary. "
                            "Use the search tool to get information about the current weather, and real-time news about the destination."
                        )
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState) -> dict:
    """Execute tool calls from the last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def build_agent():
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")

    agent = graph_builder.compile()
    return agent
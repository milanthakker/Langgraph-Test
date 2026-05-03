import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import phoenix as px
from phoenix.evals import OpenAIModel, llm_classify
from phoenix.trace.span_evaluations import SpanEvaluations
from phoenix.evals.models import set_verbosity

set_verbosity(False)

PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "travel-agent")
client = px.Client(endpoint="http://localhost:6006")
spans = client.get_spans_dataframe(project_name=PROJECT)

# Build sets of trace IDs by tool usage
traces_with_itinerary = set(
    spans[
        (spans["span_kind"] == "TOOL") &
        (spans["attributes.tool.name"].str.contains("itinerary", case=False, na=False))
    ]["context.trace_id"]
)
traces_with_search = set(
    spans[
        (spans["span_kind"] == "TOOL") &
        (spans["attributes.tool.name"].str.contains("search", case=False, na=False))
    ]["context.trace_id"]
)
traces_with_any_tool = set(spans[spans["span_kind"] == "TOOL"]["context.trace_id"])
traces_with_no_tools = set(spans["context.trace_id"]) - traces_with_any_tool

model = OpenAIModel(model="gpt-4o", temperature=0)

# =============================================================================
# LLM SPAN EVALUATIONS
# =============================================================================

llm_spans = spans[spans["span_kind"] == "LLM"].copy()
llm_spans = llm_spans.rename(columns={
    "attributes.input.value": "input",
    "attributes.output.value": "output",
})
llm_spans["itinerary_tool_called"] = llm_spans["context.trace_id"].isin(traces_with_itinerary)
llm_spans["search_tool_called"] = llm_spans["context.trace_id"].isin(traces_with_search)
llm_spans["no_tools_called"] = llm_spans["context.trace_id"].isin(traces_with_no_tools)

ITINERARY_LLM_TEMPLATE = """
You are evaluating a travel assistant. Given the user input and assistant response below,
determine whether the user asked for a travel itinerary but did not receive one.

The assistant has a tool called `build_itinerary`. If the assistant asked clarifying questions
instead of building the itinerary when the user clearly wanted one, that is a frustration.

User input: {input}
Assistant response: {output}
Itinerary tool was called in this trace: {itinerary_tool_called}

Was the user frustrated because they asked for an itinerary but didn't get one?
Respond with one of: frustrated / not_frustrated
"""

SEARCH_LLM_TEMPLATE = """
You are evaluating a travel assistant. Given the user input and assistant response below,
determine whether the user asked for real-time or current information (e.g. current weather,
live flight prices, recent news, current events) but the assistant responded from its own
training knowledge instead of using a search tool.

User input: {input}
Assistant response: {output}
Search tool was called in this trace: {search_tool_called}

Was the user frustrated because they asked for real-time information but the assistant
responded from its own knowledge without searching?
Respond with one of: frustrated / not_frustrated
"""

llm_itinerary_eval = llm_classify(
    dataframe=llm_spans,
    template=ITINERARY_LLM_TEMPLATE,
    model=model,
    rails=["frustrated", "not_frustrated"],
    provide_explanation=True,
)
llm_itinerary_eval.index = llm_spans["context.span_id"]

llm_search_eval = llm_classify(
    dataframe=llm_spans,
    template=SEARCH_LLM_TEMPLATE,
    model=model,
    rails=["frustrated", "not_frustrated"],
    provide_explanation=True,
)
llm_search_eval.index = llm_spans["context.span_id"]

# =============================================================================
# TOOL SPAN EVALUATIONS
# =============================================================================

tool_spans = spans[spans["span_kind"] == "TOOL"].copy()
tool_spans = tool_spans.rename(columns={
    "attributes.input.value": "input",
    "attributes.output.value": "output",
    "attributes.tool.name": "tool_name",
})

itinerary_tool_spans = tool_spans[tool_spans["tool_name"].str.contains("itinerary", case=False, na=False)]
search_tool_spans = tool_spans[tool_spans["tool_name"].str.contains("search", case=False, na=False)]
other_tool_spans = tool_spans[
    ~tool_spans["tool_name"].str.contains("itinerary|search", case=False, na=False)
]

ITINERARY_TOOL_TEMPLATE = """
You are evaluating a travel assistant's tool usage. The `build_itinerary` tool was invoked.
Determine whether the tool was called appropriately — i.e. the user had clearly requested
an itinerary and provided sufficient details.

Tool name: {tool_name}
Tool input: {input}
Tool output: {output}

Was this tool invoked appropriately, or does it indicate a frustration (called when the user hadn't asked for an itinerary)?
Respond with one of: frustrated / not_frustrated
"""

SEARCH_TOOL_TEMPLATE = """
You are evaluating a travel assistant's tool usage. A search tool was invoked.
Determine whether the search was relevant to the user's request and whether the
results adequately addressed their need for current information.

Tool name: {tool_name}
Tool input: {input}
Tool output: {output}

Did the search tool call and its result adequately address the user's need,
or does it indicate a frustration (e.g. irrelevant search, poor results)?
Respond with one of: frustrated / not_frustrated
"""

tool_itinerary_eval = llm_classify(
    dataframe=itinerary_tool_spans,
    template=ITINERARY_TOOL_TEMPLATE,
    model=model,
    rails=["frustrated", "not_frustrated"],
    provide_explanation=True,
)
tool_itinerary_eval.index = itinerary_tool_spans["context.span_id"]

tool_search_eval = llm_classify(
    dataframe=search_tool_spans,
    template=SEARCH_TOOL_TEMPLATE,
    model=model,
    rails=["frustrated", "not_frustrated"],
    provide_explanation=True,
)
tool_search_eval.index = search_tool_spans["context.span_id"]

# =============================================================================
# BUILD COMPLETE EVAL DATAFRAMES — one per evaluation type
# Non-relevant spans default to "not_frustrated" / score 1
# =============================================================================

def default_not_frustrated(span_ids):
    return pd.DataFrame(
        {"label": "not_frustrated", "score": 1, "explanation": "Not applicable for this span type."},
        index=span_ids,
    )

# Itinerary frustration: LLM + itinerary tool spans evaluated; others default
itinerary_frames = [
    llm_itinerary_eval.assign(score=llm_itinerary_eval["label"].map({"frustrated": 0, "not_frustrated": 1})),
    tool_itinerary_eval.assign(score=tool_itinerary_eval["label"].map({"frustrated": 0, "not_frustrated": 1})),
    default_not_frustrated(search_tool_spans["context.span_id"]),
    default_not_frustrated(other_tool_spans["context.span_id"]),
]
itinerary_evals = pd.concat(itinerary_frames)

# Search frustration: LLM + search tool spans evaluated; others default
search_frames = [
    llm_search_eval.assign(score=llm_search_eval["label"].map({"frustrated": 0, "not_frustrated": 1})),
    tool_search_eval.assign(score=tool_search_eval["label"].map({"frustrated": 0, "not_frustrated": 1})),
    default_not_frustrated(itinerary_tool_spans["context.span_id"]),
    default_not_frustrated(other_tool_spans["context.span_id"]),
]
search_evals = pd.concat(search_frames)

# =============================================================================
# LOG EVALUATIONS BACK TO PHOENIX
# =============================================================================

px.log_evaluations(
    SpanEvaluations(eval_name="itinerary_frustration", dataframe=itinerary_evals),
    SpanEvaluations(eval_name="search_frustration", dataframe=search_evals),
)
print("Evaluations logged to Phoenix.")

# =============================================================================
# REPORT
# =============================================================================

print("\n=== Frustration Evaluation Results ===\n")
print(f"Total spans evaluated:     {len(itinerary_evals)}")
print(f"  LLM spans:               {len(llm_spans)}")
print(f"  TOOL spans:              {len(tool_spans)}")
print(f"  Traces with no tools:    {len(traces_with_no_tools)}")
print(f"Itinerary frustrations:    {(itinerary_evals['label'] == 'frustrated').sum()}")
print(f"Search frustrations:       {(search_evals['label'] == 'frustrated').sum()}")

frustrated_itinerary = itinerary_evals[itinerary_evals["label"] == "frustrated"]
frustrated_search = search_evals[search_evals["label"] == "frustrated"]

if not frustrated_itinerary.empty or not frustrated_search.empty:
    print("\n--- Frustrated spans ---")
    for span_id, row in frustrated_itinerary.iterrows():
        print(f"\nSpan {span_id}  [itinerary] {row['explanation']}")
    for span_id, row in frustrated_search.iterrows():
        print(f"\nSpan {span_id}  [search] {row['explanation']}")

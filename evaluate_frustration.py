import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from phoenix.client import Client
from phoenix.evals import OpenAIModel, llm_classify
from phoenix.evals.models import set_verbosity

set_verbosity(False)

PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "travel-agent")
client = Client(base_url="http://localhost:6006")
spans = client.spans.get_spans_dataframe(project_name=PROJECT)

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

model = OpenAIModel(model="gpt-4o", temperature=0)

# =============================================================================
# LLM SPAN EVALUATIONS
# Only evaluate spans in traces where the relevant tool was NOT called —
# those are the only traces where LLM-level frustration can exist.
# Traces where the tool fired are defaulted to not_frustrated below.
# =============================================================================

llm_spans = spans[spans["span_kind"] == "LLM"].copy()
llm_spans = llm_spans.rename(columns={
    "attributes.input.value": "input",
    "attributes.output.value": "output",
})

llm_no_itinerary = llm_spans[~llm_spans["context.trace_id"].isin(traces_with_itinerary)]
llm_with_itinerary = llm_spans[llm_spans["context.trace_id"].isin(traces_with_itinerary)]

llm_no_search = llm_spans[~llm_spans["context.trace_id"].isin(traces_with_search)]
llm_with_search = llm_spans[llm_spans["context.trace_id"].isin(traces_with_search)]

ITINERARY_LLM_TEMPLATE = """
You are evaluating a travel assistant. Given the user input and assistant response below,
determine whether the user asked for a travel itinerary but did not receive one.

The assistant has a tool called `build_itinerary`. If the assistant asked clarifying questions
instead of building the itinerary when the user clearly wanted one, that is a frustration.

User input: {input}
Assistant response: {output}

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

Was the user frustrated because they asked for real-time information but the assistant
responded from its own knowledge without searching?
Respond with one of: frustrated / not_frustrated
"""

llm_itinerary_eval = llm_classify(
    dataframe=llm_no_itinerary,
    template=ITINERARY_LLM_TEMPLATE,
    model=model,
    rails=["frustrated", "not_frustrated"],
    provide_explanation=True,
)
llm_itinerary_eval.index = llm_no_itinerary["context.span_id"]

llm_search_eval = llm_classify(
    dataframe=llm_no_search,
    template=SEARCH_LLM_TEMPLATE,
    model=model,
    rails=["frustrated", "not_frustrated"],
    provide_explanation=True,
)
llm_search_eval.index = llm_no_search["context.span_id"]

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
Determine whether the tool was called appropriately — i.e. the user was looking for an itinerary or plan for their vacation.

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

def score(eval_df):
    return eval_df.assign(score=eval_df["label"].map({"frustrated": 0, "not_frustrated": 1}))

# Itinerary frustration:
#   - LLM spans in traces without build_itinerary → evaluated
#   - LLM spans in traces WITH build_itinerary → default not_frustrated
#   - build_itinerary tool spans → evaluated
#   - search/other tool spans → default not_frustrated
itinerary_evals = pd.concat([
    score(llm_itinerary_eval),
    default_not_frustrated(llm_with_itinerary["context.span_id"]),
    score(tool_itinerary_eval),
    default_not_frustrated(search_tool_spans["context.span_id"]),
    default_not_frustrated(other_tool_spans["context.span_id"]),
])

# Search frustration:
#   - LLM spans in traces without search → evaluated
#   - LLM spans in traces WITH search → default not_frustrated
#   - search tool spans → evaluated
#   - build_itinerary/other tool spans → default not_frustrated
search_evals = pd.concat([
    score(llm_search_eval),
    default_not_frustrated(llm_with_search["context.span_id"]),
    score(tool_search_eval),
    default_not_frustrated(itinerary_tool_spans["context.span_id"]),
    default_not_frustrated(other_tool_spans["context.span_id"]),
])

# =============================================================================
# LOG EVALUATIONS BACK TO PHOENIX
# =============================================================================

itinerary_evals.index.name = "span_id"
search_evals.index.name = "span_id"

client.spans.log_span_annotations_dataframe(
    dataframe=itinerary_evals,
    annotation_name="itinerary_frustration",
    annotator_kind="LLM",
)
client.spans.log_span_annotations_dataframe(
    dataframe=search_evals,
    annotation_name="search_frustration",
    annotator_kind="LLM",
)
print("Evaluations logged to Phoenix.")

# =============================================================================
# REPORT
# =============================================================================

print("\n=== Frustration Evaluation Results ===\n")
print(f"LLM spans evaluated (itinerary): {len(llm_no_itinerary)} / {len(llm_spans)}")
print(f"LLM spans evaluated (search):    {len(llm_no_search)} / {len(llm_spans)}")
print(f"Tool spans evaluated:            {len(itinerary_tool_spans)} itinerary, {len(search_tool_spans)} search")
print(f"Itinerary frustrations:          {(itinerary_evals['label'] == 'frustrated').sum()}")
print(f"Search frustrations:             {(search_evals['label'] == 'frustrated').sum()}")

frustrated_itinerary = itinerary_evals[itinerary_evals["label"] == "frustrated"]
frustrated_search = search_evals[search_evals["label"] == "frustrated"]

if not frustrated_itinerary.empty or not frustrated_search.empty:
    print("\n--- Frustrated spans ---")
    for span_id, row in frustrated_itinerary.iterrows():
        print(f"\nSpan {span_id}  [itinerary] {row['explanation']}")
    for span_id, row in frustrated_search.iterrows():
        print(f"\nSpan {span_id}  [search] {row['explanation']}")

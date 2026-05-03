from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from database import get_itinerary, list_itineraries, save_itinerary


class DayPlan(BaseModel):
    day: int
    theme: str
    morning: str
    afternoon: str
    evening: str


class Itinerary(BaseModel):
    destination: str
    duration_days: int
    travel_style: str
    daily_plans: list[DayPlan]
    packing_tips: list[str]
    best_time_to_visit: str


_llm = ChatOpenAI(model="gpt-4o", temperature=0.7).with_structured_output(Itinerary)


@tool
def build_itinerary(
    destination: str,
    num_days: int,
    interests: str,
    day_themes: Optional[str] = None,
) -> dict:
    """Build a precise, structured day-by-day travel itinerary for a destination.

    Only call this tool once you have collected ALL of the following from the user:
    - destination
    - number of days
    - interests or preferred activities (e.g. food, history, art, adventure, nature, architecture)
    - optionally: a preferred theme or focus for each day

    If any of these are missing, ask the user before calling this tool.

    Args:
        destination: The city or country to visit.
        num_days: Number of days for the trip.
        interests: Comma-separated list of interests (e.g. "food, history"). Required.
        day_themes: Optional comma-separated list of per-day themes, one per non-arrival/departure day
                    (e.g. "street food focus, museums and galleries, day trip to coast").
    """
    day_themes_instruction = (
        f"The user has requested these specific themes for each middle day (in order, excluding arrival and departure): {day_themes}."
        if day_themes
        else "Choose appropriate themes based on the user's interests."
    )

    prompt = f"""Create a detailed {num_days}-day travel itinerary for {destination}.

The traveller's interests are: {interests}.
{day_themes_instruction}

Rules:
- Day 1 is always arrival and gentle orientation.
- Day {num_days} is always departure.
- For every other day, suggest specific, named places, venues, dishes, or experiences — not generic descriptions.
- Morning, afternoon, and evening activities should each be distinct and realistic for the destination.
- Packing tips should be tailored to the destination and the traveller's interests.
- best_time_to_visit should be a specific month range with a brief reason.
"""

    itinerary: Itinerary = _llm.invoke([
        SystemMessage(content="You are an expert travel planner with deep knowledge of destinations worldwide."),
        HumanMessage(content=prompt),
    ])

    data = itinerary.model_dump()
    itinerary_id = save_itinerary(data)
    data["itinerary_id"] = itinerary_id
    return data


@tool
def get_past_itineraries(itinerary_id: Optional[int] = None) -> dict:
    """Retrieve previously created itineraries from the database.

    If itinerary_id is provided, returns the full itinerary for that ID.
    If no itinerary_id is provided, returns a summary list of all past itineraries.

    Args:
        itinerary_id: Optional ID of a specific itinerary to retrieve in full.
    """
    if itinerary_id is not None:
        result = get_itinerary(itinerary_id)
        if not result:
            return {"error": f"No itinerary found with ID {itinerary_id}"}
        return result

    summaries = list_itineraries()
    if not summaries:
        return {"message": "No itineraries have been created yet."}
    return {"itineraries": summaries}

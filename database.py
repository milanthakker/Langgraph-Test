from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DATABASE_URL = "sqlite:///./itineraries.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class ItineraryRecord(Base):
    __tablename__ = "itineraries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    destination = Column(String, nullable=False)
    duration_days = Column(Integer, nullable=False)
    travel_style = Column(String, nullable=False)
    packing_tips = Column(Text, nullable=False)  # stored as newline-separated string
    best_time_to_visit = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    day_plans = relationship("DayPlanRecord", back_populates="itinerary", cascade="all, delete-orphan")


class DayPlanRecord(Base):
    __tablename__ = "day_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    itinerary_id = Column(Integer, ForeignKey("itineraries.id"), nullable=False)
    day = Column(Integer, nullable=False)
    theme = Column(String, nullable=False)
    morning = Column(Text, nullable=False)
    afternoon = Column(Text, nullable=False)
    evening = Column(Text, nullable=False)

    itinerary = relationship("ItineraryRecord", back_populates="day_plans")


def init_db():
    Base.metadata.create_all(bind=engine)


def save_itinerary(itinerary_data: dict) -> int:
    """Persist an itinerary and its day plans. Returns the new itinerary ID."""
    with Session(engine) as session:
        record = ItineraryRecord(
            destination=itinerary_data["destination"],
            duration_days=itinerary_data["duration_days"],
            travel_style=itinerary_data["travel_style"],
            packing_tips="\n".join(itinerary_data["packing_tips"]),
            best_time_to_visit=itinerary_data["best_time_to_visit"],
        )
        session.add(record)
        session.flush()

        for day in itinerary_data["daily_plans"]:
            session.add(
                DayPlanRecord(
                    itinerary_id=record.id,
                    day=day["day"],
                    theme=day["theme"],
                    morning=day["morning"],
                    afternoon=day["afternoon"],
                    evening=day["evening"],
                )
            )

        session.commit()
        return record.id


def clear_all_itineraries() -> None:
    """Delete all itineraries and day plans from the database."""
    with Session(engine) as session:
        session.query(DayPlanRecord).delete()
        session.query(ItineraryRecord).delete()
        session.commit()


def list_itineraries() -> list[dict]:
    """Return a summary of all stored itineraries."""
    with Session(engine) as session:
        records = session.query(ItineraryRecord).order_by(ItineraryRecord.created_at.desc()).all()
        return [
            {
                "id": r.id,
                "destination": r.destination,
                "duration_days": r.duration_days,
                "travel_style": r.travel_style,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]


def get_itinerary(itinerary_id: int) -> dict | None:
    """Retrieve a full itinerary by ID."""
    with Session(engine) as session:
        record = session.get(ItineraryRecord, itinerary_id)
        if not record:
            return None
        return {
            "id": record.id,
            "destination": record.destination,
            "duration_days": record.duration_days,
            "travel_style": record.travel_style,
            "packing_tips": record.packing_tips.split("\n"),
            "best_time_to_visit": record.best_time_to_visit,
            "created_at": record.created_at.isoformat(),
            "daily_plans": [
                {
                    "day": p.day,
                    "theme": p.theme,
                    "morning": p.morning,
                    "afternoon": p.afternoon,
                    "evening": p.evening,
                }
                for p in sorted(record.day_plans, key=lambda x: x.day)
            ],
        }

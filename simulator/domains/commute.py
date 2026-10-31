"""Light domain 1: commute and travel planning. No life events."""

from __future__ import annotations

from simulator.core import Attribute, Domain, State, Task

HOME_AREAS = ("city_center", "inner_suburb", "outer_suburb", "rural")
WORK_LOCATIONS = ("downtown", "business_park", "university_campus", "industrial_zone")

# distance[home_area][work_location] -> short | medium | long
DISTANCE: dict[str, dict[str, str]] = {
    "city_center": {"downtown": "short", "business_park": "medium",
                    "university_campus": "medium", "industrial_zone": "long"},
    "inner_suburb": {"downtown": "medium", "business_park": "short",
                     "university_campus": "short", "industrial_zone": "medium"},
    "outer_suburb": {"downtown": "long", "business_park": "medium",
                     "university_campus": "long", "industrial_zone": "short"},
    "rural": {"downtown": "long", "business_park": "long",
              "university_campus": "medium", "industrial_zone": "medium"},
}

ATTRIBUTES = (
    Attribute("home_area", HOME_AREAS, "slow"),
    Attribute("work_location", WORK_LOCATIONS, "slow"),
    Attribute("office_days", ("remote", "hybrid_light", "hybrid_heavy", "full_office"), "fast"),
    Attribute("has_car", ("yes", "no"), "slow"),
    Attribute("weekly_budget", ("low", "medium", "high"), "slow"),
    Attribute("preferred_language", ("en", "es", "fr", "de"), "static"),
)


def _distance(v: State) -> str:
    return DISTANCE[v["home_area"]][v["work_location"]]


def commute_plan(v: State) -> str:
    """How should I get to work this week? Remote -> no plan; short distance -> bike;
    otherwise per-trip or monthly driving or transit by office days."""
    if v["office_days"] == "remote":
        return "work_from_home"
    d = _distance(v)
    if d == "short":
        return "bike"
    drive = v["has_car"] == "yes" and d == "long"
    if v["office_days"] == "hybrid_light":
        return "daily_parking" if drive else "pay_per_ride"
    return "monthly_parking" if drive else "monthly_transit_pass"


def weekend_outing(v: State) -> str:
    """Plan a cheap weekend outing. office_days (fast) does not matter here."""
    car = v["has_car"] == "yes"
    if v["weekly_budget"] == "low":
        return "local_park"
    if v["weekly_budget"] == "medium":
        return "nearby_hike" if car else "city_museum"
    return "day_trip_by_car" if car else "day_trip_by_train"


def transit_fare(v: State) -> str:
    """Which transit fare should I buy? Tiered by trips per month; not needed if remote
    or if driving is affordable."""
    if v["office_days"] == "remote":
        return "none_needed"
    if v["has_car"] == "yes" and v["weekly_budget"] == "high":
        return "none_needed"
    return {"hybrid_light": "pay_per_ride", "hybrid_heavy": "ten_ride_bundle",
            "full_office": "monthly_pass"}[v["office_days"]]


def commute_podcast(v: State) -> str:
    """Suggest a podcast for my commute (language and episode length)."""
    length = "short" if _distance(v) == "short" else "long"
    return f"{v['preferred_language']}_{length}"


TASKS = (
    Task("commute_plan", ("office_days", "has_car", "home_area", "work_location"),
         ("work_from_home", "bike", "pay_per_ride", "daily_parking", "monthly_transit_pass",
          "monthly_parking"), commute_plan,
         "How should I get to work this week, and what should I pay for (parking, passes)?"),
    Task("weekend_outing", ("weekly_budget", "has_car"),
         ("local_park", "nearby_hike", "city_museum", "day_trip_by_car", "day_trip_by_train"),
         weekend_outing, "Plan a cheap weekend outing for me."),
    Task("transit_fare", ("office_days", "has_car", "weekly_budget"),
         ("none_needed", "pay_per_ride", "ten_ride_bundle", "monthly_pass"), transit_fare,
         "Which public transit ticket or pass should I buy?"),
    Task("commute_podcast", ("preferred_language", "home_area", "work_location"),
         tuple(f"{lang}_{n}" for lang in ("en", "es", "fr", "de") for n in ("short", "long")),
         commute_podcast, "Suggest a podcast to listen to on my commute."),
)

DOMAIN = Domain(name="commute", attributes=ATTRIBUTES, tasks=TASKS)

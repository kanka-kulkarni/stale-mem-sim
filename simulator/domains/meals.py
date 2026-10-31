"""Case study: grocery and meal planning (food logistics only).

No nutrition targets, weight or health conditions: dietary_choice is a choice
(none / vegetarian / vegan), not a medical need. Three life events change several
attributes together.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from simulator.core import Attribute, Domain, LifeEvent, State, Task

HOUSEHOLD = ("1", "2", "3-4", "5+")
BUDGET = ("tight", "moderate", "comfortable")
COOK_TIME = ("15min", "30min", "60min")
PANTRY = ("pasta_and_sauce", "rice_and_beans", "eggs_and_bread", "nearly_empty")
APPLIANCES = ("microwave_only", "stovetop", "stovetop_and_oven", "full_kitchen")
CUISINES = ("italian", "mexican", "indian", "east_asian")
STORES = ("discount_grocer", "supermarket", "farmers_market", "convenience_only")
SHOP_DAYS = ("weekday_evening", "saturday", "sunday")
DIETS = ("none", "vegetarian", "vegan")

ATTRIBUTES = (
    Attribute("household_size", HOUSEHOLD, "slow"),
    Attribute("weekly_grocery_budget", BUDGET, "slow"),
    Attribute("cooking_time_weeknights", COOK_TIME, "fast"),
    Attribute("pantry_staples", PANTRY, "fast"),
    Attribute("appliances", APPLIANCES, "slow"),
    Attribute("cuisine_likes", CUISINES, "slow"),
    Attribute("nearby_stores", STORES, "slow"),
    Attribute("shopping_day", SHOP_DAYS, "fast"),
    Attribute("dietary_choice", DIETS, "slow"),
)

HAS_OVEN = ("stovetop_and_oven", "full_kitchen")


# --------------------------------------------------------------------------- tasks


def dinner_tonight(v: State) -> str:
    """What can I cook tonight with what I have? (fast + slow)"""
    pantry, time, app = v["pantry_staples"], v["cooking_time_weeknights"], v["appliances"]
    if pantry == "nearly_empty":
        return "quick_shop_first"
    if pantry == "pasta_and_sauce":
        if app == "microwave_only":
            return "microwave_pasta"
        return "baked_pasta" if app in HAS_OVEN and time == "60min" else "stovetop_pasta"
    if pantry == "rice_and_beans":
        return "rice_and_bean_bowl" if app == "microwave_only" or time == "15min" else "bean_chili"
    if v["dietary_choice"] == "vegan":  # eggs_and_bread
        return "beans_on_toast"
    return "veggie_frittata" if app in HAS_OVEN and time != "15min" else "scrambled_eggs_on_toast"


DIET_TAG = {"none": "mixed", "vegetarian": "vegetarian", "vegan": "vegan"}
STYLE = {"15min": "assembly", "30min": "one_pot", "60min": "from_scratch"}


def week_dinner_plan(v: State) -> str:
    """Plan this week's dinners: cuisine x diet x cooking style. (fast + slow)"""
    return (f"{v['cuisine_likes']}_{DIET_TAG[v['dietary_choice']]}_"
            f"{STYLE[v['cooking_time_weeknights']]}")


PORTION = {"1": "single", "2": "couple", "3-4": "family", "5+": "bulk"}
PROTEIN = {"none": "meat_veg", "vegetarian": "eggs_dairy_legumes", "vegan": "legumes_tofu"}
TIER = {"tight": "store_brand", "moderate": "mixed_brands", "comfortable": "premium"}


def grocery_list(v: State) -> str:
    """Build my grocery list for the week: portions x protein source x brand tier.
    (slow only)"""
    return (f"{PORTION[v['household_size']]}_{PROTEIN[v['dietary_choice']]}_"
            f"{TIER[v['weekly_grocery_budget']]}")


def which_store(v: State) -> str:
    """Which store should I go to this week? (slow only)"""
    budget, store = v["weekly_grocery_budget"], v["nearby_stores"]
    if v["household_size"] == "5+" and budget != "comfortable":
        return "warehouse_club"
    if store == "convenience_only":
        return "online_delivery" if budget == "comfortable" else "convenience_store"
    if store == "discount_grocer":
        return "discount_grocer"
    if store == "supermarket":
        return "supermarket_store_brands" if budget == "tight" else "supermarket"
    return "farmers_market_end_of_day" if budget == "tight" else "farmers_market"


def grocery_run_timing(v: State) -> str:
    """When should I do my grocery run this week? (fast only)"""
    pantry, day = v["pantry_staples"], v["shopping_day"]
    if pantry == "nearly_empty":
        return "shop_today"
    kind = "full_shop" if pantry == "eggs_and_bread" else "top_up"
    return f"{kind}_{day}"


def batch_cook(v: State) -> str:
    """Should I batch-cook this weekend, and how? (fast + slow)"""
    app = v["appliances"]
    if v["cooking_time_weeknights"] == "60min":
        return "no_batch_needed"
    if app == "microwave_only":
        return "buy_prepared_meals"
    if v["household_size"] == "1":
        return "freeze_single_portions"
    return {"full_kitchen": "slow_cooker_batch", "stovetop_and_oven": "sheet_pan_batch",
            "stovetop": "big_pot_batch"}[app]


def cook_or_takeout(v: State) -> str:
    """Should I cook tonight or get takeout? (fast + slow)"""
    empty = v["pantry_staples"] == "nearly_empty"
    rushed = v["cooking_time_weeknights"] == "15min"
    budget = v["weekly_grocery_budget"]
    if budget == "tight":
        return "quick_shop_then_cook" if empty else "cook_at_home"
    if budget == "comfortable":
        return "takeout" if empty or rushed else "cook_at_home"
    if empty:
        return "takeout" if rushed else "quick_shop_then_cook"
    return "cook_at_home"


def _opts(*parts: tuple[str, ...]) -> tuple[str, ...]:
    out = [""]
    for p in parts:
        out = [f"{o}_{x}" if o else x for o in out for x in p]
    return tuple(out)


TASKS = (
    Task("dinner_tonight",
         ("pantry_staples", "cooking_time_weeknights", "appliances", "dietary_choice"),
         ("quick_shop_first", "stovetop_pasta", "baked_pasta", "microwave_pasta",
          "rice_and_bean_bowl", "bean_chili", "veggie_frittata", "scrambled_eggs_on_toast",
          "beans_on_toast"),
         dinner_tonight, "What can I cook tonight with what I have at home?"),
    Task("week_dinner_plan", ("cuisine_likes", "dietary_choice", "cooking_time_weeknights"),
         _opts(CUISINES, tuple(DIET_TAG.values()), tuple(STYLE.values())),
         week_dinner_plan, "Plan this week's dinners for me."),
    Task("grocery_list", ("household_size", "dietary_choice", "weekly_grocery_budget"),
         _opts(tuple(PORTION.values()), tuple(PROTEIN.values()), tuple(TIER.values())),
         grocery_list, "Build my grocery list for the week."),
    Task("which_store", ("nearby_stores", "weekly_grocery_budget", "household_size"),
         ("warehouse_club", "online_delivery", "convenience_store", "discount_grocer",
          "supermarket_store_brands", "supermarket", "farmers_market_end_of_day",
          "farmers_market"),
         which_store, "Which store should I go to for groceries this week?"),
    Task("grocery_run_timing", ("shopping_day", "pantry_staples"),
         ("shop_today",) + _opts(("full_shop", "top_up"), SHOP_DAYS),
         grocery_run_timing, "When should I do my grocery run this week?"),
    Task("batch_cook", ("appliances", "cooking_time_weeknights", "household_size"),
         ("no_batch_needed", "buy_prepared_meals", "freeze_single_portions",
          "slow_cooker_batch", "sheet_pan_batch", "big_pot_batch"),
         batch_cook, "Should I batch-cook this weekend, and if so, how?"),
    Task("cook_or_takeout",
         ("cooking_time_weeknights", "pantry_staples", "weekly_grocery_budget"),
         ("cook_at_home", "takeout", "quick_shop_then_cook"),
         cook_or_takeout, "Should I cook tonight or get takeout?"),
)


# --------------------------------------------------------------------------- events


def _step(values: tuple[str, ...], current: str, delta: int) -> str:
    i = values.index(current) + delta
    return values[min(max(i, 0), len(values) - 1)]


def _other(values: tuple[str, ...], current: str, rng: np.random.Generator) -> str:
    others = [v for v in values if v != current]
    return others[int(rng.integers(len(others)))]


def moved_house(s: State, rng: np.random.Generator) -> State:
    """New neighbourhood: different stores, re-chosen shopping day, and half the time
    the grocery budget shifts one level up or down."""
    out = {"nearby_stores": _other(STORES, s["nearby_stores"], rng),
           "shopping_day": SHOP_DAYS[int(rng.integers(len(SHOP_DAYS)))]}
    shift = int(rng.integers(4))  # 0: down, 1: up, 2-3: unchanged
    if shift < 2:
        out["weekly_grocery_budget"] = _step(BUDGET, s["weekly_grocery_budget"],
                                             -1 if shift == 0 else 1)
    return out


def new_household_member(s: State, rng: np.random.Generator) -> State:
    """Someone moves in: household one size up, budget one level tighter, less
    weeknight cooking time."""
    return {"household_size": _step(HOUSEHOLD, s["household_size"], 1),
            "weekly_grocery_budget": _step(BUDGET, s["weekly_grocery_budget"], -1),
            "cooking_time_weeknights": _step(COOK_TIME, s["cooking_time_weeknights"], -1)}


def new_job_schedule(s: State, rng: np.random.Generator) -> State:
    """New work hours: different weeknight cooking time and shopping day."""
    return {"cooking_time_weeknights": _other(COOK_TIME, s["cooking_time_weeknights"], rng),
            "shopping_day": _other(SHOP_DAYS, s["shopping_day"], rng)}


EVENTS = (
    LifeEvent("moved_house", 0.01, ("nearby_stores", "shopping_day", "weekly_grocery_budget"),
              moved_house),
    LifeEvent("new_household_member", 0.01,
              ("household_size", "weekly_grocery_budget", "cooking_time_weeknights"),
              new_household_member),
    LifeEvent("new_job_schedule", 0.02, ("cooking_time_weeknights", "shopping_day"),
              new_job_schedule),
)

DOMAIN = Domain(name="meals", attributes=ATTRIBUTES, tasks=TASKS, events=EVENTS)


# --------------------------------------------------------------------------- calibrated rates

# Per-session change rates derived from VitaBench 2.0 (calibration/vitabench_rates.py,
# per_tag_rate: update + delete per stored tag per subtask transition). Attributes with
# no VitaBench analogue keep their class default. Life events are unchanged.
CALIBRATED_RATES: dict[str, float] = {
    "cuisine_likes": 0.059,          # food
    "dietary_choice": 0.059,         # food
    "weekly_grocery_budget": 0.043,  # shopping/spending
    "nearby_stores": 0.043,          # shopping/spending
    "appliances": 0.022,             # other
}
CALIBRATION_SOURCE = {"cuisine_likes": "food", "dietary_choice": "food",
                      "weekly_grocery_budget": "shopping_spending",
                      "nearby_stores": "shopping_spending", "appliances": "other"}

DOMAIN_CALIBRATED = Domain(
    name="meals_calibrated",
    attributes=tuple(replace(a, rate=CALIBRATED_RATES.get(a.name, a.rate)) for a in ATTRIBUTES),
    tasks=TASKS, events=EVENTS,
)

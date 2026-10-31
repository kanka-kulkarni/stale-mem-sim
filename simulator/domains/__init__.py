"""Domain registry."""

from __future__ import annotations

from simulator.core import Domain


def get_domain(name: str) -> Domain:
    """Return a domain by name."""
    if name == "commute":
        from simulator.domains.commute import DOMAIN
        return DOMAIN
    if name == "meals":
        from simulator.domains.meals import DOMAIN
        return DOMAIN
    if name == "meals_calibrated":
        from simulator.domains.meals import DOMAIN_CALIBRATED
        return DOMAIN_CALIBRATED
    raise KeyError(f"unknown domain {name!r}")


# Domain whose rates are the "default class rates" counterpart of each domain.
DEFAULT_RATE_DOMAIN = {"meals_calibrated": "meals"}

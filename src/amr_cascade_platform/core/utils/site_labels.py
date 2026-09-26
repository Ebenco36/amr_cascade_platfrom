"""Display names for the source sites, shared by tables and figures."""

from __future__ import annotations

ALL_SITES = "all_sites"
SITE_DISPLAY = {
    ALL_SITES: "All sites",
    "armd": "Stanford",
    "armd_ecuh": "ECU Health",
    "armd_utsw": "UT Southwestern",
}


def site_label(site: str) -> str:
    """"armd" -> "Stanford"; an unknown code is returned unchanged."""
    return SITE_DISPLAY.get(site, site)


def scope_order(sites: tuple[str, ...] | list[str], present: object = ()) -> list[str]:
    """All sites first, then the configured sites, then any other site present in the data."""
    configured = [site for site in sites if site != ALL_SITES]
    extra = sorted({str(site) for site in present} - set(configured) - {ALL_SITES})
    return [ALL_SITES, *configured, *extra]

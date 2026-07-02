"""
Seed examples for LGeo relevance scoring.

SEED_EXAMPLES is a flat pool of (text, label) pairs used as the baseline
neighbourhood for the k-NN scorer in matcher/score.py — label 1 means
"LGeo would bid on this", label 0 means "LGeo would not". There is no
separate blocklist or negative-query mechanism: everything the scorer
knows about is just a labeled example in this pool, weighted the same
way as a real human-labeled RFP (see SEED_WEIGHT / REAL_WEIGHT in
matcher/score.py).

Add more examples here as you identify new false positives/negatives
that aren't already covered by an existing entry.
"""

SEED_EXAMPLES: list[tuple[str, int]] = [
    # ── Relevant: GIS & spatial analysis ────────────────────────────────────
    ("Consultant to provide GIS analysis, spatial data management, and interactive web mapping services", 1),
    ("Develop or maintain a geospatial database, web mapping application, or spatial decision-support tool", 1),

    # ── Relevant: Climate adaptation ─────────────────────────────────────────
    ("Consultant to prepare a municipal climate adaptation plan assessing flood, wildfire, drought, and heat risks", 1),
    ("Conduct a climate hazard and vulnerability assessment using spatial analysis for a municipality or region", 1),
    ("Greenhouse gas emissions inventory, carbon reduction strategy, or GHG accounting consulting services", 1),

    # ── Relevant: Environmental planning ─────────────────────────────────────
    ("Environmental impact assessment, ecological modelling, or sustainability analysis consulting study", 1),

    # ── Relevant: Urban & land use planning ──────────────────────────────────
    ("Consultant to develop an official community plan, land use policy, or growth management strategy", 1),
    ("Community planning, neighbourhood design study, or urban policy analysis and engagement consulting", 1),

    # ── Relevant: Transportation planning (studies and consulting, not construction) ──
    ("Transportation demand management study or active transportation network planning consulting services", 1),
    ("Transit system planning, network analysis, or mobility feasibility study for a transit authority", 1),
    ("Wayfinding system design and feasibility study or transit network navigation strategy consulting", 1),

    # ── Relevant: Data platforms & tools ─────────────────────────────────────
    ("Design and develop a data visualization dashboard or geospatial web application for decision support", 1),
    ("Data analytics platform, custom reporting tool, or mapping software development consulting", 1),

    # ── Relevant: Demographics & population ──────────────────────────────────
    ("Population and housing needs assessment, demographic forecasting, or growth projection study", 1),
    ("School enrollment forecasting or student population projection study for a school district", 1),

    # ── Relevant: Site analysis & spatial optimization ───────────────────────
    ("Spatial site selection study or facility location optimization analysis using GIS criteria", 1),

    # ── Relevant: Remote sensing ──────────────────────────────────────────────
    ("Remote sensing analysis, satellite imagery interpretation, or land cover change detection study", 1),

    # ── Relevant: Parks & open space planning ────────────────────────────────
    ("Parks, trails, and open space master plan or recreation facility network planning consulting", 1),

    # ── Not relevant: physical construction & trades ─────────────────────────
    ("Physical construction, demolition, or installation of buildings or structures", 0),
    ("Facility maintenance, repair, renovation, or trades work", 0),
    ("Roofing replacement, roof repair, or building envelope trades work", 0),
    ("Road paving, asphalt resurfacing, or pavement rehabilitation construction work", 0),
    ("Supply and installation of playground equipment or park recreational structures", 0),
    ("Boiler, chiller, sprinkler system, hot water tank, or building mechanical and piping installation and repair", 0),
    ("Diesel generator supply, backup power equipment, or emergency generator installation", 0),

    # ── Not relevant: equipment, vehicles & fleet ─────────────────────────────
    ("Supply or procurement of equipment, vehicles, hardware, or machinery", 0),
    ("Refuse collection, waste hauling, or plow truck and heavy equipment fleet procurement", 0),

    # ── Not relevant: environmental/physical field services ──────────────────
    ("Decontamination, remediation, or environmental cleanup physical services", 0),
    ("Physical tree planting, landscaping, or grounds maintenance", 0),
    ("Physical tree removal, hazard tree felling, brushing, or silviculture forestry work", 0),
    ("Archaeological impact assessment or heritage resource fieldwork services", 0),
    ("Aerial photography, photogrammetry image capture, or professional photography services", 0),

    # ── Not relevant: traffic hardware ────────────────────────────────────────
    ("Traffic signal installation, traffic control devices, road marking, or signage hardware supply", 0),

    # ── Not relevant: professional services outside LGeo's scope ─────────────
    ("Financial audit, accounting, or bookkeeping professional services", 0),
    ("Janitorial, security guard, or catering services", 0),
]

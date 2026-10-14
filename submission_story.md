FloodLens: Project Journey & Submission Story


What Inspired Us
Disaster response in mountainous and remote river corridors often fails not because of a lack of satellite data, but because of latency and fragility in spatial pipelines. During emergency windows, analysts face rigid API timeouts, missing pre-event road infrastructure data, and fragmented water bodies that obscure true flood impact.

We were inspired to build FloodLens to bridge the gap between raw Sentinel-1 synthetic aperture radar (SAR) imagery and actionable local intelligence. Our goal was to create an automated, end-to-end resilient system that marries AI-driven water extraction with robust infrastructure safety caching, ensuring that disaster teams get reliable impact assessments even when network conditions fail.


How We Built It
FloodLens combines geospatial signal processing, machine learning, and resilient local engineering into a cohesive pipeline:

Core Processing & AI Engine: We integrated a hybrid approach combining custom rule-based change detection with fine-tuned U-Net models for water body identification, paired with HAND (Height Above Nearest Drainage) filters using pysheds to filter out false-positive terrain anomalies.

Resilient Infrastructure Data (Overpass Caching): To conquer API rate-limits and server blocks (like 504 Gateway Timeouts), we engineered an intelligent Overpass cache-warming script. It splits large geographic areas of interest (AOIs) into custom 11km tiles, uses exponential backoff, and saves immutable historical snapshots locally.

User Interface & Dashboard: We built a fully interactive frontend using Streamlit (app/frontend/Home.py) to give responders a clean dashboard for viewing flood spreads, infrastructure impact, and validation metrics.


Challenges We Faced
API Rate Limits & Timeouts: Overpass and geospatial servers frequently threw connection errors or blocked requests under heavy hackathon workloads. We solved this by implementing smart grid tile-splitting and strict local caching to make executions fully idempotent.

Narrow Mountain Channel Detection: Early tests showed the U-Net model excelled at capturing broad water pools but struggled along narrow 30m mountain debris corridors. We honestly documented this limitation and balanced our pipeline fallback rules to handle regional variations transparently.

Environment & Dependency Clashes: Managing specific scientific libraries (numpy, pysheds) under tight time constraints required strict version pinning and isolating environment configurations.


What We Learned
Resilience Over Feature Creep: In emergency or hackathon scenarios, making existing pipelines bulletproof (like handling server timeouts gracefully) matters infinitely more than adding speculative features.

Honest Validation: Real-world geospatial models have blind spots (such as narrow channel misses). Embracing these findings through transparent metrics comparisons (against EMSR datasets) yields much stronger system credibility.

Smart Caching Saves Deadlines: Implementing robust caching layers early prevents last-minute panic when external APIs drop offline.
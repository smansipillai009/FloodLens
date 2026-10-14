# validation_only/  — CHECK-ONLY data

Put EMSR927 / UNOSAT / any published damage map and any post-event OSM extract **here only**.
* Nothing in `app/` may reference this folder or "EMSR927" (enforced by `tests/test_compliance.py`).
* Run `python scripts/freeze_config.py` and commit `configs/config.lock` **before** you open these files.
* `compare_emsr927.py` refuses to run if the config changed after the freeze.
* Credit: "European Union, Copernicus Emergency Management Service data."

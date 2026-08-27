# Schema Rules

`strip_none` must be applied to every LLM output dict before downstream consumption — `dict.get(key, default)` returns `None` (not `default`) when the key exists but its value is `None`, causing `None.get(...)` crashes downstream.

`_repair_json` handles common LLM JSON errors (trailing commas, missing left-quote on string values) iteratively before falling back to defaults. Fallback is never silent — `parse_method` records which strategy fired.

`OutputValidator` coerces types (scores → int, items → list) and fills Pydantic defaults; `model_dump(exclude_none=True)` keeps output consistent with `strip_none`.

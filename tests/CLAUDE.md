# Test Rules

Tests protect the current contract, not historical implementation. Priority: deterministic unit → graph routing → service integration → targeted end-to-end.

Agent tests mock model/tool calls and test prompt inputs, parsed outputs, routing, fallback behavior, and context projection. Do not assert brittle full prompt strings when behavior can be tested semantically.

Every Context projection needs tests proving: required info is included, irrelevant info is excluded, and full State does not leak through fallbacks.

Style tests use Chinese fixtures covering natural, fragmented, dialogue-heavy, action-heavy, and mixed prose — test false positives explicitly.

When fixing a bug, add the smallest regression test that would fail before the fix. When production code is deleted as obsolete, delete tests/fixtures that only protect that obsolete behavior.

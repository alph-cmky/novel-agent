# Graph Rules

Before adding a `NovelState` field, verify:

1. the value does not already exist in State or Storage;
2. it cannot be recomputed;
3. it must survive checkpoint/resume.

Do not store large durable objects (full chapter versions, world snapshots, derived context) in State when they already exist in Storage — store identifiers and compact decision metadata instead.

# Storage Rules

Pre-production: no production database requiring backward migration. When a schema contract changes, update the schema + tests/fixtures and remove obsolete migration/backfill logic — do not add compatibility migrations for hypothetical old versions.

Storage stores identifiers, current workflow state, and compact decision metadata — not repeated full chapter versions, world snapshots, or large derived context (those live in State or are recomputed).

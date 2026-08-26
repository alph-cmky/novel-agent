"""Tests for world_relations (edge) persistence."""

import pytest

from novel_agent.storage.manager import ProjectManager


@pytest.fixture
def manager(tmp_path):
    return ProjectManager(tmp_path)


@pytest.fixture
def project_id(manager):
    return manager.init_project(name="test")


def _report(entities: list[dict]) -> dict:
    """Wrap entities into a worldbuilding_report shape."""
    return {"new_entities": entities}


def _entity(name: str, relationships: list[dict]) -> dict:
    return {
        "name": name,
        "entity_type": "character",
        "properties": {},
        "relationships": relationships,
    }


def test_save_world_relations_persists_edges(manager, project_id):
    report = _report(
        [
            _entity(
                "Alice",
                [
                    {"target": "Bob", "relation": "friend"},
                    {"target": "Hogwarts", "relation": "attends"},
                ],
            ),
            _entity("Bob", [{"target": "Alice", "relation": "rival"}]),
        ]
    )

    manager.save_world_relations(project_id, 1, report)

    rows = manager.get_all_world_relations(project_id)
    assert len(rows) == 3
    by_pair = {(r["source"], r["target"], r["relation_type"]) for r in rows}
    assert ("Alice", "Bob", "friend") in by_pair
    assert ("Alice", "Hogwarts", "attends") in by_pair
    assert ("Bob", "Alice", "rival") in by_pair


def test_save_world_relations_dedups_and_keeps_first_appearance(manager, project_id):
    report = _report(
        [
            _entity("Alice", [{"target": "Bob", "relation": "friend"}]),
        ]
    )

    manager.save_world_relations(project_id, 1, report)
    manager.save_world_relations(project_id, 5, report)  # same edge appears again later

    rows = manager.get_all_world_relations(project_id)
    assert len(rows) == 1
    assert rows[0]["first_appearance_chapter"] == 1


def test_save_world_relations_skips_empty_and_malformed(manager, project_id):
    # No entities → no edges
    assert manager.save_world_relations(project_id, 1, _report([])) == 0
    # Missing target / non-dict relationship → skipped
    report = _report(
        [
            _entity("Alice", [{"target": ""}]),
            _entity("Bob", "not-a-list"),
        ]
    )
    manager.save_world_relations(project_id, 1, report)
    assert manager.get_all_world_relations(project_id) == []


def test_get_all_world_relations_orders_by_first_appearance(manager, project_id):
    manager.save_world_relations(
        project_id,
        3,
        _report(
            [
                _entity("Carol", [{"target": "Dave", "relation": "knows"}]),
            ]
        ),
    )
    manager.save_world_relations(
        project_id,
        1,
        _report(
            [
                _entity("Alice", [{"target": "Bob", "relation": "friend"}]),
            ]
        ),
    )

    rows = manager.get_all_world_relations(project_id)
    assert [r["first_appearance_chapter"] for r in rows] == [1, 3]

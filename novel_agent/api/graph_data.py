"""Build graph visualization data from stored world entities and relationships."""

import json

from novel_agent.storage.manager import ProjectManager

# Shape mapping for Cytoscape.js rendering
ENTITY_SHAPES = {
    "character": "ellipse",
    "location": "diamond",
    "item": "rectangle",
    "faction": "hexagon",
    "organization": "hexagon",  # legacy alias
    "rule": "round-rectangle",
    "event": "triangle",
    "unknown": "ellipse",
}

# Color mapping
ENTITY_COLORS = {
    "character": "#4a90d9",
    "location": "#67c23a",
    "item": "#e6a23c",
    "faction": "#7b4dd3",
    "organization": "#7b4dd3",  # legacy alias
    "rule": "#e040fb",
    "event": "#f56c6c",
    "unknown": "#909399",
}


def build_graph_data(
    mgr: ProjectManager,
    project_id: str,
    until_chapter: int = 0,
) -> dict:
    """Build nodes and edges for Cytoscape.js visualization.

    Args:
        mgr: ProjectManager instance.
        project_id: Project to query.
        until_chapter: If > 0, only include entities introduced up to this chapter.

    Returns:
        {"nodes": [...], "edges": [...]}
    """
    entities = mgr.get_all_world_entities(project_id)
    chapters = mgr.get_all_chapters(project_id)

    # Filter by chapter if specified
    if until_chapter > 0:
        entities = [
            e for e in entities
            if e.get("first_appearance_chapter", 999) <= until_chapter
        ]

    # Collect conflicts from worldbuilding reports
    conflict_names: set[str] = set()
    relationships: list[dict] = []

    for ch in chapters:
        if until_chapter > 0 and ch["chapter_number"] > until_chapter:
            break
        try:
            wb = json.loads(ch.get("worldbuilding_report", "{}"))
        except (json.JSONDecodeError, TypeError):
            wb = {}

        # Extract relationships from entities (agent nests them inside new_entities)
        for entity in wb.get("new_entities", []):
            entity_name = entity.get("name", "")
            for rel in entity.get("relationships", []):
                relationships.append({
                    "source": entity_name,
                    "target": rel.get("target", ""),
                    "relationship_type": rel.get("relation", "related_to"),
                    "description": rel.get("relation", ""),
                })

        for conflict in wb.get("conflicts", []):
            for name in conflict.get("entity_names", []):
                conflict_names.add(name)
            for name in conflict.get("entities", []):
                if isinstance(name, str):
                    conflict_names.add(name)
            # Also mark entities mentioned in conflict description
            existing = conflict.get("existing_entity", "")
            if existing:
                conflict_names.add(existing)

    # Build node list
    # Count connections per entity for importance
    connection_counts: dict[str, int] = {}
    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        if src:
            connection_counts[src] = connection_counts.get(src, 0) + 1
        if tgt:
            connection_counts[tgt] = connection_counts.get(tgt, 0) + 1

    nodes = []
    for e in entities:
        name = e.get("name", "?")
        etype = e.get("entity_type", "unknown")
        try:
            props = json.loads(e.get("properties", "{}"))
        except (json.JSONDecodeError, TypeError):
            props = {}

        importance = 1 + connection_counts.get(name, 0)
        nodes.append({
            "id": f"{etype}:{name}",
            "label": name,
            "type": etype,
            "properties": props,
            "first_chapter": e.get("first_appearance_chapter", 0),
            "importance": importance,
            "has_conflict": name in conflict_names,
            "shape": ENTITY_SHAPES.get(etype, "ellipse"),
            "color": ENTITY_COLORS.get(etype, "#909399"),
        })

    # Build edge list
    edges = []
    seen_edge_ids: set[str] = set()
    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rtype = rel.get("relationship_type", rel.get("type", "related_to"))
        description = rel.get("description", "")
        edge_id = f"{src}->{tgt}:{rtype}"
        if edge_id in seen_edge_ids:
            continue
        seen_edge_ids.add(edge_id)

        src_id = _find_entity_id(entities, src)
        tgt_id = _find_entity_id(entities, tgt)
        if not src_id or not tgt_id:
            continue

        edges.append({
            "id": edge_id,
            "source": f"{_entity_type(entities, src)}:{src}",
            "target": f"{_entity_type(entities, tgt)}:{tgt}",
            "label": rtype,
            "description": description,
            "relationship_type": rtype,
        })

    return {"nodes": nodes, "edges": edges}


def _find_entity_id(entities: list[dict], name: str) -> str | None:
    for e in entities:
        if e.get("name") == name:
            return e.get("id")
    return None


def _entity_type(entities: list[dict], name: str) -> str:
    for e in entities:
        if e.get("name") == name:
            return e.get("entity_type", "unknown")
    return "unknown"

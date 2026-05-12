import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class Neo4jClient:
    def __init__(self):
        self._driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        self._database = os.getenv("NEO4J_DATABASE", "neo4j")

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def run(self, cypher: str, params: dict = None) -> list[dict]:
        with self._driver.session(database=self._database) as session:
            return session.run(cypher, params or {}).data()

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_constraints(self):
        stmts = [
            "CREATE CONSTRAINT ent_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        ]
        for s in stmts:
            try:
                self.run(s)
            except Exception:
                pass

    # ── Write ─────────────────────────────────────────────────────────────────

    def merge_entity(self, eid: str, etype: str, name: str, source_pdf: str):
        self.run(
            """
            MERGE (e:Entity {id: $id})
            SET e.type = $type, e.name = $name, e.source_pdf = $source_pdf
            """,
            {"id": eid, "type": etype, "name": name, "source_pdf": source_pdf},
        )

    def merge_chunk(self, chunk_id: str, text: str, source_pdf: str):
        self.run(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.text = $text, c.source_pdf = $source_pdf
            """,
            {"chunk_id": chunk_id, "text": text, "source_pdf": source_pdf},
        )

    def merge_relation(self, src_id: str, tgt_id: str, relation: str):
        rel_type = relation.upper().replace(" ", "_").replace("-", "_")
        self.run(
            f"""
            MATCH (a:Entity {{id: $src}})
            MATCH (b:Entity {{id: $tgt}})
            MERGE (a)-[:{rel_type}]->(b)
            """,
            {"src": src_id, "tgt": tgt_id},
        )

    def link_entity_to_chunk(self, eid: str, chunk_id: str):
        self.run(
            """
            MATCH (e:Entity {id: $eid})
            MATCH (c:Chunk {chunk_id: $cid})
            MERGE (e)-[:FOUND_IN]->(c)
            """,
            {"eid": eid, "cid": chunk_id},
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_subgraph_for_entities(self, entity_names: list[str], limit: int = 60) -> dict:
        rows = self.run(
            """
            MATCH (e:Entity)
            WHERE any(name IN $names WHERE toLower(e.name) CONTAINS toLower(name))
            WITH e LIMIT $limit
            OPTIONAL MATCH (e)-[r]->(nb:Entity)
            RETURN
              e.id AS eid, e.name AS ename, e.type AS etype, e.source_pdf AS epdf,
              type(r) AS rtype,
              nb.id AS nid, nb.name AS nname, nb.type AS ntype, nb.source_pdf AS npdf
            """,
            {"names": entity_names, "limit": limit},
        )
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for row in rows:
            if row["eid"]:
                nodes[row["eid"]] = {
                    "id": row["eid"],
                    "name": row["ename"],
                    "type": row.get("etype") or "concept",
                    "source_pdf": row.get("epdf") or "",
                }
            if row["nid"]:
                nodes[row["nid"]] = {
                    "id": row["nid"],
                    "name": row["nname"],
                    "type": row.get("ntype") or "concept",
                    "source_pdf": row.get("npdf") or "",
                }
            if row.get("rtype") and row["eid"] and row["nid"]:
                edges.append({"source": row["eid"], "target": row["nid"], "relation": row["rtype"]})
        return {"nodes": list(nodes.values()), "edges": edges}

    def get_all_entity_names(self) -> list[str]:
        rows = self.run("MATCH (e:Entity) RETURN e.name AS name")
        return [r["name"] for r in rows if r.get("name")]

    # ── Admin ─────────────────────────────────────────────────────────────────

    def clear_all(self):
        self.run("MATCH (n) DETACH DELETE n")

    def node_count(self) -> int:
        rows = self.run("MATCH (n) RETURN count(n) AS c")
        return rows[0]["c"] if rows else 0

    def edge_count(self) -> int:
        rows = self.run("MATCH ()-[r]->() RETURN count(r) AS c")
        return rows[0]["c"] if rows else 0
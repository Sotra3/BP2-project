import os
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class Attribute:
    name: str
    type: str
    unique: bool
    distinct_values: int

@dataclass
class IndexInfo:
    name: str
    attributes: list[str]
    index_type: str
    clustered: bool
    tree_height: Optional[int] = None

@dataclass
class Table:
    name: str
    row_count: int
    block_count: int
    rows_per_block: int
    attributes: list[Attribute]
    indexes: list[IndexInfo]

    def get_attribute(self, name: str) -> Optional[Attribute]:
        for a in self.attributes:
            if a.name == name:
                return a
        return None

    def get_index_for(self, attr_names: list[str]) -> Optional[IndexInfo]:
        for idx in self.indexes:
            if idx.attributes == attr_names:
                return idx
        return None

@dataclass
class Schema:
    buffer_blocks: int
    tables: dict[str, Table]

    @staticmethod
    def load(path: str) -> "Schema":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        tables = {}
        for t in data["schema"]["tables"]:
            attrs = [Attribute(a["name"], a["type"], a["unique"], a["distinctValues"])
                     for a in t["attributes"]]
            idxs = [IndexInfo(i["name"], i["attributes"], i["type"],
                               i["clustered"], i.get("treeHeight"))
                    for i in t["indexes"]]
            tables[t["name"]] = Table(t["name"], t["rowCount"], t["blockCount"],
                                       t["rowsPerBlock"], attrs, idxs)

        return Schema(data["bufferBlocks"], tables)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "input", "primer_ulaza.json")
    schema = Schema.load(path)
    print(schema.tables["Student"])
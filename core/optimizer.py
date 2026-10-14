import math

from core.operations import SelectionOp, JoinOp, SortOp, ProjectionOp
from core.cost_formulas import size_conjunction


class SelectedTable:
    """Rezultat primene (jednog ili vise) WHERE uslova nad baznom tabelom.

    Po pravilu A7 (BP2 - 5, slajd 8): bira se uslov i algoritam sa
    najmanjom cenom kao access path, a preostali uslovi se testiraju nad
    vec dohvacenim redovima bez dodatne cene (pipelining). Konacna
    procena broja redova ipak uzima u obzir SVE uslove konjunkcije
    (size_conjunction), ne samo access uslov, da bi downstream spajanja i
    projekcija imali tacniju procenu velicine.
    """

    def __init__(self, base_table, access_op, row_count):
        self.base_table = base_table
        self.access_op = access_op
        self.rows_per_block = base_table.rows_per_block
        self.row_count = max(1, round(row_count))
        self.block_count = max(1, math.ceil(self.row_count / self.rows_per_block))
        self.cost = access_op.cost

    def get_attribute(self, name):
        return self.base_table.get_attribute(name)

    def get_index_for(self, attr_names):
        return self.base_table.get_index_for(attr_names)


class Optimizer:
    def __init__(self, schema, query):
        self.schema = schema
        self.query = query
        self.buffer_size = schema.buffer_blocks
        self.steps = []

    def build_plan(self):
        # 1. selekcije - za svaku tabelu primeni SVE lokalne (ne-join) uslove
        current_tables = {}
        for alias, table_name in self.query.table_aliases.items():
            table = self.schema.tables[table_name]
            local_conditions = [c for c in self.query.conditions
                                 if not c.is_join_condition() and c.left_table == alias]

            current_tables[alias] = self._apply_selections(table, local_conditions)

        # 2. spajanja - redosled tabela se odredjuje topoloski po grafu
        # join uslova (svaka sledeca tabela mora biti direktno povezana sa
        # vec spojenim skupom), a ne slepo po redosledu iz FROM klauzule.
        join_conditions = [c for c in self.query.conditions if c.is_join_condition()]
        aliases = list(current_tables.keys())
        join_order = self._topological_join_order(aliases, join_conditions)

        result = current_tables[join_order[0]]
        joined_aliases = [join_order[0]]

        for next_alias in join_order[1:]:
            next_table = current_tables[next_alias]
            matching_cond = self._find_join_condition(join_conditions, joined_aliases, next_alias)
            if matching_cond is None:
                raise ValueError(
                    f"Tabela '{next_alias}' nije povezana join uslovom sa "
                    f"prethodno spojenim tabelama {joined_aliases} "
                    f"(Dekartov proizvod nije podržan)"
                )
            best_op = self._choose_best_join(result, next_table, matching_cond)
            self.steps.append(best_op)
            result = best_op
            joined_aliases.append(next_alias)

        # 3. sortiranje, ako postoji ORDER BY
        if self.query.order_by:
            alias, attr = self.query.order_by
            sort_op = SortOp(result, attr, self.buffer_size)
            self.steps.append(sort_op)
            result = sort_op

        # 4. projekcija - prosledjuju se samo imena atributa (ne (alias, attr)
        # parovi), kako bi provera kandidat-kljuca u ProjectionOp radila
        select_attr_names = [attr for _, attr in self.query.select_attrs]
        proj_op = ProjectionOp(result, select_attr_names, self.buffer_size)
        self.steps.append(proj_op)

        return self.steps

    def _apply_selections(self, table, local_conditions):
        if not local_conditions:
            return table

        # Kandidat operacije za svaki uslov (nad baznom tabelom), da bismo
        # odredili koji je pojedinacno najjeftiniji (access path).
        per_condition_ops = [self._choose_best_selection(table, c) for c in local_conditions]
        access_op = min(per_condition_ops, key=lambda op: op.cost)
        self.steps.append(access_op)

        if len(local_conditions) == 1:
            return SelectedTable(table, access_op, access_op.row_count)

        # Selektivnost svakog uslova (nezavisno od access uslova) racunamo
        # ponovnom upotrebom SelectionOp._compute_output_size logike (preko
        # linear_scan varijante - algoritam ne utice na procenu velicine,
        # samo na cenu, a cenu access uslova vec imamo iz access_op).
        selectivities = []
        for cond in local_conditions:
            probe = SelectionOp(table, cond, "linear_scan")
            selectivities.append(probe.row_count / table.row_count)

        combined_row_count = size_conjunction(table.row_count, selectivities)
        return SelectedTable(table, access_op, combined_row_count)

    def _choose_best_selection(self, table, condition):
        candidates = [SelectionOp(table, condition, "linear_scan")]

        idx = table.get_index_for([condition.left_attr])
        op = condition.operator

        if idx and op == "=":
            if idx.index_type == "HASH":
                candidates.append(SelectionOp(table, condition, "secondary_index"))
            elif idx.index_type == "B_PLUS_TREE":
                algorithm = "clustering_index" if idx.clustered else "secondary_index"
                candidates.append(SelectionOp(table, condition, algorithm))

        elif idx and op in ("<", "<=", ">", ">="):
            # Hes indeks ne podrzava poredjenje (samo jednakost) - vidi
            # operations.py._comparison_index_cost - pa ga ovde ne nudimo
            # kao kandidata da izbegnemo nepotrebnu gresku.
            if idx.index_type == "B_PLUS_TREE":
                algorithm = "clustering_index" if idx.clustered else "secondary_index"
                candidates.append(SelectionOp(table, condition, algorithm))

        # Za "!=" (negaciju) po slajdu 9 se koristi linearno skeniranje
        # osim ako je rezultat vrlo mali i postoji indeks - ovo pojednostavljenje
        # namerno ostavlja samo linear_scan kandidata.

        return min(candidates, key=lambda op: op.cost)

    def _choose_best_join(self, left, right, condition):
        algorithms = ["nested_loop", "block_nested_loop", "merge_join", "hash_join"]

        idx = right.get_index_for([condition.right_attr]) if condition and hasattr(right, "get_index_for") else None
        if idx:
            algorithms.append("indexed_nested_loop")

        candidates = [JoinOp(left, right, condition, alg, self.buffer_size) for alg in algorithms]
        return min(candidates, key=lambda op: op.cost)

    def _find_join_condition(self, join_conditions, known_aliases, next_alias):
        for cond in join_conditions:
            if cond.left_table in known_aliases and cond.right_table == next_alias:
                return cond
            if cond.right_table in known_aliases and cond.left_table == next_alias:
                return cond
        return None

    def _topological_join_order(self, aliases, join_conditions):
        """Odredjuje redosled obradjivanja tabela tako da je svaka sledeca
        tabela direktno povezana join uslovom sa vec obradjenim skupom.
        Time se izbegava situacija da _find_join_condition ne pronadje
        odgovarajuci uslov usled proizvoljnog FROM redosleda."""
        order = [aliases[0]]
        remaining = set(aliases[1:])

        while remaining:
            next_alias = None
            for alias in remaining:
                if self._find_join_condition(join_conditions, order, alias) is not None:
                    next_alias = alias
                    break

            if next_alias is None:
                raise ValueError(
                    f"Tabele {sorted(remaining)} nisu povezane join uslovom sa "
                    f"ostatkom upita {order} (Dekartov proizvod nije podržan)"
                )

            order.append(next_alias)
            remaining.remove(next_alias)

        return order
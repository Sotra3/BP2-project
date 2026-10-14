import math

from core.cost_formulas import (
    linear_scan_cost, clustering_index_key_cost, clustering_index_nonkey_cost,
    secondary_index_cost, nested_loop_cost, block_nested_loop_cost,
    indexed_nested_loop_cost, merge_join_cost, hash_join_cost, external_sort_cost,
    hash_index_cost, effective_tree_height, join_size_key_reference, join_size_general,
    distinct_values_after_join
)


class SelectionOp:
    def __init__(self, table, condition, algorithm):
        self.table = table
        self.condition = condition
        self.algorithm = algorithm
        self.rows_per_block = table.rows_per_block
        self.row_count = self._compute_output_size()
        self.block_count = max(1, math.ceil(self.row_count / self.rows_per_block))
        self.cost = self._compute_cost()

    def _compute_output_size(self):
        attr = self.table.get_attribute(self.condition.left_attr)
        n_r = self.table.row_count
        op = self.condition.operator

        if op == "=":
            if attr.unique:
                return 1
            return max(1, round(n_r / attr.distinct_values))

        if op == "!=":
            equal_size = 1 if attr.unique else max(1, round(n_r / attr.distinct_values))
            return max(1, n_r - equal_size)

        # Poređenje (<, <=, >, >=): u šemi nemamo min(A,r)/max(A,r), pa po
        # uputstvu sa slajda 25 ("U nedostatku statistickih informacija,
        # pretpostavlja se da je c = n_r/2") koristimo n_r/2 kao procenu.
        return max(1, round(n_r / 2))

    def _compute_cost(self):
        attr = self.table.get_attribute(self.condition.left_attr)
        op = self.condition.operator

        if self.algorithm == "linear_scan":
            is_key_equality = attr.unique and op == "="
            return linear_scan_cost(self.table.block_count, is_key=is_key_equality)

        idx = self.table.get_index_for([self.condition.left_attr])
        if idx is None:
            raise ValueError(f"Ne postoji indeks nad atributom {self.condition.left_attr}")

        if op == "=":
            return self._equality_index_cost(attr, idx)

        if op in ("<", "<=", ">", ">="):
            return self._comparison_index_cost(attr, idx)

        raise ValueError(
            f"Algoritam '{self.algorithm}' ne podržava operator '{op}'"
        )

    def _equality_index_cost(self, attr, idx):
        if self.algorithm == "clustering_index":
            h_i = effective_tree_height(idx.tree_height)
            if attr.unique:
                return clustering_index_key_cost(h_i)
            b = max(1, math.ceil(self.row_count / self.table.rows_per_block))
            return clustering_index_nonkey_cost(h_i, b)

        if self.algorithm == "secondary_index":
            if idx.index_type == "HASH":
                n = 1 if attr.unique else self.row_count
                return hash_index_cost(n)
            h_i = effective_tree_height(idx.tree_height)
            return secondary_index_cost(h_i, self.row_count, is_key=attr.unique)

        raise ValueError(f"Nepoznat algoritam selekcije: {self.algorithm}")

    def _comparison_index_cost(self, attr, idx):
        # NAPOMENA: slajdovi (BP2 - 5, slajd 7) opisuju A5/A6 samo
        # algoritamski (pronaci pocetnu tacku indeksom, potom sekvencijalno
        # skenirati), bez posebne formule cene. Mehanizam je strukturno
        # isti kao A3 (grupišući indeks) odnosno A4 (sekundarni indeks) za
        # ne-ključni atribut - razlika je samo u tome kako je procenjena
        # velicina rezultata (ovde preko n_r/2 umesto V(A,r)). Zbog toga
        # ovde ponovo koristimo formule A3/A4 sa vec izracunatim
        # self.row_count kao "b" odnosno "n".
        if idx.index_type == "HASH":
            raise ValueError(
                "Heš indeks ne podržava poređenje (samo jednakost), "
                "koristiti B+ stablo ili linear_scan"
            )

        h_i = effective_tree_height(idx.tree_height)

        if self.algorithm == "clustering_index":
            b = max(1, math.ceil(self.row_count / self.table.rows_per_block))
            return clustering_index_nonkey_cost(h_i, b)

        if self.algorithm == "secondary_index":
            return secondary_index_cost(h_i, self.row_count, is_key=False)

        raise ValueError(f"Nepoznat algoritam selekcije: {self.algorithm}")


class JoinOp:
    def __init__(self, left, right, condition, algorithm, buffer_size):
        self.left = left
        self.right = right
        self.condition = condition
        self.algorithm = algorithm
        self.buffer_size = buffer_size
        # Širina spojenog reda je >= širine reda svake od strana pojedinačno,
        # pa je broj redova po bloku posle spajanja <= min(rows_per_block od
        # obe strane). Bez tačne veličine reda u bajtovima (šema je ne daje)
        # ovo je najbolja dostupna aproksimacija - konzervativnija je od
        # pukog preuzimanja rows_per_block desne tabele.
        self.rows_per_block = min(left.rows_per_block, right.rows_per_block)
        self.row_count = self._compute_output_size()
        self.block_count = max(1, math.ceil(self.row_count / self.rows_per_block))
        # V(A,r) procene za atribute spajanja, koje kasniji JoinOp u lancu
        # spajanja (npr. treca/cetvrta tabela u upitu) moze iskoristiti
        # preko get_distinct_values(), umesto da pretpostavi n_r.
        self.distinct_values = self._compute_distinct_values()
        self.cost = self._compute_cost()

    def get_distinct_values(self, attr_name):
        """Vraca procenu V(attr_name, self) - najpre iz sacuvanih procena
        za atribute spajanja, zatim iz bazne tabele ako je left/right
        bazna tabela, u suprotnom konzervativno self.row_count."""
        if attr_name in self.distinct_values:
            return self.distinct_values[attr_name]
        for side in (self.left, self.right):
            if hasattr(side, "get_attribute"):
                attr = side.get_attribute(attr_name)
                if attr is not None:
                    return min(attr.distinct_values, self.row_count)
        return self.row_count

    def _compute_distinct_values(self):
        result = {}
        v_left = self._side_distinct_values(self.left, self.condition.left_attr)
        v_right = self._side_distinct_values(self.right, self.condition.right_attr)
        if v_left is not None:
            result[self.condition.left_attr] = distinct_values_after_join(
                v_left, self.row_count
            )
        if v_right is not None:
            result[self.condition.right_attr] = distinct_values_after_join(
                v_right, self.row_count
            )
        return result

    @staticmethod
    def _side_distinct_values(side, attr_name):
        if hasattr(side, "get_attribute"):
            attr = side.get_attribute(attr_name)
            return attr.distinct_values if attr is not None else None
        if hasattr(side, "get_distinct_values"):
            return side.get_distinct_values(attr_name)
        return None

    def _compute_output_size(self):
        left_has_attrs = hasattr(self.left, "get_attribute")
        right_has_attrs = hasattr(self.right, "get_attribute")

        if left_has_attrs and right_has_attrs:
            la = self.left.get_attribute(self.condition.left_attr)
            ra = self.right.get_attribute(self.condition.right_attr)

            if la.unique:
                return join_size_key_reference(self.left.row_count, self.right.row_count, True)
            if ra.unique:
                return join_size_key_reference(self.left.row_count, self.right.row_count, False)

            estimate = join_size_general(
                self.left.row_count, self.right.row_count,
                ra.distinct_values, la.distinct_values
            )
            return max(1, round(estimate))

        # Bar jedna strana je međurezultat prethodnog spajanja - koristimo
        # V(A,r) propagirano kroz distinct_values (ili get_distinct_values),
        # umesto pukog max(n_r, n_s) koji ignoriše selektivnost spajanja.
        v_left = self._side_distinct_values(self.left, self.condition.left_attr)
        v_right = self._side_distinct_values(self.right, self.condition.right_attr)

        if v_left is not None or v_right is not None:
            estimate = join_size_general(
                self.left.row_count, self.right.row_count,
                v_right if v_right is not None else self.right.row_count,
                v_left if v_left is not None else self.left.row_count,
            )
            return max(1, round(estimate))

        return max(self.left.row_count, self.right.row_count)

    def _lookup_cost(self, table, attr_name):
        if not hasattr(table, "get_index_for"):
            raise ValueError(
                "indexed_nested_loop zahteva da unutrašnja relacija bude "
                "bazna tabela sa indeksom (dobijen međurezultat bez indeksa)"
            )

        idx = table.get_index_for([attr_name])
        if idx is None:
            raise ValueError(f"Nema indeksa nad {attr_name}, indexed_nested_loop nije moguć")

        attr = table.get_attribute(attr_name)
        n_matches = 1 if attr.unique else max(1, round(table.row_count / attr.distinct_values))

        if idx.index_type == "HASH":
            return hash_index_cost(n_matches)

        h_i = effective_tree_height(idx.tree_height)
        if idx.clustered:
            if attr.unique:
                return clustering_index_key_cost(h_i)
            b = max(1, math.ceil(n_matches / table.rows_per_block))
            return clustering_index_nonkey_cost(h_i, b)

        return secondary_index_cost(h_i, n_matches, is_key=attr.unique)

    def _compute_cost(self):
        b_r = self.left.block_count
        b_s = self.right.block_count
        n_r = self.left.row_count

        if self.algorithm == "nested_loop":
            return nested_loop_cost(n_r, b_s, b_r)

        if self.algorithm == "block_nested_loop":
            return block_nested_loop_cost(b_r, b_s, self.buffer_size)

        if self.algorithm == "indexed_nested_loop":
            cost_per_lookup = self._lookup_cost(self.right, self.condition.right_attr)
            return indexed_nested_loop_cost(b_r, cost_per_lookup, n_r)

        if self.algorithm == "merge_join":
            return merge_join_cost(b_r, b_s)

        if self.algorithm == "hash_join":
            return hash_join_cost(b_r, b_s, self.buffer_size)

        raise ValueError(f"Nepoznat algoritam spajanja: {self.algorithm}")


class SortOp:
    def __init__(self, table, attr, buffer_size):
        self.table = table
        self.attr = attr
        self.buffer_size = buffer_size
        self.rows_per_block = table.rows_per_block
        self.row_count = table.row_count
        self.block_count = table.block_count
        self.cost = self._compute_cost()

    def _compute_cost(self):
        return external_sort_cost(self.table.block_count, self.buffer_size)


class ProjectionOp:
    def __init__(self, table, attributes, buffer_size):
        self.table = table
        self.attributes = attributes
        self.buffer_size = buffer_size
        self.rows_per_block = table.rows_per_block
        self.row_count = table.row_count
        self.block_count = table.block_count
        self.needs_dedup = not self._includes_candidate_key()
        self.cost = self._compute_cost()

    def _includes_candidate_key(self):
        if not hasattr(self.table, "attributes"):
            return False
        selected = set(self.attributes)
        for attr in self.table.attributes:
            if attr.unique and attr.name in selected:
                return True
        return False

    def _compute_cost(self):
        if not self.needs_dedup:
            return 0
        return external_sort_cost(self.block_count, self.buffer_size)
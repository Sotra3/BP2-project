import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Condition:
    left_table: str      # alias tabele, npr "s"
    left_attr: str        # npr "indeks"
    operator: str          # "=", ">", "<", ">=", "<=", "!="
    right_table: Optional[str]   # alias, ako je desna strana atribut (join uslov)
    right_attr: Optional[str]     # atribut, ako je join uslov
    right_value: Optional[str]     # literal (broj/string), ako je selekcija

    def is_join_condition(self) -> bool:
        return self.right_table is not None


@dataclass
class Query:
    select_attrs: list[tuple[str, str]]     # [(alias, atribut), ...]
    table_aliases: dict[str, str]            # {alias: pravo_ime_tabele}
    conditions: list[Condition]
    order_by: Optional[tuple[str, str]]       # (alias, atribut) ili None


class QueryParser:

    OPERATORS = ["<=", ">=", "!=", "=", "<", ">"]

    # alias.attribute mora da bude identifikator.identifikator -> ne hvata brojeve poput 8.5
    JOIN_REF_PATTERN = re.compile(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*$")

    def parse(self, sql_text: str) -> Query:
        sql_text = sql_text.strip().rstrip(";")

        select_part, from_part, where_part, order_part = self._split_clauses(sql_text)

        table_aliases = self._parse_from(from_part)
        select_attrs = self._parse_select(select_part, table_aliases)
        conditions = self._parse_where(where_part, table_aliases) if where_part else []
        order_by = self._parse_order_by(order_part, table_aliases) if order_part else None

        return Query(select_attrs, table_aliases, conditions, order_by)

    def _split_clauses(self, sql_text: str):
        upper = sql_text.upper()

        from_idx = upper.index("FROM")
        where_idx = upper.find("WHERE")
        order_idx = upper.find("ORDER BY")

        select_part = sql_text[len("SELECT"):from_idx].strip()

        if where_idx != -1:
            from_part = sql_text[from_idx + len("FROM"):where_idx].strip()
        elif order_idx != -1:
            from_part = sql_text[from_idx + len("FROM"):order_idx].strip()
        else:
            from_part = sql_text[from_idx + len("FROM"):].strip()

        where_part = None
        if where_idx != -1:
            end = order_idx if order_idx != -1 else len(sql_text)
            where_part = sql_text[where_idx + len("WHERE"):end].strip()

        order_part = None
        if order_idx != -1:
            order_part = sql_text[order_idx + len("ORDER BY"):].strip()

        return select_part, from_part, where_part, order_part

    def _resolve_unqualified(self, attr: str, table_aliases: dict[str, str]) -> str:
        """Vraca alias tabele kojoj neoznaceni atribut pripada, ako je to nedvosmisleno."""
        if len(table_aliases) == 1:
            return next(iter(table_aliases))
        raise ValueError(
            f"Atribut '{attr}' mora biti kvalifikovan tabelom/aliasom "
            f"jer upit obuhvata vise od jedne tabele: {list(table_aliases)}"
        )

    def _parse_select(self, select_part: str, table_aliases: dict[str, str]) -> list[tuple[str, str]]:
        result = []
        for item in select_part.split(","):
            item = item.strip()
            if "." in item:
                alias, attr = item.split(".")
                result.append((alias.strip(), attr.strip()))
            else:
                alias = self._resolve_unqualified(item, table_aliases)
                result.append((alias, item))
        return result

    def _parse_from(self, from_part: str) -> dict[str, str]:
        aliases = {}
        for item in from_part.split(","):
            # filtriramo eventualni "AS" da ne bude pogresno protumacen kao alias
            parts = [p for p in item.strip().split() if p.upper() != "AS"]
            table_name = parts[0]
            alias = parts[1] if len(parts) > 1 else table_name
            aliases[alias] = table_name
        return aliases

    def _parse_where(self, where_part: str, table_aliases: dict[str, str]) -> list[Condition]:
        conditions = []
        # case-insensitive deljenje po AND, otporno na visestruke razmake/prelome reda
        parts = re.split(r"\s+AND\s+", where_part.strip(), flags=re.IGNORECASE)
        for cond_text in parts:
            conditions.append(self._parse_condition(cond_text.strip(), table_aliases))
        return conditions

    def _parse_condition(self, cond_text: str, table_aliases: dict[str, str]) -> Condition:
        operator = None
        for op in self.OPERATORS:
            if op in cond_text:
                operator = op
                break

        if operator is None:
            raise ValueError(f"Nepoznat operator u uslovu: '{cond_text}'")

        # split sa maxsplit=1 da ne pukne ako se npr. '=' pojavi unutar string literala
        left, right = cond_text.split(operator, 1)
        left = left.strip()
        right = right.strip()

        if "." in left:
            left_table, left_attr = left.split(".")
            left_table = left_table.strip()
            left_attr = left_attr.strip()
        else:
            left_table = self._resolve_unqualified(left, table_aliases)
            left_attr = left

        if self.JOIN_REF_PATTERN.match(right):
            right_table, right_attr = right.split(".")
            return Condition(left_table, left_attr, operator,
                              right_table.strip(), right_attr.strip(), None)
        else:
            value = right.strip("'\"")
            return Condition(left_table, left_attr, operator, None, None, value)

    def _parse_order_by(self, order_part: str, table_aliases: dict[str, str]) -> tuple[str, str]:
        order_part = order_part.strip()
        if "." in order_part:
            alias, attr = order_part.split(".")
            return (alias.strip(), attr.strip())
        alias = self._resolve_unqualified(order_part, table_aliases)
        return (alias, order_part)


if __name__ == "__main__":
    query_text = "SELECT Student.ime, Predmet.naziv, Ispit.ocena FROM Student, Ispit, Predmet, Stipendija WHERE Student.indeks = Ispit.studentIndeks AND Ispit.predmetId = Predmet.predmetId AND Student.indeks = Stipendija.studentIndeks AND Ispit.ocena >= 9 ORDER BY Student.ime;"
    query = QueryParser().parse(query_text)
    print(query)

    # test koji je pre pucao: DOUBLE atribut
    query_text2 = "SELECT Student.ime FROM Student WHERE Student.prosek > 8.5 ORDER BY prosek;"
    query2 = QueryParser().parse(query_text2)
    print(query2)

    # test koji si upravo prijavio: nekvalifikovan atribut, jedna tabela
    query_text3 = "SELECT ime, smer FROM Student WHERE ime = 'Ana'"
    query3 = QueryParser().parse(query_text3)
    print(query3)
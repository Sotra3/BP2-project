from core.schema import Schema
from core.query import QueryParser
from core.optimizer import Optimizer
from output.plan_printer import print_plan

schema = Schema.load("input/primer_ulaza.json")

with open("input/queries/simple_10.txt", encoding="utf-8") as f:
    sql_text = f.read()
query = QueryParser().parse(sql_text)

optimizer = Optimizer(schema, query)
plan = optimizer.build_plan()

print_plan(plan)
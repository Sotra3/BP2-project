def print_plan(steps):
    total_cost = 0
    for i, step in enumerate(steps, 1):
        op_type = type(step).__name__
        algorithm = getattr(step, "algorithm", "-")
        print(f"{i}. {op_type} | algoritam: {algorithm} | cena: {step.cost}")
        total_cost += step.cost

    print(f"\nUkupna cena: {total_cost}")
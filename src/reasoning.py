"""Rule-based reasoning: first-order-logic knowledge base with forward and backward chaining."""


# ------------------------- first-order-logic knowledge base -------------------------
def make_rules(HM, HJ, BH, CH):
    """13 stratified Horn clauses. A condition is [attribute, op, value] or an atom ('~Atom' = negation as failure)."""
    R = [("R1", "PriorSuccess", [["poutcome", "==", "success"]], 0), ("R2", "HighSeason", [["month", "in", HM]], 0),
         ("R3", "TargetJob", [["job", "in", HJ]], 0), ("R4", "HighBalance", [["balance", ">=", BH]], 0),
         ("R5", "HasDebt", [["housing", "==", "yes"]], 0), ("R6", "HasDebt", [["loan", "==", "yes"]], 0),
         ("R7", "OverContacted", [["campaign", ">=", CH]], 0),
         ("R8", "Likely", ["PriorSuccess"], 1), ("R9", "Likely", ["HighSeason", "~HasDebt"], 1),
         ("R10", "Likely", ["TargetJob", "HighBalance"], 1), ("R11", "Unlikely", ["OverContacted"], 1),
         ("R12", "Unlikely", ["HasDebt", "~PriorSuccess", "~HighSeason"], 1),
         ("R13", "Recommend", ["Likely", "~Unlikely"], 2)]
    return [dict(id=i, head=h, cond=c, s=s) for i, h, c, s in R]


def _test(c, row, F):
    if isinstance(c, list):
        a, op, v = c; x = row[a]
        return x == v if op == "==" else (x in v if op == "in" else x >= v)
    return (c[1:] not in F) if c[0] == "~" else c in F


def forward_chain(rules, row):
    """Data-driven: fire rules stratum by stratum until no new fact appears. Returns (facts, rule ids fired in order)."""
    F, fired = set(), []
    for s in range(max(r["s"] for r in rules) + 1):
        changed = True
        while changed:
            changed = False
            for r in rules:
                if r["s"] == s and r["head"] not in F and all(_test(c, row, F) for c in r["cond"]):
                    F.add(r["head"]); fired.append(r["id"]); changed = True
    return F, fired


def fmt_cond(c):
    if isinstance(c, list):
        a, op, v = c; n = a.capitalize()
        return f"{n}(c, {v})" if op == "==" else (f"{n}(c) ∈ {{{', '.join(v)}}}" if op == "in" else f"{n}(c) ≥ {v:g}")
    return f"¬{c[1:]}(c)" if c[0] == "~" else f"{c}(c)"


def fol_rule(r):
    return f"∀c: {' ∧ '.join(fmt_cond(c) for c in r['cond'])} → {r['head']}(c)"


def prove(rules, goal, row, seen=()):
    """Goal-driven (backward chaining): returns a proof tree {goal, ok, kids}."""
    if goal[0] == "~":
        t = prove(rules, goal[1:], row, seen)
        return dict(goal=f"¬{goal[1:]}(c)  [negation as failure]", ok=not t["ok"], kids=[t])
    if goal in seen:
        return dict(goal=f"{goal}(c)", ok=False, kids=[])
    kids = []
    for r in (r for r in rules if r["head"] == goal):
        sub = [dict(goal=f"{fmt_cond(c)}  [c.{c[0]} = {row[c[0]]}]", ok=bool(_test(c, row, set())), kids=[])
               if isinstance(c, list) else prove(rules, c, row, seen + (goal,)) for c in r["cond"]]
        kids.append(dict(goal=f"by {r['id']}: {fol_rule(r)}", ok=all(k["ok"] for k in sub), kids=sub))
    return dict(goal=f"{goal}(c)", ok=any(k["ok"] for k in kids), kids=kids)


def proof_lines(t, depth=0):
    out = ["    " * depth + ("✔ " if t["ok"] else "✘ ") + t["goal"]]
    for k in t["kids"]:
        out += proof_lines(k, depth + 1)
    return out

import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

VAR_NAMES = ["x"]
MAX_NODES = 300
RNG_SEED = 11
np.seterr(all="ignore")

FUNC_SET = {
    "add": (lambda a, b: a + b, 2, "+"),
    "sub": (lambda a, b: a - b, 2, "-"),
    "mul": (lambda a, b: a * b, 2, "*"),
    "sq":  (lambda a: a * a, 1, "sq"),
}

class Node:
    def __init__(self, op=None, val=None, var=None, ch=()):
        self.op = op
        self.val = val
        self.var = var
        self.ch  = ch
    def is_terminal(self): return self.op is None
    def copy(self): return Node(self.op, self.val, self.var, tuple(c.copy() for c in self.ch))

def random_const(): return Node(val=random.uniform(-3.0, 3.0))
def random_var():   return Node(var=0)

def random_func():
    op = random.choice(list(FUNC_SET.keys()))
    _, arity, _ = FUNC_SET[op]
    return op, arity

def count_nodes(n): return 1 if n.is_terminal() else 1 + sum(count_nodes(c) for c in n.ch)
def clamp_size(n):  return random_var() if count_nodes(n) > MAX_NODES else n

def random_tree(max_depth):
    if max_depth <= 0 or random.random() < 0.3:
        return random.choice([random_const, random_var])()
    op, arity = random_func()
    children = tuple(random_tree(max_depth - 1) for _ in range(arity))
    return clamp_size(Node(op=op, ch=children))

def ramped_half_and_half(pop_size, max_depth):
    pop, depths = [], list(range(1, max_depth + 1))
    while len(pop) < pop_size:
        pop.append(random_tree(depths[len(pop) % len(depths)]))
    return pop

def is_const(n): return n.is_terminal() and n.var is None
def is_var(n):   return n.is_terminal() and n.var is not None
def same_var(a,b): return is_var(a) and is_var(b) and a.var == b.var

def simplify(n: Node) -> Node:
    if n.is_terminal(): return n
    op = n.op
    ch = tuple(simplify(c) for c in n.ch)
    if op == "sq":
        a = ch[0]
        if is_const(a): return Node(val=float(a.val*a.val))
        return Node(op=op, ch=(a,))
    a, b = ch if len(ch) == 2 else (None, None)
    if a is not None and b is not None and is_const(a) and is_const(b):
        fn, arity, _ = FUNC_SET[op]
        if arity == 2:
            val = fn(np.array([a.val]), np.array([b.val]))[0]
        else:
            val = fn(np.array([a.val]))[0]
        if np.isfinite(val): return Node(val=float(val))
    if op == "add":
        if is_const(a) and abs(a.val) < 1e-12: return b
        if is_const(b) and abs(b.val) < 1e-12: return a
    if op == "sub":
        if is_const(b) and abs(b.val) < 1e-12: return a
        if same_var(a, b): return Node(val=0.0)
    if op == "mul":
        if is_const(a):
            if abs(a.val - 1.0) < 1e-12: return b
            if abs(a.val) < 1e-12: return Node(val=0.0)
        if is_const(b):
            if abs(b.val - 1.0) < 1e-12: return a
            if abs(b.val) < 1e-12: return Node(val=0.0)
    return Node(op=op, ch=ch)

def poly_degree(n: Node) -> int:
    if n.is_terminal(): return 1 if n.var is not None else 0
    op = n.op
    if op == "sq":
        d = poly_degree(n.ch[0]); return 2*d if d < 999 else 999
    if op in ("add","sub"):
        return max(poly_degree(n.ch[0]), poly_degree(n.ch[1]))
    if op == "mul":
        da, db = poly_degree(n.ch[0]), poly_degree(n.ch[1])
        return da + db if da < 999 and db < 999 else 999
    return 999

def eval_tree(node, X):
    if node.is_terminal():
        return X[:,0] if node.var is not None else np.full(X.shape[0], node.val, dtype=float)
    fn, arity, _ = FUNC_SET[node.op]
    if arity == 1:
        a = eval_tree(node.ch[0], X); out = fn(a)
    else:
        a = eval_tree(node.ch[0], X); b = eval_tree(node.ch[1], X); out = fn(a, b)
    return np.where(np.isfinite(out), out, np.inf)

def tree_size(node): return 1 if node.is_terminal() else 1 + sum(tree_size(c) for c in node.ch)

def tree_str(node):
    if node.is_terminal():
        return VAR_NAMES[0] if node.var is not None else f"{node.val:.4g}"
    _, arity, sym = FUNC_SET[node.op]
    if node.op == "sq":
        inner = tree_str(node.ch[0])
        return f"({inner}^2)"
    if arity == 1:
        return f"{sym}({tree_str(node.ch[0])})"
    a, b = node.ch
    return f"({tree_str(a)} {sym} {tree_str(b)})"

def latex_for_coeffs(c0, c1, c2, var="t"):
    def fmt(x): return f"{x:.4g}"
    terms = [fmt(c0)]
    if abs(c1) > 1e-12: terms.append(f"{'+' if c1>=0 else '-'} {fmt(abs(c1))}{var}")
    if abs(c2) > 1e-12: terms.append(f"{'+' if c2>=0 else '-'} {fmt(abs(c2))}{var}^2")
    return r"$y = " + " ".join(terms) + r"$"

def mse(y_true, y_pred): d = y_true - y_pred; return float(np.mean(d*d))

def fitness(node, X, y, alpha, deg_cap=2):
    node = simplify(node)
    deg = poly_degree(node)
    if deg > deg_cap:
        return 1e10 + 1e6*(deg - deg_cap) + 1e3*tree_size(node)
    yhat = eval_tree(node, X)
    if not np.all(np.isfinite(yhat)): return 1e12
    return mse(y, yhat) + alpha * tree_size(node)

def mutate(node, max_depth):
    if max_depth <= 0: return random.choice([random_const, random_var])()
    if random.random() < 0.25: return random_tree(max_depth)
    if node.is_terminal(): return random.choice([random_const, random_var])()
    ch = list(node.ch); idx = random.randrange(len(ch))
    ch[idx] = mutate(ch[idx], max_depth - 1)
    return clamp_size(simplify(Node(op=node.op, ch=tuple(ch))))

def crossover(a, b, max_depth):
    if max_depth <= 0: return random.choice([random_const, random_var])()
    if random.random() < 0.5: return b.copy()
    if a.is_terminal(): return b.copy()
    ch = list(a.ch); idx = random.randrange(len(ch))
    ch[idx] = crossover(ch[idx], b, max_depth - 1)
    return clamp_size(simplify(Node(op=a.op, ch=tuple(ch))))

def evolve(X, y, pop_size=160, gens=100, max_depth=4, elite=4,
        alpha_start=3e-3, alpha_end=2e-4, seed=RNG_SEED):
    random.seed(seed); np.random.seed(seed)
    pop = ramped_half_and_half(pop_size, max_depth)
    def alpha_at(g):
        t = g/(gens-1) if gens>1 else 1.0
        return (1-t)*alpha_start + t*alpha_end
    for g in range(gens):
        a = alpha_at(g)
        scored = sorted([(fitness(ind, X, y, alpha=a), ind) for ind in pop], key=lambda x:x[0])
        best_score, best_ind = scored[0]
        print(f"Gen {g:02d}  alpha {a:.1e}  best {best_score:.6f}: {tree_str(simplify(best_ind))}")
        newpop = [simplify(best_ind.copy()) for _ in range(elite)]
        half = max(2, pop_size//2)
        while len(newpop) < pop_size:
            p1 = scored[random.randint(0, half-1)][1]
            p2 = scored[random.randint(0, half-1)][1]
            child = crossover(p1, p2, max_depth) if random.random() < 0.7 else mutate(p1, max_depth)
            newpop.append(child)
        pop = newpop
    scored = sorted([(fitness(ind, X, y, alpha=alpha_end), ind) for ind in pop], key=lambda x:x[0])
    return simplify(scored[0][1])

def poly_from_node(n):
    if n.is_terminal():
        if n.var is None:
            return (float(n.val), 0.0, 0.0)
        else:
            return (0.0, 1.0, 0.0)
    op = n.op
    if op == "sq":
        a = poly_from_node(n.ch[0])
        if a is None: return None
        c0, c1, c2 = a
        if abs(c2) > 1e-12: return None
        return (c0*c0, 2*c0*c1, c1*c1)
    a = poly_from_node(n.ch[0])
    b = poly_from_node(n.ch[1])
    if op in ("add","sub"):
        if a is None or b is None: return None
        sgn = 1.0 if op == "add" else -1.0
        return (a[0] + sgn*b[0], a[1] + sgn*b[1], a[2] + sgn*b[2])
    if op == "mul":
        if a is None or b is None: return None
        a0,a1,a2 = a; b0,b1,b2 = b
        if abs(a2) > 1e-12 and (abs(b1) > 1e-12 or abs(b2) > 1e-12): return None
        if abs(b2) > 1e-12 and (abs(a1) > 1e-12 or abs(a2) > 1e-12): return None
        c0 = a0*b0
        c1 = a0*b1 + a1*b0
        c2 = a0*b2 + a1*b1 + a2*b0
        return (c0, c1, c2)
    return None

def build_from_coeffs(c0,c1,c2):
    def const(x): return Node(val=float(x))
    parts = []
    if abs(c0) > 1e-12: parts.append(const(c0))
    if abs(c1) > 1e-12: parts.append(Node(op="mul", ch=(const(c1), Node(var=0))))
    if abs(c2) > 1e-12: parts.append(Node(op="mul", ch=(const(c2), Node(op="sq", ch=(Node(var=0),)))))
    if not parts: return const(0.0)
    out = parts[0]
    for p in parts[1:]:
        out = Node(op="add", ch=(out, p))
    return simplify(out)

def compress_to_quadratic(node):
    coeffs = poly_from_node(node)
    if coeffs is None:
        return node, None
    c0,c1,c2 = coeffs
    return build_from_coeffs(c0,c1,c2), coeffs

def fit_quadratic_sr(T, Y, *, pop_size=180, gens=120, max_depth=4,
                    alpha_start=3e-3, alpha_end=2e-4, seed=11):
    best = evolve(T, Y, pop_size=pop_size, gens=gens, max_depth=max_depth,
                elite=4, alpha_start=alpha_start, alpha_end=alpha_end, seed=seed)
    best_simple, _ = compress_to_quadratic(best)
    t = T[:, 0]
    X = np.stack([np.ones_like(t), t, t*t], axis=1)
    c_refined, *_ = np.linalg.lstsq(X, Y, rcond=None)
    best_refined = build_from_coeffs(*c_refined)
    return best_refined, tuple(map(float, c_refined))

def predict_from_coeffs(t_array, coeffs):
    c0, c1, c2 = coeffs
    t = np.asarray(t_array)
    return c0 + c1*t + c2*(t**2)

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "Outputs" / "ball_trajectory.csv"
    equation_path = repo_root / "Outputs" / "yx_equation_pixel_custom.txt"
    equation_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    T = df[["X"]].to_numpy(dtype=float)
    Y = df["Y"].to_numpy(dtype=float)
    best_model, coeffs = fit_quadratic_sr(T, Y,
                                          pop_size=180, gens=120, max_depth=4,
                                          alpha_start=3e-3, alpha_end=2e-4, seed=11)
    expr = tree_str(best_model)
    with open(equation_path, "w") as f:
        f.write(f"y = {expr}\n")
    print(f"Wrote equation to {equation_path}")

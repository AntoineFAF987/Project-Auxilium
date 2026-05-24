# -*- coding: utf-8 -*-
import re
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

EXEC_MATH = ThreadPoolExecutor(max_workers=2)  # petit pool

# --------- Suivi "explique/démontre" ---------
EXPLAIN_PATTERNS = [
    r"\b(explique|d(é|e)tail(le|le)?|d(ée|e)montre|méthode|d(é|e)marche|étapes?)\b",
    r"\b(how did you|explain|steps?)\b",
]
def _is_explain_followup(q: str) -> bool:
    return bool(q and (any(re.search(p, q, flags=re.IGNORECASE) for p in EXPLAIN_PATTERNS)))

# --------- Détection équation ---------
def _looks_like_equation(q: str) -> bool:
    if not q:
        return False
    t = q.strip()
    return bool(re.search(r"[=+\-*/^()]", t)) and bool(re.search(r"\d|[xyzXYZ]", t))

# --------- Solveur maths (SymPy) ---------
def _solve_math_impl(q: str, detailed: bool = False) -> Optional[str]:
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )
    except Exception:
        return None

    try:
        s = (q or "").strip().replace("^", "**")
        x, y, z = sp.symbols("x y z")
        allowed = {
            "x": x, "y": y, "z": z,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "pi": sp.pi, "E": sp.E,
        }
        transformations = (standard_transformations + (implicit_multiplication_application,))

        if "=" in s:
            left, right = s.split("=", 1)
            L = parse_expr(left, local_dict=allowed, transformations=transformations, evaluate=True)
            R = parse_expr(right, local_dict=allowed, transformations=transformations, evaluate=True)
            eq = sp.Eq(L, R)
            expr0 = L - R
        else:
            expr = parse_expr(s, local_dict=allowed, transformations=transformations, evaluate=True)
            eq = sp.Eq(expr, 0)
            expr0 = expr

        free = list(eq.free_symbols) or [x]
        var = x if x in free else (y if y in free else (z if z in free else free[0]))

        steps: List[str] = []
        latex = sp.latex
        steps.append(f"**Équation à résoudre :**  \n$$ {latex(eq.lhs)} = {latex(eq.rhs)} $$")

        poly_form = sp.expand(expr0)
        steps.append(f"**Forme étudiée :**  \n$$ {latex(poly_form)} = 0 $$")

        fact = sp.factor(poly_form)
        if sp.simplify(fact - poly_form) != 0:
            steps.append(f"**Factorisation :**  \n$$ {latex(poly_form)} = {latex(fact)} $$")

        try:
            sols = sp.solve(sp.Eq(poly_form, 0), var, dict=False)
        except Exception:
            sols = []

        if sols:
            steps.append(
                "**Solutions exactes :**  \n" + " \n".join([f"$$ {latex(var)} = {latex(s)} $$" for s in (sols if isinstance(sols, (list, tuple)) else [sols])])
            )
        else:
            try:
                roots = sp.nroots(poly_form)
                steps.append("**Solutions numériques (approx.) :**  \n" + " \n".join([f"$$ {latex(var)} \\approx {sp.N(r, 8)} $$" for r in roots]))
            except Exception:
                steps.append("_Aucune solution trouvée._")

        return "\n\n".join(steps)
    except Exception:
        return None

def _solve_math(q: str, *, detailed: bool = False, timeout_sec: int = 8) -> Optional[str]:
    fut = EXEC_MATH.submit(_solve_math_impl, q, detailed)
    try:
        return fut.result(timeout=timeout_sec)
    except FuturesTimeout:
        return None
    except Exception:
        return None

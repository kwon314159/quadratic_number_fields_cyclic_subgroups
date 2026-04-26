# solver_core.sage
# Core Sage routines for explicit quadratic-point constructions on X_0(N).
#
# File guide
# ----------
# 1. `get_quadratic_family_metadata`, `setup_modular_forms`
#    Fixed level metadata and the target q-series f, g, f_N, g_N.
# 2. `find_minimal_robust_solution`
#    Default exact-kernel search used by the public generic solver.
# 3. `find_minimal_robust_solution_mod_p`
#    Mod-p screening plus exact lifting, used by the dedicated public N = 131 path.
# 4. `_build_search_runtime`, `_find_map_with_plan`, `_solve_maps_for_level`
#    Shared search orchestration for recovering u, u_N, v, v_N.
# 5. `_solve_maps_for_level_131`
#    Fixed-degree search path for N = 131.
# 6. `build_family_report`
#    Converts recovered maps into notebook / text report payloads.
# 7. `solve_isogeny_result`, `solve_isogeny_report`, `solve_single_case_report`,
#    `run_batch_and_save`
#    Public entry points for notebooks and headless batch generation.
#
# This file loads `sagecode/x0_model_data.sage` for plane models and explicit
# generator data.


from functools import reduce
from time import perf_counter

# Fixed quadratic-family classification from the paper.
QUADRATIC_FAMILY_ELLIPTIC = {
    11, 14, 15, 17, 19, 20, 21, 24, 27, 32, 36, 49
}
QUADRATIC_FAMILY_HYPERELLIPTIC = {
    22, 23, 26, 28, 29, 30, 31, 33, 35, 37, 39, 40, 41, 46, 47, 48, 50, 59, 71
}
QUADRATIC_FAMILY_BIELLIPTIC = {
    43: '43a1',
    53: '53a1',
    61: '61a1',
    65: '65a1',
    79: '79a1',
    83: '83a1',
    89: '89a1',
    101: '101a1',
    131: '131a1',
}

# Public special-case search data for the bielliptic level N = 131.
# The generic exact-kernel search remains the default elsewhere.
SPECIAL_LEVEL_131_FIXED_PLAN = (
    {'key': 'u',  'target_key': 'f',  'label': 'u',   'degree': 36, 'max_deg_y': 12},
    {'key': 'uN', 'target_key': 'fN', 'label': 'u_N', 'degree': 36, 'max_deg_y': 12},
    {'key': 'v',  'target_key': 'g',  'label': 'v',   'degree': 39, 'max_deg_y': 13},
    {'key': 'vN', 'target_key': 'gN', 'label': 'v_N', 'degree': 39, 'max_deg_y': 13},
)

def get_quadratic_family_metadata(N):
    """
    Returns the fixed paper classification for the quadratic-point family layer.
    """
    metadata = {
        'quadratic_family_type': 'general',
        'family_parameter_var': 't',
        'family_extension_var': 'alpha_t',
        'quotient_label': None,
    }
    if N in QUADRATIC_FAMILY_ELLIPTIC:
        metadata['quadratic_family_type'] = 'elliptic'
    elif N in QUADRATIC_FAMILY_HYPERELLIPTIC:
        metadata['quadratic_family_type'] = 'hyperelliptic'
    elif N in QUADRATIC_FAMILY_BIELLIPTIC:
        metadata['quadratic_family_type'] = 'bielliptic'
        metadata['quotient_label'] = QUADRATIC_FAMILY_BIELLIPTIC[N]
    return metadata

def _compute_triangular_cuspform_basis(N, prec):
    """
    Compute a triangular weight-2 cuspform basis x_i = q^(i+1) + O(q^(i+2)).
    """
    R_q = LaurentSeriesRing(QQ, 'q', default_prec=prec)
    S2 = CuspForms(N, 2)
    initial_basis = S2.integral_basis()
    genus = S2.dimension()
    if genus == 0 or not initial_basis:
        return []

    coeff_matrix = Matrix(
        QQ,
        [[f.q_expansion(genus + 10)[n] for n in range(1, genus + 1)] for f in initial_basis]
    )
    coeff_matrix_inv = coeff_matrix.inverse()

    basis = []
    for i in range(genus):
        coeffs = coeff_matrix_inv.row(i)
        f_new = sum(coeffs[j] * initial_basis[j] for j in range(genus))
        basis.append(R_q(f_new.q_expansion(prec)))
    return basis

# ==============================================================================
# 0. Model Data Repository
# ==============================================================================
load("sagecode/x0_model_data.sage")

# ==============================================================================
# 1. Setup & Definitions (Modular Forms)
# ==============================================================================

def setup_modular_forms(N, prec=200):
    """
    Build the standard modular forms E2, E4, E6 and the target functions
    f, g, fN, gN used in the parameterization.
    """
    R_q = LaurentSeriesRing(QQ, 'q', default_prec=prec)
    q = R_q.gen()
    
    # Standard Eisenstein Series
    E2 = 24 * eisenstein_series_qexp(2, prec)
    E4 = 240 * eisenstein_series_qexp(4, prec)
    E6 = -504 * eisenstein_series_qexp(6, prec)
    
    # Level N substitutions
    E2_Ntau = E2(q=q^N)
    E4_Ntau = E4(q=q^N)
    E6_Ntau = E6(q=q^N)
    
    # Weight 2 form on Gamma0(N)
    E2N = -1/24 * (N * E2_Ntau - E2)
    
    # Target modular functions (Weight 0)
    f  = -1/48 * E4 / E2N^2
    g  =  1/864 * E6 / E2N^3
    fN = -1/48 * E4_Ntau / E2N^2
    gN =  1/864 * E6_Ntau / E2N^3
    
    return {'q': q, 'f': f, 'g': g, 'fN': fN, 'gN': gN, 'E2N': E2N}

# ==============================================================================
# 2. Solver Core (Linear Algebra)
# ==============================================================================

def _series_valuation_precision(series):
    """Return `(valuation, absolute_precision)` for a q-series, or `None` for zero."""
    if series == 0:
        return None
    try:
        v = series.valuation()
    except Exception:
        return None
    try:
        p = series.precision_absolute()
    except Exception:
        try:
            p = series.prec()
        except Exception:
            p = float('inf')
    return v, p

def _scan_series_list(series_list):
    """Collect reusable valuation/precision metadata for matrix assembly."""
    col_data = []
    min_val = None
    min_prec = None
    for col, series in enumerate(series_list):
        vp = _series_valuation_precision(series)
        if vp is None:
            continue
        valuation, precision = vp
        col_data.append((col, series, valuation, precision))
        min_val = valuation if min_val is None else min(min_val, valuation)
        min_prec = precision if min_prec is None else min(min_prec, precision)
    if min_val is None:
        return None
    return {
        "col_data": col_data,
        "min_val": min_val,
        "min_prec": min_prec,
    }

def _quick_split_feasible(scan, num_eqs, verification_order):
    """Conservative split pruning based on usable rows and available precision."""
    if scan is None:
        return False, 0

    min_val = scan["min_val"]
    min_prec = scan["min_prec"]
    if min_prec is None or min_prec == float('inf'):
        rows_eff = num_eqs
    else:
        rows_usable = max(0, min_prec - min_val)
        rows_eff = min(num_eqs, rows_usable)

    if rows_eff <= 0:
        return False, rows_eff
    if min_prec is not None and min_prec != float('inf'):
        if verification_order is not None and verification_order > min_prec:
            return False, rows_eff
    return True, rows_eff

def normalize_vector_to_ints(vec):
    """Normalizes a rational vector to the smallest integer vector."""
    vec_int = vec * vec.denominator()
    int_list = [c.numerator() for c in vec_int]
    non_zero = [c for c in int_list if c != 0]
    common = reduce(gcd, non_zero) if non_zero else 1
    return vector(ZZ, [c // common for c in int_list])

def _build_solver_precomputed(x0, y0, max_total_degree, max_deg_y, precomputed=None):
    """Build reusable degree-indexed monomial data, reusing `precomputed` when possible."""
    if precomputed is None:
        precomputed = {}

    same_deg_y = precomputed.get("max_deg_y", None) == max_deg_y
    enough_total = precomputed.get("max_total_degree", -1) >= max_total_degree
    has_basis = isinstance(precomputed.get("basis_by_degree", None), list)
    has_neg_basis = isinstance(precomputed.get("neg_basis_by_degree", None), list)
    has_exponents = isinstance(precomputed.get("exponents_by_degree", None), list)
    if same_deg_y and enough_total and has_basis and has_neg_basis and has_exponents:
        if "target_basis_by_degree" not in precomputed:
            precomputed["target_basis_by_degree"] = {}
        return precomputed

    x_pows = [x0**i for i in range(max_total_degree + 1)]
    y_pows = [y0**j for j in range(max_deg_y + 1)]
    basis_by_degree = []
    neg_basis_by_degree = []
    exponents_by_degree = []

    for deg in range(max_total_degree + 1):
        basis_deg = []
        exponents_deg = []
        for j in range(min(deg, max_deg_y) + 1):
            y_pow = y_pows[j]
            for i in range(deg - j + 1):
                basis_deg.append(x_pows[i] * y_pow)
                exponents_deg.append((i, j))
        basis_by_degree.append(basis_deg)
        neg_basis_by_degree.append([-basis for basis in basis_deg])
        exponents_by_degree.append(exponents_deg)

    precomputed.clear()
    precomputed.update({
        "max_total_degree": max_total_degree,
        "max_deg_y": max_deg_y,
        "basis_by_degree": basis_by_degree,
        "neg_basis_by_degree": neg_basis_by_degree,
        "exponents_by_degree": exponents_by_degree,
        "target_basis_by_degree": {},
    })
    return precomputed

def _usable_row_count(scan):
    """Return the number of q-rows supported by the scanned series data."""
    if scan is None:
        return 0
    min_prec = scan["min_prec"]
    if min_prec is None or min_prec == float('inf'):
        return float('inf')
    return max(0, int(min_prec - scan["min_val"]))

def _effective_row_window(scan, row_start, row_end):
    """Clip a requested row window to the actually available q-range."""
    rows_available = _usable_row_count(scan)
    row_start_eff = max(0, int(row_start))
    row_end_eff = max(row_start_eff, int(row_end))
    if rows_available != float('inf'):
        row_start_eff = min(row_start_eff, rows_available)
        row_end_eff = min(row_end_eff, rows_available)
    return row_start_eff, row_end_eff

def assemble_matrix_generic_range(series_list, row_start, row_end, verbose=False, pre_scan=None):
    """Construct a sparse linear system matrix from a clipped q-row window."""
    scan = pre_scan if pre_scan is not None else _scan_series_list(series_list)
    if scan is None:
        return None, 0, 0, 0

    row_start_eff, row_end_eff = _effective_row_window(scan, row_start, row_end)
    if row_end_eff <= row_start_eff:
        return None, scan["min_val"], row_start_eff, row_end_eff

    col_data = scan["col_data"]
    min_val = scan["min_val"]
    entries = {}
    q_start = min_val + row_start_eff
    q_end = min_val + row_end_eff
    for col, series, valuation, precision in col_data:
        col_start = max(q_start, valuation)
        col_end = q_end if precision == float('inf') else min(q_end, precision)
        if col_start >= col_end:
            continue
        coeff_dict = None
        try:
            coeff_dict = series.dict()
        except Exception:
            coeff_dict = None

        if coeff_dict is not None:
            for exponent, coeff in coeff_dict.items():
                if coeff == 0:
                    continue
                if exponent < col_start or exponent >= col_end:
                    continue
                entries[(exponent - q_start, col)] = coeff
        else:
            for exponent in range(col_start, col_end):
                coeff = series[exponent]
                if coeff != 0:
                    entries[(exponent - q_start, col)] = coeff

    M = matrix(QQ, row_end_eff - row_start_eff, len(series_list), entries, sparse=True)
    return M, min_val, row_start_eff, row_end_eff

def assemble_matrix_generic(series_list, num_eqs, verbose=False, pre_scan=None):
    """Construct the sparse linear system matrix from q-expansion data."""
    M, min_val, row_start_eff, row_end_eff = assemble_matrix_generic_range(
        series_list, 0, num_eqs, verbose=verbose, pre_scan=pre_scan
    )
    return M, min_val, max(0, row_end_eff - row_start_eff)

def _default_mod_p_screening_primes():
    """Return large primes that are unlikely to hit q-series denominators."""
    return (
        2147483647,
        2013265921,
        1811939329,
        1224736769,
        1004535809,
        1000000009,
        1000000007,
        998244353,
        754974721,
        469762049,
        167772161,
    )

def _coeff_mod_prime(coeff, Fp):
    """Reduce a rational coefficient modulo `Fp`, returning `None` on bad denominator."""
    if coeff == 0:
        return Fp(0)
    coeff_q = QQ(coeff)
    den_mod = Fp(coeff_q.denominator())
    if den_mod == 0:
        return None
    return Fp(coeff_q.numerator()) / den_mod

def assemble_matrix_generic_range_mod_p(series_list, row_start, row_end, prime, verbose=False, pre_scan=None):
    """Construct a sparse matrix over GF(prime) from a clipped q-row window."""
    scan = pre_scan if pre_scan is not None else _scan_series_list(series_list)
    if scan is None:
        return None, 0, 0, 0, False

    row_start_eff, row_end_eff = _effective_row_window(scan, row_start, row_end)
    if row_end_eff <= row_start_eff:
        return None, scan["min_val"], row_start_eff, row_end_eff, False

    Fp = GF(int(prime))
    col_data = scan["col_data"]
    min_val = scan["min_val"]
    entries = {}
    q_start = min_val + row_start_eff
    q_end = min_val + row_end_eff
    for col, series, valuation, precision in col_data:
        col_start = max(q_start, valuation)
        col_end = q_end if precision == float('inf') else min(q_end, precision)
        if col_start >= col_end:
            continue
        coeff_dict = None
        try:
            coeff_dict = series.dict()
        except Exception:
            coeff_dict = None

        if coeff_dict is not None:
            for exponent, coeff in coeff_dict.items():
                if coeff == 0:
                    continue
                if exponent < col_start or exponent >= col_end:
                    continue
                coeff_mod = _coeff_mod_prime(coeff, Fp)
                if coeff_mod is None:
                    return None, min_val, row_start_eff, row_end_eff, True
                if coeff_mod != 0:
                    entries[(exponent - q_start, col)] = coeff_mod
        else:
            for exponent in range(col_start, col_end):
                coeff = series[exponent]
                if coeff == 0:
                    continue
                coeff_mod = _coeff_mod_prime(coeff, Fp)
                if coeff_mod is None:
                    return None, min_val, row_start_eff, row_end_eff, True
                if coeff_mod != 0:
                    entries[(exponent - q_start, col)] = coeff_mod

    M = matrix(Fp, row_end_eff - row_start_eff, len(series_list), entries, sparse=True)
    return M, min_val, row_start_eff, row_end_eff, False

def _screening_row_target(rows_available, k_min, L0, Delta, verification_order):
    """Choose how many q-rows to use for screening/exact verification."""
    if rows_available != float('inf'):
        return max(0, int(rows_available))

    min_verify_rows = 0
    if verification_order is not None:
        min_verify_rows = max(0, int(verification_order) - int(k_min))
    return max(int(L0) + max(0, int(Delta)), min_verify_rows)

def _screen_split_mod_p(series_list, row_target, prime_candidates=None, prime_limit=8, verbose=False, pre_scan=None):
    """Use several good primes to cheaply rule out splits with no kernel."""
    if prime_candidates is None:
        prime_candidates = _default_mod_p_screening_primes()

    best = None
    good_prime_count = 0
    bad_prime_count = 0
    tested = []

    for prime in prime_candidates:
        if good_prime_count >= max(1, int(prime_limit)):
            break
        M_p, min_val, row_start_eff, row_end_eff, bad_prime = assemble_matrix_generic_range_mod_p(
            series_list,
            0,
            row_target,
            prime,
            verbose=verbose,
            pre_scan=pre_scan,
        )
        if bad_prime:
            bad_prime_count += 1
            continue
        if M_p is None or row_end_eff <= row_start_eff:
            continue

        tested.append(int(prime))
        good_prime_count += 1
        nullity = max(0, M_p.ncols() - M_p.rank())
        info = {
            "prime": int(prime),
            "matrix": M_p,
            "nullity": int(nullity),
            "rows_used": int(row_end_eff),
            "rows_available": row_target,
            "pivots": tuple(int(v) for v in M_p.pivots()),
            "pivot_rows": tuple(int(v) for v in M_p.pivot_rows()),
            "free_cols": tuple(int(v) for v in M_p.nonpivots()),
        }
        if best is None or info["nullity"] < best["nullity"]:
            best = info
        if info["nullity"] == 0:
            return {
                "rejected": True,
                "best": info,
                "good_prime_count": good_prime_count,
                "bad_prime_count": bad_prime_count,
                "tested_primes": tested,
            }

    return {
        "rejected": False,
        "best": best,
        "good_prime_count": good_prime_count,
        "bad_prime_count": bad_prime_count,
        "tested_primes": tested,
    }

def _candidate_vectors_from_mod_p_screen(M_exact, screen_info, max_candidates=8):
    """Lift a small set of normalized kernel candidates from mod-p pivot data."""
    if screen_info is None:
        return []

    pivot_rows = tuple(screen_info.get("pivot_rows", ()))
    pivot_cols = tuple(screen_info.get("pivots", ()))
    free_cols = tuple(screen_info.get("free_cols", ()))
    if not free_cols:
        return []

    candidate_vectors = []
    max_candidates = max(1, int(max_candidates))
    ambient_dim = M_exact.ncols()

    if pivot_cols:
        pivot_block = M_exact.matrix_from_rows_and_columns(pivot_rows, pivot_cols)
    else:
        pivot_block = None

    for free_col in free_cols[:max_candidates]:
        if pivot_block is None:
            candidate_entries = [QQ(0)] * ambient_dim
            candidate_entries[int(free_col)] = QQ(1)
            candidate_vectors.append(vector(QQ, candidate_entries))
            continue

        rhs = vector(
            QQ,
            [-M_exact[int(row_idx), int(free_col)] for row_idx in pivot_rows],
        )
        try:
            pivot_solution = pivot_block.solve_right(rhs)
        except Exception:
            continue

        candidate_entries = [QQ(0)] * ambient_dim
        for idx, pivot_col in enumerate(pivot_cols):
            candidate_entries[int(pivot_col)] = pivot_solution[idx]
        candidate_entries[int(free_col)] = QQ(1)
        candidate_vectors.append(vector(QQ, candidate_entries))

    if len(free_cols) > 1 and len(candidate_vectors) < max_candidates:
        if pivot_block is None:
            candidate_entries = [QQ(1) if idx in free_cols else QQ(0) for idx in range(ambient_dim)]
            candidate_vectors.append(vector(QQ, candidate_entries))
        else:
            rhs = vector(
                QQ,
                [-sum(M_exact[int(row_idx), int(free_col)] for free_col in free_cols) for row_idx in pivot_rows],
            )
            try:
                pivot_solution = pivot_block.solve_right(rhs)
                candidate_entries = [QQ(0)] * ambient_dim
                for idx, pivot_col in enumerate(pivot_cols):
                    candidate_entries[int(pivot_col)] = pivot_solution[idx]
                for free_col in free_cols:
                    candidate_entries[int(free_col)] = QQ(1)
                candidate_vectors.append(vector(QQ, candidate_entries))
            except Exception:
                pass

    return candidate_vectors

def reconstruct_polynomial(coeffs, deg, max_deg_y, P_xy, monomial_exponents=None):
    """Reconstructs a polynomial from coefficient data and monomial ordering."""
    x, y = P_xy.gens()
    poly = P_xy(0)

    if monomial_exponents is None:
        monomial_exponents = []
        for j in range(min(deg, max_deg_y) + 1):
            for i in range(deg - j + 1):
                monomial_exponents.append((i, j))

    upto = min(len(coeffs), len(monomial_exponents))
    for n in range(upto):
        coeff = coeffs[n]
        if coeff == 0:
            continue
        i, j = monomial_exponents[n]
        poly += coeff * (x**i * y**j)
    return poly

def find_minimal_robust_solution(target_f, x0, y0, P_xy,
                                 min_total_degree=1,
                                 max_total_degree=10, max_deg_y=5,
                                 L0_margin=5, Delta=30, verification_order=100,
                                 stop_at_first_total_degree=True, verbose=False,
                                 profile=False, use_cache=True, precomputed=None,
                                 include_q_series=True, return_diagnostics=False):
    """Search for minimal robust rational approximations A/B to `target_f`."""
    phase_times = {
        "basis": 0.0,
        "prune": 0.0,
        "matrix": 0.0,
        "kernel": 0.0,
        "verify": 0.0,
    }
    solutions = []
    diagnostics = {
        "split_total": 0,
        "precision_limited_splits": 0,
        "no_kernel_splits": 0,
        "kernel_splits": 0,
        "kernel_but_no_valid_splits": 0,
        "candidate_vectors": 0,
        "valid_solutions": 0,
    }
    start_degree = max(1, Integer(min_total_degree))
    if start_degree > max_total_degree:
        return (solutions, diagnostics) if return_diagnostics else solutions

    basis_by_degree = None
    neg_basis_by_degree = None
    exponents_by_degree = None
    target_basis_by_degree = None
    if use_cache:
        t0 = perf_counter() if profile else None
        cache = _build_solver_precomputed(
            x0, y0, max_total_degree, max_deg_y, precomputed=precomputed
        )
        basis_by_degree = cache["basis_by_degree"]
        neg_basis_by_degree = cache["neg_basis_by_degree"]
        exponents_by_degree = cache["exponents_by_degree"]
        target_basis_by_degree = cache.setdefault("target_basis_by_degree", {})
        target_key = (id(target_f), max_total_degree, max_deg_y)
        if target_key not in target_basis_by_degree:
            target_basis_by_degree[target_key] = {}
        if profile:
            phase_times["basis"] += perf_counter() - t0
    else:
        target_key = None

    for total_degree in range(start_degree, max_total_degree + 1):
        if verbose:
            print(f"  Searching degree {total_degree}...")
        degree_has_solution = False

        for deg_B_total in range(total_degree + 1):
            diagnostics["split_total"] += 1
            deg_A_total = total_degree - deg_B_total
            t_basis = perf_counter() if profile else None
            if use_cache:
                basis_A_q = basis_by_degree[deg_A_total]
                neg_basis_A_q = neg_basis_by_degree[deg_A_total]
                basis_B_q = basis_by_degree[deg_B_total]
                exp_A = exponents_by_degree[deg_A_total]
                exp_B = exponents_by_degree[deg_B_total]

                target_cache_for_f = target_basis_by_degree[target_key]
                basis_B_target_q = target_cache_for_f.get(deg_B_total, None)
                if basis_B_target_q is None:
                    basis_B_target_q = [basis * target_f for basis in basis_B_q]
                    target_cache_for_f[deg_B_total] = basis_B_target_q
            else:
                exp_A = [
                    (i, j)
                    for j in range(min(deg_A_total, max_deg_y) + 1)
                    for i in range(deg_A_total - j + 1)
                ]
                exp_B = [
                    (i, j)
                    for j in range(min(deg_B_total, max_deg_y) + 1)
                    for i in range(deg_B_total - j + 1)
                ]
                basis_A_q = [x0**i * y0**j for (i, j) in exp_A]
                neg_basis_A_q = [-basis for basis in basis_A_q]
                basis_B_q = [x0**i * y0**j for (i, j) in exp_B]
                basis_B_target_q = [basis * target_f for basis in basis_B_q]
            if profile:
                phase_times["basis"] += perf_counter() - t_basis

            k_A, k_B = len(basis_A_q), len(basis_B_q)
            series_list = neg_basis_A_q + basis_B_target_q
            L0 = k_A + k_B + L0_margin

            t_prune = perf_counter() if profile else None
            split_scan = _scan_series_list(series_list)
            split_ok, _ = _quick_split_feasible(split_scan, L0, verification_order)
            if profile:
                phase_times["prune"] += perf_counter() - t_prune
            if not split_ok:
                diagnostics["precision_limited_splits"] += 1
                continue

            t_matrix = perf_counter() if profile else None
            M_L0, k_min, L0_eff = assemble_matrix_generic(
                series_list, L0, verbose, pre_scan=split_scan
            )
            if profile:
                phase_times["matrix"] += perf_counter() - t_matrix
            if M_L0 is None or L0_eff <= 0:
                diagnostics["precision_limited_splits"] += 1
                continue

            t_kernel = perf_counter() if profile else None
            K_L0 = M_L0.right_kernel()
            if profile:
                phase_times["kernel"] += perf_counter() - t_kernel
            if K_L0.dimension() == 0:
                diagnostics["no_kernel_splits"] += 1
                continue
            diagnostics["kernel_splits"] += 1

            min_required_valuation = max(k_min + L0_eff, verification_order)
            M_ext = None
            ext_has_new_rows = None
            L_ext_target = L0_eff + max(0, Delta)
            split_has_valid_solution = False

            for candidate_vec in K_L0.basis():
                if not any(candidate_vec[i] != 0 for i in range(k_A)):
                    continue
                if not any(candidate_vec[k_A + i] != 0 for i in range(k_B)):
                    continue
                diagnostics["candidate_vectors"] += 1

                t_verify = perf_counter() if profile else None
                is_valid = True

                if is_valid and Delta > 0:
                    if ext_has_new_rows is None:
                        t_matrix_ext = perf_counter() if profile else None
                        M_ext, _, rows_ext = assemble_matrix_generic(
                            series_list, L_ext_target, verbose, pre_scan=split_scan
                        )
                        if profile:
                            phase_times["matrix"] += perf_counter() - t_matrix_ext
                        ext_has_new_rows = (M_ext is not None and rows_ext > L0_eff)
                    if ext_has_new_rows and not (M_ext * candidate_vec).is_zero():
                        is_valid = False

                if is_valid:
                    A_q_chk = sum(candidate_vec[i] * basis_A_q[i] for i in range(k_A))
                    B_q_chk = sum(candidate_vec[k_A + i] * basis_B_q[i] for i in range(k_B))
                    S = B_q_chk * target_f - A_q_chk
                    if S.valuation() < min_required_valuation:
                        is_valid = False

                if profile:
                    phase_times["verify"] += perf_counter() - t_verify
                if not is_valid:
                    continue

                final_coeffs = normalize_vector_to_ints(candidate_vec)
                A_poly = reconstruct_polynomial(final_coeffs[:k_A], deg_A_total, max_deg_y, P_xy, exp_A)
                B_poly = reconstruct_polynomial(final_coeffs[k_A:], deg_B_total, max_deg_y, P_xy, exp_B)
                sol = {
                    "A_poly": A_poly,
                    "B_poly": B_poly,
                    "deg_A_total": deg_A_total,
                    "deg_B_total": deg_B_total,
                    "total_degree": total_degree,
                    "max_deg_y": max_deg_y,
                }
                if include_q_series:
                    sol["A_q"] = sum(final_coeffs[i] * basis_A_q[i] for i in range(k_A))
                    sol["B_q"] = sum(final_coeffs[k_A + i] * basis_B_q[i] for i in range(k_B))

                solutions.append(sol)
                degree_has_solution = True
                split_has_valid_solution = True
                diagnostics["valid_solutions"] += 1
                if verbose:
                    print(f"  [Found] Solution at degree {total_degree}")

            if not split_has_valid_solution:
                diagnostics["kernel_but_no_valid_splits"] += 1

        if degree_has_solution and stop_at_first_total_degree:
            break

    if profile:
        print(
            "[Profiler] "
            f"basis={phase_times['basis']:.3f}s, "
            f"prune={phase_times['prune']:.3f}s, "
            f"matrix={phase_times['matrix']:.3f}s, "
            f"kernel={phase_times['kernel']:.3f}s, "
            f"verify={phase_times['verify']:.3f}s"
        )
    return (solutions, diagnostics) if return_diagnostics else solutions

def find_minimal_robust_solution_mod_p(target_f, x0, y0, P_xy,
                                       min_total_degree=1,
                                       max_total_degree=10, max_deg_y=5,
                                       L0_margin=5, Delta=30, verification_order=100,
                                       stop_at_first_total_degree=True, verbose=False,
                                       profile=False, use_cache=True, precomputed=None,
                                       include_q_series=True, return_diagnostics=False,
                                       fixed_split=None,
                                       mod_p_screening_primes=None,
                                       mod_p_screening_prime_limit=8,
                                       mod_p_exact_free_col_limit=8):
    """Search for rational approximations A/B using mod-p screening and exact lifting."""
    phase_times = {
        "basis": 0.0,
        "prune": 0.0,
        "screen": 0.0,
        "matrix": 0.0,
        "solve": 0.0,
        "verify": 0.0,
    }
    solutions = []
    diagnostics = {
        "split_total": 0,
        "precision_limited_splits": 0,
        "no_kernel_splits": 0,
        "kernel_splits": 0,
        "kernel_but_no_valid_splits": 0,
        "ambiguous_kernel_splits": 0,
        "max_kernel_dim_seen": 0,
        "candidate_vectors": 0,
        "valid_solutions": 0,
        "screen_good_primes": 0,
        "screen_bad_primes": 0,
        "screen_rejected_splits": 0,
    }
    start_degree = max(1, Integer(min_total_degree))
    if fixed_split is not None:
        fixed_split = tuple(Integer(v) for v in fixed_split)
        fixed_total_degree = fixed_split[0] + fixed_split[1]
    else:
        fixed_total_degree = None
    if start_degree > max_total_degree:
        return (solutions, diagnostics) if return_diagnostics else solutions

    basis_by_degree = None
    neg_basis_by_degree = None
    exponents_by_degree = None
    target_basis_by_degree = None
    if use_cache:
        t0 = perf_counter() if profile else None
        cache = _build_solver_precomputed(
            x0, y0, max_total_degree, max_deg_y, precomputed=precomputed
        )
        basis_by_degree = cache["basis_by_degree"]
        neg_basis_by_degree = cache["neg_basis_by_degree"]
        exponents_by_degree = cache["exponents_by_degree"]
        target_basis_by_degree = cache.setdefault("target_basis_by_degree", {})
        target_key = (id(target_f), max_total_degree, max_deg_y)
        if target_key not in target_basis_by_degree:
            target_basis_by_degree[target_key] = {}
        if profile:
            phase_times["basis"] += perf_counter() - t0
    else:
        target_key = None

    for total_degree in range(start_degree, max_total_degree + 1):
        if fixed_total_degree is not None and total_degree != fixed_total_degree:
            continue
        if verbose:
            print(f"  Searching degree {total_degree}...")
        degree_has_solution = False

        deg_B_iter = [fixed_split[1]] if fixed_split is not None else range(total_degree + 1)

        for deg_B_total in deg_B_iter:
            diagnostics["split_total"] += 1
            deg_A_total = total_degree - deg_B_total
            if fixed_split is not None and (deg_A_total, deg_B_total) != fixed_split:
                continue

            t_basis = perf_counter() if profile else None
            if use_cache:
                basis_A_q = basis_by_degree[deg_A_total]
                neg_basis_A_q = neg_basis_by_degree[deg_A_total]
                basis_B_q = basis_by_degree[deg_B_total]
                exp_A = exponents_by_degree[deg_A_total]
                exp_B = exponents_by_degree[deg_B_total]

                target_cache_for_f = target_basis_by_degree[target_key]
                basis_B_target_q = target_cache_for_f.get(deg_B_total, None)
                if basis_B_target_q is None:
                    basis_B_target_q = [basis * target_f for basis in basis_B_q]
                    target_cache_for_f[deg_B_total] = basis_B_target_q
            else:
                exp_A = [
                    (i, j)
                    for j in range(min(deg_A_total, max_deg_y) + 1)
                    for i in range(deg_A_total - j + 1)
                ]
                exp_B = [
                    (i, j)
                    for j in range(min(deg_B_total, max_deg_y) + 1)
                    for i in range(deg_B_total - j + 1)
                ]
                basis_A_q = [x0**i * y0**j for (i, j) in exp_A]
                neg_basis_A_q = [-basis for basis in basis_A_q]
                basis_B_q = [x0**i * y0**j for (i, j) in exp_B]
                basis_B_target_q = [basis * target_f for basis in basis_B_q]
            if profile:
                phase_times["basis"] += perf_counter() - t_basis

            k_A, k_B = len(basis_A_q), len(basis_B_q)
            series_list = neg_basis_A_q + basis_B_target_q
            L0 = k_A + k_B + L0_margin

            t_prune = perf_counter() if profile else None
            split_scan = _scan_series_list(series_list)
            split_ok, _ = _quick_split_feasible(split_scan, L0, verification_order)
            if profile:
                phase_times["prune"] += perf_counter() - t_prune
            if not split_ok:
                diagnostics["precision_limited_splits"] += 1
                continue

            k_min = split_scan["min_val"]
            rows_available = _usable_row_count(split_scan)
            row_target = _screening_row_target(
                rows_available, k_min, L0, Delta, verification_order
            )

            t_screen = perf_counter() if profile else None
            screen_info = _screen_split_mod_p(
                series_list,
                row_target,
                prime_candidates=mod_p_screening_primes,
                prime_limit=mod_p_screening_prime_limit,
                verbose=verbose,
                pre_scan=split_scan,
            )
            if profile:
                phase_times["screen"] += perf_counter() - t_screen

            diagnostics["screen_good_primes"] += screen_info.get("good_prime_count", 0)
            diagnostics["screen_bad_primes"] += screen_info.get("bad_prime_count", 0)

            best_screen = screen_info.get("best")
            best_nullity = 0 if best_screen is None else int(best_screen.get("nullity", 0))
            diagnostics["max_kernel_dim_seen"] = max(
                diagnostics["max_kernel_dim_seen"], best_nullity
            )

            if screen_info.get("rejected", False):
                diagnostics["no_kernel_splits"] += 1
                diagnostics["screen_rejected_splits"] += 1
                continue

            if best_screen is None:
                diagnostics["precision_limited_splits"] += 1
                continue

            diagnostics["kernel_splits"] += 1

            if best_nullity > max(1, int(mod_p_exact_free_col_limit)):
                diagnostics["kernel_but_no_valid_splits"] += 1
                diagnostics["ambiguous_kernel_splits"] += 1
                continue

            t_matrix = perf_counter() if profile else None
            M_exact, _, row_start_eff, row_end_eff = assemble_matrix_generic_range(
                series_list, 0, row_target, verbose=verbose, pre_scan=split_scan
            )
            if profile:
                phase_times["matrix"] += perf_counter() - t_matrix
            rows_used = row_end_eff
            if M_exact is None or rows_used <= row_start_eff:
                diagnostics["precision_limited_splits"] += 1
                continue

            candidate_vectors = _candidate_vectors_from_mod_p_screen(
                M_exact,
                best_screen,
                max_candidates=mod_p_exact_free_col_limit,
            )

            split_has_valid_solution = False
            min_required_valuation = max(k_min + rows_used, verification_order)

            for candidate_vec in candidate_vectors:
                if not any(candidate_vec[i] != 0 for i in range(k_A)):
                    continue
                if not any(candidate_vec[k_A + i] != 0 for i in range(k_B)):
                    continue

                diagnostics["candidate_vectors"] += 1

                t_solve = perf_counter() if profile else None
                residual = M_exact * candidate_vec
                if profile:
                    phase_times["solve"] += perf_counter() - t_solve
                if not residual.is_zero():
                    continue

                t_verify = perf_counter() if profile else None
                A_q_chk = sum(candidate_vec[i] * basis_A_q[i] for i in range(k_A))
                B_q_chk = sum(candidate_vec[k_A + i] * basis_B_q[i] for i in range(k_B))
                S = B_q_chk * target_f - A_q_chk
                if profile:
                    phase_times["verify"] += perf_counter() - t_verify
                if S.valuation() < min_required_valuation:
                    continue

                final_coeffs = normalize_vector_to_ints(candidate_vec)
                A_poly = reconstruct_polynomial(
                    final_coeffs[:k_A], deg_A_total, max_deg_y, P_xy, exp_A
                )
                B_poly = reconstruct_polynomial(
                    final_coeffs[k_A:], deg_B_total, max_deg_y, P_xy, exp_B
                )
                sol = {
                    "A_poly": A_poly,
                    "B_poly": B_poly,
                    "deg_A_total": deg_A_total,
                    "deg_B_total": deg_B_total,
                    "total_degree": total_degree,
                    "max_deg_y": max_deg_y,
                }
                if include_q_series:
                    sol["A_q"] = sum(final_coeffs[i] * basis_A_q[i] for i in range(k_A))
                    sol["B_q"] = sum(final_coeffs[k_A + i] * basis_B_q[i] for i in range(k_B))

                solutions.append(sol)
                degree_has_solution = True
                split_has_valid_solution = True
                diagnostics["valid_solutions"] += 1
                if verbose:
                    print(f"  [Found] Solution at degree {total_degree}")

            if not split_has_valid_solution:
                diagnostics["kernel_but_no_valid_splits"] += 1
                if best_nullity > 1:
                    diagnostics["ambiguous_kernel_splits"] += 1

        if degree_has_solution and stop_at_first_total_degree:
            break

    if profile:
        print(
            "[Profiler] "
            f"basis={phase_times['basis']:.3f}s, "
            f"prune={phase_times['prune']:.3f}s, "
            f"screen={phase_times['screen']:.3f}s, "
            f"matrix={phase_times['matrix']:.3f}s, "
            f"solve={phase_times['solve']:.3f}s, "
            f"verify={phase_times['verify']:.3f}s"
        )
    return (solutions, diagnostics) if return_diagnostics else solutions

def AB_ratio(sol):
    """Return the rational function A/B extracted from one solver output record."""
    if not sol:
        return None
    return sol['A_poly'] / sol['B_poly']

# ==============================================================================
# 3. Search Runtime, Family Recovery, and Report Construction
# ==============================================================================

def _adjust_start_degree(N, start_deg):
    """Apply a small per-level lower bound before generic degree scanning starts."""
    if N == 67 and start_deg < 20:
        return 20
    if N == 163 and start_deg < 45:
        return 45
    return start_deg


def _degree_search_hints(N):
    """Return fixed search windows for levels where the public search is pre-tuned."""
    hints = {
        43: {
            'u':  {'start': 8, 'min': 8, 'max': 8, 'step': 1, 'max_deg_y': 3},
            'uN': {'start': 8, 'min': 8, 'max': 8, 'step': 1, 'max_deg_y': 3},
            'v':  {'start': 12, 'min': 12, 'max': 12, 'step': 1, 'max_deg_y': 3},
            'vN': {'start': 12, 'min': 12, 'max': 12, 'step': 1, 'max_deg_y': 3},
        },
        67: {
            'u':  {'start': 15, 'min': 15, 'max': 15, 'step': 1, 'max_deg_y': 6},
            'uN': {'start': 15, 'min': 15, 'max': 15, 'step': 1, 'max_deg_y': 6},
            'v':  {'start': 19, 'min': 19, 'max': 19, 'step': 1, 'max_deg_y': 6},
            'vN': {'start': 19, 'min': 19, 'max': 19, 'step': 1, 'max_deg_y': 6},
        },
        163: {
            'u':  {'start': 34, 'min': 34, 'max': 34, 'step': 1, 'max_deg_y': 11},
            'uN': {'start': 34, 'min': 34, 'max': 34, 'step': 1, 'max_deg_y': 11},
            'v':  {'start': 38, 'min': 38, 'max': 38, 'step': 1, 'max_deg_y': 12},
            'vN': {'start': 38, 'min': 38, 'max': 38, 'step': 1, 'max_deg_y': 12},
        },
    }
    return hints.get(int(N), {})


def _dedupe_positive_ints(values):
    """Normalize a list of candidate integers by removing duplicates and nonpositive values."""
    deduped = []
    for value in values:
        if value is None:
            continue
        ivalue = max(1, int(value))
        if ivalue not in deduped:
            deduped.append(ivalue)
    return deduped


def _max_deg_y_candidates(runtime, total_deg, preferred=None, explicit_plan=None):
    """Choose one or more y-degree caps for a given total degree search step."""
    if explicit_plan is not None:
        expanded = []
        for value in explicit_plan:
            if value is None:
                expanded.append(max(1, int(total_deg) // 3))
            else:
                expanded.append(value)
        return _dedupe_positive_ints(expanded)

    family_type = runtime['model_data'].get(
        'quadratic_family_type',
        runtime['model_data'].get('type', 'general')
    )
    if family_type == 'elliptic':
        return [1]

    default_max_deg_y = max(1, int(total_deg) // 3)
    curve_y_degree = runtime.get('curve_y_degree', None)

    candidates = []
    if preferred is not None:
        candidates.append(preferred)

    if curve_y_degree is not None and curve_y_degree >= 2:
        candidates.append(min(default_max_deg_y, curve_y_degree - 1))

    candidates.append(default_max_deg_y)

    if family_type == 'hyperelliptic':
        candidates.insert(0, 1)

    return _dedupe_positive_ints(candidates)


def get_context(runtime, work_prec):
    """Build or reuse the q-expansion context at the requested precision."""
    p = int(work_prec)
    context_cache = runtime['context_cache']
    if p not in context_cache:
        forms_p = setup_modular_forms(runtime['N'], p)
        x_p, y_p = runtime['model_data']['generators'](forms_p['q'])
        context_cache[p] = {
            'prec': p,
            'forms': forms_p,
            'x0': x_p,
            'y0': y_p,
            'solver_cache_by_deg_y': {}
        }
    return context_cache[p]


def get_solver_cache_for_context(ctx, max_deg_y):
    """Return the per-precision solver cache associated with one y-degree cap."""
    key = int(max_deg_y)
    cache_by_deg_y = ctx['solver_cache_by_deg_y']
    if key not in cache_by_deg_y:
        cache_by_deg_y[key] = {}
    return cache_by_deg_y[key]


def monomial_count(deg, max_deg_y):
    """Count monomials x^i y^j with i + j <= deg and j <= max_deg_y."""
    j_max = min(deg, max_deg_y)
    return (j_max + 1) * (deg + 1) - (j_max * (j_max + 1)) // 2


def min_monomial_valuation(runtime, deg, max_deg_y):
    """Estimate the smallest q-valuation among monomials in the search basis."""
    j_max = min(deg, max_deg_y)
    best_val = None
    val_x0 = runtime['val_x0']
    val_y0 = runtime['val_y0']
    for j in range(j_max + 1):
        i = (deg - j) if val_x0 < 0 else 0
        v = i * val_x0 + j * val_y0
        if best_val is None or v < best_val:
            best_val = v
    return int(best_val if best_val is not None else 0)


def estimate_required_prec(runtime, target_key, total_deg, max_deg_y):
    """Estimate a conservative working precision for one degree window."""
    req = runtime['prec_min_floor']
    target_val = runtime['target_valuations'].get(target_key, 0)
    for deg_B_total in range(total_deg + 1):
        deg_A_total = total_deg - deg_B_total
        k_A = monomial_count(deg_A_total, max_deg_y)
        k_B = monomial_count(deg_B_total, max_deg_y)
        L0 = k_A + k_B + runtime['solver_L0_margin']
        k_min_A = min_monomial_valuation(runtime, deg_A_total, max_deg_y)
        k_min_B = min_monomial_valuation(runtime, deg_B_total, max_deg_y) + target_val
        k_min = min(k_min_A, k_min_B)
        req_split = k_min + L0 + max(0, runtime['solver_Delta']) + runtime['prec_est_safety']
        req = max(req, req_split)
    req = int(max(req, runtime['prec_min_floor']))
    step = runtime['prec_quant_step']
    req = ((req + step - 1) // step) * step
    req = max(min(runtime['prec'], req), min(runtime['prec'], runtime['prec_min_floor']))
    return req


def get_attempt_precisions(runtime, target_key, total_deg, max_deg_y):
    """Choose the working precisions tried for one search window."""
    if not runtime['use_dual_precision_consensus']:
        return [runtime['prec']]
    if total_deg > runtime['prec_adaptive_deg_cap']:
        return [runtime['prec']]
    est_prec = estimate_required_prec(runtime, target_key, total_deg, max_deg_y)
    if est_prec >= runtime['prec']:
        return [runtime['prec']]
    if (runtime['prec'] - est_prec) < runtime['dual_prec_min_gap']:
        return [runtime['prec']]
    return [est_prec, runtime['prec']]


def verify_rat_on_full_precision(runtime, rat, target_key):
    """Check a recovered rational function against the full-precision target q-series."""
    if rat is None:
        return False
    try:
        full_ctx = runtime['full_ctx']
        S = rat(x=full_ctx['x0'], y=full_ctx['y0']) - full_ctx['forms'][target_key]
        return S.valuation() >= runtime['verify_order_full_prec']
    except Exception:
        return False


def _elliptic_pole_weights(runtime):
    """Return pole weights inferred from the q-valuations of x0 and y0."""
    wx = max(1, -int(runtime.get('val_x0', -2)))
    wy = max(1, -int(runtime.get('val_y0', -3)))
    return wx, wy


def _elliptic_monomial_exponents(runtime, pole_cap, y_cap=1):
    """Enumerate monomials ordered by weighted pole size for elliptic levels."""
    wx, wy = _elliptic_pole_weights(runtime)
    exponents = []
    for j in range(max(0, int(y_cap)) + 1):
        remaining = int(pole_cap) - j * wy
        if remaining < 0:
            continue
        max_i = remaining // wx
        for i in range(max_i + 1):
            exponents.append((i, j))
    exponents.sort(key=lambda ij: (wx * ij[0] + wy * ij[1], ij[1], ij[0]))
    return exponents


def _get_elliptic_basis_data(runtime, ctx, pole_cap):
    """Cache the elliptic weighted basis attached to one pole bound."""
    cache = ctx.setdefault('elliptic_basis_by_cap', {})
    key = int(pole_cap)
    if key not in cache:
        exponents = _elliptic_monomial_exponents(runtime, key, y_cap=1)
        basis_q = [ctx['x0']**i * ctx['y0']**j for (i, j) in exponents]
        cache[key] = {
            'exponents': exponents,
            'basis_q': basis_q,
        }
    return cache[key]


def _run_elliptic_pole_search(target_key, runtime, label, start_cap, max_cap, step=1,
                              progress_cb=None):
    """Search elliptic levels using a single weighted pole cap for A and B."""
    P_xy = runtime['P_xy']
    start_cap = max(1, int(start_cap))
    max_cap = max(start_cap, int(max_cap))
    step = max(1, int(step))

    for pole_cap in range(start_cap, max_cap + 1, step):
        attempt_precs = get_attempt_precisions(runtime, target_key, pole_cap, 1)

        for work_prec in attempt_precs:
            if progress_cb is not None:
                progress_cb(label, pole_cap, max_cap, 'pole', work_prec)

            ctx = get_context(runtime, work_prec)
            target_f = ctx['forms'][target_key]
            basis_data = _get_elliptic_basis_data(runtime, ctx, pole_cap)
            exponents = basis_data['exponents']
            basis_q = basis_data['basis_q']
            k = len(basis_q)
            if k == 0:
                continue

            series_list = [-basis for basis in basis_q] + [basis * target_f for basis in basis_q]
            L0 = 2 * k + runtime['solver_L0_margin']
            split_scan = _scan_series_list(series_list)
            split_ok, _ = _quick_split_feasible(split_scan, L0, runtime['solver_verification_order'])
            if not split_ok:
                continue

            M_L0, k_min, L0_eff = assemble_matrix_generic(
                series_list, L0, verbose=False, pre_scan=split_scan
            )
            if M_L0 is None or L0_eff <= 0:
                continue

            K_L0 = M_L0.right_kernel()
            if K_L0.dimension() == 0:
                continue

            min_required_valuation = max(k_min + L0_eff, runtime['solver_verification_order'])
            M_ext = None
            ext_has_new_rows = None
            L_ext_target = L0_eff + max(0, runtime['solver_Delta'])

            for candidate_vec in K_L0.basis():
                if not any(candidate_vec[i] != 0 for i in range(k)):
                    continue
                if not any(candidate_vec[k + i] != 0 for i in range(k)):
                    continue

                is_valid = True
                if runtime['solver_Delta'] > 0:
                    if ext_has_new_rows is None:
                        M_ext, _, rows_ext = assemble_matrix_generic(
                            series_list, L_ext_target, verbose=False, pre_scan=split_scan
                        )
                        ext_has_new_rows = (M_ext is not None and rows_ext > L0_eff)
                    if ext_has_new_rows and not (M_ext * candidate_vec).is_zero():
                        is_valid = False

                if is_valid:
                    A_q_chk = sum(candidate_vec[i] * basis_q[i] for i in range(k))
                    B_q_chk = sum(candidate_vec[k + i] * basis_q[i] for i in range(k))
                    S = B_q_chk * target_f - A_q_chk
                    if S.valuation() < min_required_valuation:
                        is_valid = False
                if not is_valid:
                    continue

                final_coeffs = normalize_vector_to_ints(candidate_vec)
                A_poly = reconstruct_polynomial(final_coeffs[:k], pole_cap, 1, P_xy, exponents)
                B_poly = reconstruct_polynomial(final_coeffs[k:], pole_cap, 1, P_xy, exponents)
                rat_candidate = A_poly / B_poly
                if not verify_rat_on_full_precision(runtime, rat_candidate, target_key):
                    continue

                return rat_candidate, pole_cap, {
                    'max_deg_y': 1,
                    'search_deg': pole_cap,
                    'work_prec': work_prec,
                    'basis_mode': 'elliptic_pole',
                    'basis_size': k,
                    'pole_cap': pole_cap,
                }

    return None, None, None


def _run_elliptic_split_pole_search(target_key, runtime, label, start_cap, max_cap, step=1,
                                    progress_cb=None):
    """Fallback elliptic search that splits the pole cap between numerator and denominator."""
    P_xy = runtime['P_xy']
    start_cap = max(1, int(start_cap))
    max_cap = max(start_cap, int(max_cap))
    step = max(1, int(step))

    for cap_total in range(start_cap, max_cap + 1, step):
        attempt_precs = get_attempt_precisions(runtime, target_key, cap_total, 1)

        for cap_B in range(cap_total + 1):
            cap_A = cap_total - cap_B

            for work_prec in attempt_precs:
                if progress_cb is not None:
                    progress_cb(label, cap_total, max_cap, f'split:{cap_A}/{cap_B}', work_prec)

                ctx = get_context(runtime, work_prec)
                target_f = ctx['forms'][target_key]
                basis_A = _get_elliptic_basis_data(runtime, ctx, cap_A)
                basis_B = _get_elliptic_basis_data(runtime, ctx, cap_B)
                exp_A = basis_A['exponents']
                exp_B = basis_B['exponents']
                basis_A_q = basis_A['basis_q']
                basis_B_q = basis_B['basis_q']
                k_A = len(basis_A_q)
                k_B = len(basis_B_q)
                if k_A == 0 or k_B == 0:
                    continue

                series_list = [-basis for basis in basis_A_q] + [basis * target_f for basis in basis_B_q]
                L0 = k_A + k_B + runtime['solver_L0_margin']
                split_scan = _scan_series_list(series_list)
                split_ok, _ = _quick_split_feasible(split_scan, L0, runtime['solver_verification_order'])
                if not split_ok:
                    continue

                M_L0, k_min, L0_eff = assemble_matrix_generic(
                    series_list, L0, verbose=False, pre_scan=split_scan
                )
                if M_L0 is None or L0_eff <= 0:
                    continue

                K_L0 = M_L0.right_kernel()
                if K_L0.dimension() == 0:
                    continue

                min_required_valuation = max(k_min + L0_eff, runtime['solver_verification_order'])
                M_ext = None
                ext_has_new_rows = None
                L_ext_target = L0_eff + max(0, runtime['solver_Delta'])

                for candidate_vec in K_L0.basis():
                    if not any(candidate_vec[i] != 0 for i in range(k_A)):
                        continue
                    if not any(candidate_vec[k_A + i] != 0 for i in range(k_B)):
                        continue

                    is_valid = True
                    if runtime['solver_Delta'] > 0:
                        if ext_has_new_rows is None:
                            M_ext, _, rows_ext = assemble_matrix_generic(
                                series_list, L_ext_target, verbose=False, pre_scan=split_scan
                            )
                            ext_has_new_rows = (M_ext is not None and rows_ext > L0_eff)
                        if ext_has_new_rows and not (M_ext * candidate_vec).is_zero():
                            is_valid = False

                    if is_valid:
                        A_q_chk = sum(candidate_vec[i] * basis_A_q[i] for i in range(k_A))
                        B_q_chk = sum(candidate_vec[k_A + i] * basis_B_q[i] for i in range(k_B))
                        S = B_q_chk * target_f - A_q_chk
                        if S.valuation() < min_required_valuation:
                            is_valid = False
                    if not is_valid:
                        continue

                    final_coeffs = normalize_vector_to_ints(candidate_vec)
                    A_poly = reconstruct_polynomial(final_coeffs[:k_A], cap_A, 1, P_xy, exp_A)
                    B_poly = reconstruct_polynomial(final_coeffs[k_A:], cap_B, 1, P_xy, exp_B)
                    rat_candidate = A_poly / B_poly
                    if not verify_rat_on_full_precision(runtime, rat_candidate, target_key):
                        continue

                    return rat_candidate, cap_total, {
                        'max_deg_y': 1,
                        'search_deg': cap_total,
                        'work_prec': work_prec,
                        'basis_mode': 'elliptic_split_pole',
                        'cap_A': cap_A,
                        'cap_B': cap_B,
                        'basis_size_A': k_A,
                        'basis_size_B': k_B,
                    }

    return None, None, None


def _build_search_runtime(N, model_data, P_xy, prec):
    """Assemble the reusable runtime dictionary for one level computation."""
    truncated_generator_verify_levels = set()
    effective_prec = int(prec)
    if N in truncated_generator_verify_levels:
        effective_prec = min(effective_prec, 120)
    runtime = {
        'N': N,
        'model_data': model_data,
        'P_xy': P_xy,
        'prec': effective_prec,
        'context_cache': {},
        'solver_L0_margin': 5,
        'solver_Delta': 30,
        'solver_verification_order': 100,
        'prec_est_safety': 20,
        'prec_quant_step': 20,
        'dual_prec_min_gap': 30,
    }
    full_ctx = get_context(runtime, runtime['prec'])
    runtime['full_ctx'] = full_ctx

    verify_order_full_prec = min(
        max(runtime['solver_verification_order'] + 10, 470),
        max(50, runtime['prec'] - 5)
    )
    if N == 163:
        verify_order_full_prec = max(
            verify_order_full_prec,
            min(max(250, runtime['prec'] // 2), max(80, runtime['prec'] - 5))
        )
    if N in truncated_generator_verify_levels:
        verify_order_full_prec = min(110, max(50, runtime['prec'] - 5))
    runtime['verify_order_full_prec'] = verify_order_full_prec
    runtime['prec_min_floor'] = max(120, runtime['solver_verification_order'] + 10)
    family_type = model_data.get('quadratic_family_type', model_data.get('type', 'general'))
    if N >= 20:
        runtime['use_dual_precision_consensus'] = True
        if family_type == 'elliptic':
            runtime['prec_adaptive_deg_cap'] = 40
        elif family_type == 'hyperelliptic':
            runtime['prec_adaptive_deg_cap'] = 55
        else:
            runtime['prec_adaptive_deg_cap'] = 65
    else:
        runtime['use_dual_precision_consensus'] = False
        runtime['prec_adaptive_deg_cap'] = 30 if N >= 67 else 25
    if N in truncated_generator_verify_levels:
        runtime['use_dual_precision_consensus'] = False

    try:
        runtime['val_x0'] = int(full_ctx['x0'].valuation())
    except Exception:
        runtime['val_x0'] = -2
    try:
        runtime['val_y0'] = int(full_ctx['y0'].valuation())
    except Exception:
        runtime['val_y0'] = -3

    target_valuations = {}
    for key in ['f', 'fN', 'g', 'gN']:
        try:
            target_valuations[key] = int(full_ctx['forms'][key].valuation())
        except Exception:
            target_valuations[key] = 0
    runtime['target_valuations'] = target_valuations

    try:
        x_tmp, y_tmp = P_xy.gens()
        curve_y_degree = model_data['poly_xy'](x_tmp, y_tmp).degree(y_tmp)
        if curve_y_degree <= 0:
            curve_y_degree = None
    except Exception:
        curve_y_degree = None
    runtime['curve_y_degree'] = curve_y_degree
    runtime['degree_search_hints'] = _degree_search_hints(N)
    wx, wy = _elliptic_pole_weights(runtime)
    runtime['elliptic_pole_step'] = 1
    runtime['elliptic_pole_cap_max'] = max(120, 100 + wy)
    runtime['elliptic_split_pole_cap_max'] = max(320, 2 * runtime['elliptic_pole_cap_max'])
    runtime['elliptic_split_fallback'] = True
    # User-specified monotone chain:
    #   deg(u) <= deg(uN) <= deg(v) <= deg(vN)
    # Keep a relaxed fallback if this heuristic fails on a level.
    runtime['strict_degree_chain'] = True
    runtime['strict_degree_chain_fallback'] = True
    return runtime


def _run_degree_search_window(target_key, runtime, label, start_deg, max_deg, step,
                              preferred_max_deg_y=None, explicit_max_deg_y_plan=None,
                              progress_cb=None):
    """Run the default degree window search used by the public generic solver."""
    P_xy = runtime['P_xy']
    next_min_by_y = {}

    for deg in range(start_deg, max_deg + 1, step):
        y_candidates = _max_deg_y_candidates(
            runtime, deg,
            preferred=preferred_max_deg_y,
            explicit_plan=explicit_max_deg_y_plan
        )

        for max_deg_y in y_candidates:
            min_total_degree = next_min_by_y.get(max_deg_y, deg)
            y_label = str(max_deg_y)
            attempt_precs = get_attempt_precisions(runtime, target_key, deg, max_deg_y)

            for work_prec in attempt_precs:
                if progress_cb is not None:
                    progress_cb(label, deg, max_deg, y_label, work_prec)

                ctx = get_context(runtime, work_prec)
                target_f = ctx['forms'][target_key]
                x0 = ctx['x0']
                y0 = ctx['y0']
                local_cache = get_solver_cache_for_context(ctx, max_deg_y)

                sols = find_minimal_robust_solution(
                    target_f, x0, y0, P_xy,
                    min_total_degree=min_total_degree,
                    max_total_degree=deg,
                    max_deg_y=max_deg_y,
                    L0_margin=runtime['solver_L0_margin'],
                    Delta=runtime['solver_Delta'],
                    verification_order=runtime['solver_verification_order'],
                    verbose=False,
                    use_cache=True,
                    precomputed=local_cache,
                    profile=False,
                    include_q_series=False
                )

                if not sols:
                    continue

                cand = sols[0]
                rat_candidate = AB_ratio(cand)
                if not verify_rat_on_full_precision(runtime, rat_candidate, target_key):
                    continue

                return rat_candidate, cand['total_degree'], {
                    'max_deg_y': cand.get('max_deg_y', max_deg_y),
                    'search_deg': deg,
                    'work_prec': work_prec,
                }

        for max_deg_y in y_candidates:
            next_min_by_y[max_deg_y] = deg + 1

    return None, None, None


def _find_map_with_plan(target_key, runtime, label, start_deg=10, max_deg=100, step=5,
                        preferred_max_deg_y=None, degree_hint=None, progress_cb=None):
    """Recover one target map by combining the level-specific search heuristics."""
    N = runtime['N']
    family_type = runtime['model_data'].get(
        'quadratic_family_type',
        runtime['model_data'].get('type', 'general')
    )
    if family_type == 'elliptic':
        start_cap = max(1, int(start_deg))
        max_cap = max(start_cap, int(runtime.get('elliptic_pole_cap_max', 120)))
        step_cap = max(1, int(runtime.get('elliptic_pole_step', 1)))
        rat, found_deg, meta = _run_elliptic_pole_search(
            target_key, runtime, label,
            start_cap, max_cap, step=step_cap,
            progress_cb=progress_cb,
        )
        if rat is not None:
            if meta is not None:
                meta['used_degree_hint'] = False
            return rat, found_deg, meta
        if runtime.get('elliptic_split_fallback', False):
            split_max_cap = max(start_cap, int(runtime.get('elliptic_split_pole_cap_max', max_cap)))
            rat, found_deg, meta = _run_elliptic_split_pole_search(
                target_key, runtime, label,
                start_cap, split_max_cap, step=step_cap,
                progress_cb=progress_cb,
            )
            if meta is not None:
                meta['used_degree_hint'] = False
            return rat, found_deg, meta
        return None, None, None

    generic_start_deg = _adjust_start_degree(N, start_deg)
    if degree_hint is not None:
        hinted_start = int(degree_hint.get('start', start_deg))
        hinted_min = int(degree_hint.get('min', hinted_start))
        hinted_max = min(max_deg, int(degree_hint.get('max', hinted_start)))
        hinted_step = max(1, int(degree_hint.get('step', 1)))
        hinted_plan = [int(degree_hint['max_deg_y'])] if degree_hint.get('max_deg_y') is not None else None
        rat, found_deg, meta = _run_degree_search_window(
            target_key, runtime, label,
            hinted_min, hinted_max, hinted_step,
            preferred_max_deg_y=preferred_max_deg_y,
            explicit_max_deg_y_plan=hinted_plan,
            progress_cb=progress_cb,
        )
        if rat is not None:
            if meta is not None:
                meta['used_degree_hint'] = True
            return rat, found_deg, meta

    rat, found_deg, meta = _run_degree_search_window(
        target_key, runtime, label,
        generic_start_deg, max_deg, step,
        preferred_max_deg_y=preferred_max_deg_y,
        explicit_max_deg_y_plan=None,
        progress_cb=progress_cb,
    )
    if meta is not None:
        meta['used_degree_hint'] = False
    return rat, found_deg, meta


def _solve_maps_with_plan(runtime, solve_plan, progress_cb=None):
    """Run the staged public plan that recovers u, u_N, v, and v_N in order."""
    maps = {}
    degrees = {}
    search_meta = {}

    for entry in solve_plan:
        key = entry['key']
        target_key = entry['target_key']
        label = entry['label']
        start_from = entry.get('start_from', None)

        if start_from is None:
            start_deg = 10
            preferred_max_deg_y = None
        else:
            start_deg = degrees.get(start_from, 10)
            preferred_max_deg_y = search_meta.get(start_from, {}).get('max_deg_y', None)

        rat, found_deg, meta = _find_map_with_plan(
            target_key, runtime, label,
            start_deg=start_deg,
            preferred_max_deg_y=preferred_max_deg_y,
            degree_hint=runtime['degree_search_hints'].get(key, None),
            progress_cb=progress_cb
        )
        if rat is None:
            return None

        maps[key] = rat
        degrees[key] = found_deg
        search_meta[key] = meta or {}

    return {
        'maps': maps,
        'degrees': degrees,
        'search_meta': search_meta,
    }

def _find_fixed_degree_map_mod_p(target_key, runtime, label, total_degree, max_deg_y,
                                 progress_cb=None, prime_limit=8, free_col_limit=8):
    """
    Recover one map at a fixed degree using mod-p screening before exact lifting.

    This helper is reserved for the public N = 131 path, where the generic
    exact-kernel search is noticeably less effective.
    """
    P_xy = runtime['P_xy']
    attempt_precs = get_attempt_precisions(runtime, target_key, total_degree, max_deg_y)

    for work_prec in attempt_precs:
        if progress_cb is not None:
            progress_cb(label, total_degree, total_degree, str(max_deg_y), work_prec)

        ctx = get_context(runtime, work_prec)
        target_f = ctx['forms'][target_key]
        x0 = ctx['x0']
        y0 = ctx['y0']
        local_cache = get_solver_cache_for_context(ctx, max_deg_y)

        sols = find_minimal_robust_solution_mod_p(
            target_f, x0, y0, P_xy,
            min_total_degree=total_degree,
            max_total_degree=total_degree,
            max_deg_y=max_deg_y,
            L0_margin=runtime['solver_L0_margin'],
            Delta=runtime['solver_Delta'],
            verification_order=runtime['solver_verification_order'],
            verbose=False,
            use_cache=True,
            precomputed=local_cache,
            profile=False,
            include_q_series=False,
            mod_p_screening_prime_limit=prime_limit,
            mod_p_exact_free_col_limit=free_col_limit,
        )

        for cand in sols:
            rat_candidate = AB_ratio(cand)
            if rat_candidate is None:
                continue
            if not verify_rat_on_full_precision(runtime, rat_candidate, target_key):
                continue
            return rat_candidate, cand['total_degree'], {
                'max_deg_y': cand.get('max_deg_y', max_deg_y),
                'search_deg': total_degree,
                'work_prec': work_prec,
                'solver_mode': 'fixed_degree_mod_p',
            }

    return None, None, None

def _solve_maps_for_level_131(prec=500, progress_cb=None):
    """
    Solve N = 131 using the fixed-degree public reconstruction path.

    The final maps occur at known degrees, so this path avoids the generic
    exact-kernel scan and instead applies mod-p screening on those fixed windows.
    """
    N = 131
    model_data = get_model_data(N)
    P_xy = PolynomialRing(QQ, names=['x', 'y'])
    runtime = _build_search_runtime(N, model_data, P_xy, prec)

    maps = {}
    degrees = {}
    search_meta = {}
    for entry in SPECIAL_LEVEL_131_FIXED_PLAN:
        rat, found_deg, meta = _find_fixed_degree_map_mod_p(
            entry['target_key'],
            runtime,
            entry['label'],
            entry['degree'],
            entry['max_deg_y'],
            progress_cb=progress_cb,
            prime_limit=8,
            free_col_limit=8,
        )
        if rat is None:
            return None
        maps[entry['key']] = rat
        degrees[entry['key']] = found_deg
        search_meta[entry['key']] = meta or {}

    return {
        'N': N,
        'model_data': model_data,
        'P_xy': P_xy,
        'runtime': runtime,
        'maps': {
            'u': maps['u'],
            'uN': maps['uN'],
            'v': maps['v'],
            'vN': maps['vN'],
        },
        'degrees': degrees,
        'search_meta': search_meta,
    }


def _solve_maps_for_level(N, prec=500, progress_cb=None, strict_degree_chain=None):
    """Recover the four coefficient maps for a single level N."""
    if int(N) == 131:
        return _solve_maps_for_level_131(prec=prec, progress_cb=progress_cb)

    model_data = get_model_data(N)
    P_xy = PolynomialRing(QQ, names=['x', 'y'])
    runtime = _build_search_runtime(N, model_data, P_xy, prec)
    if strict_degree_chain is not None:
        runtime['strict_degree_chain'] = bool(strict_degree_chain)

    strict_plan = [
        {'key': 'u', 'target_key': 'f', 'label': 'u'},
        {'key': 'uN', 'target_key': 'fN', 'label': 'u_N', 'start_from': 'u'},
        {'key': 'v', 'target_key': 'g', 'label': 'v', 'start_from': 'uN'},
        {'key': 'vN', 'target_key': 'gN', 'label': 'v_N', 'start_from': 'v'},
    ]
    relaxed_plan = [
        {'key': 'u', 'target_key': 'f', 'label': 'u'},
        {'key': 'uN', 'target_key': 'fN', 'label': 'u_N', 'start_from': 'u'},
        {'key': 'v', 'target_key': 'g', 'label': 'v', 'start_from': 'u'},
        {'key': 'vN', 'target_key': 'gN', 'label': 'v_N', 'start_from': 'v'},
    ]

    if runtime['strict_degree_chain']:
        solved = _solve_maps_with_plan(runtime, strict_plan, progress_cb=progress_cb)
        if solved is None and runtime['strict_degree_chain_fallback']:
            solved = _solve_maps_with_plan(runtime, relaxed_plan, progress_cb=progress_cb)
    else:
        solved = _solve_maps_with_plan(runtime, relaxed_plan, progress_cb=progress_cb)

    if solved is None:
        return None

    return {
        'N': N,
        'model_data': model_data,
        'P_xy': P_xy,
        'runtime': runtime,
        'maps': {
            'u': solved['maps']['u'],
            'uN': solved['maps']['uN'],
            'v': solved['maps']['v'],
            'vN': solved['maps']['vN'],
        },
        'degrees': solved['degrees'],
        'search_meta': solved['search_meta'],
    }

# ==============================================================================
# Logic: Function Field Arithmetic
# ==============================================================================

def _ratfunc_to_latex(rat_func_xy, L, t, alpha_t, R_disp, t_d, a_d, factor_output=True):
    """Convert a rational function on X_0(N) into display-ready LaTeX over the family variables."""
    if rat_func_xy is None:
        return r"\text{Undefined}"

    num_xy = rat_func_xy.numerator()
    den_xy = rat_func_xy.denominator()

    def _poly_terms(poly):
        if hasattr(poly, 'dict'):
            return poly.dict().items()
        return [((0, 0), poly)]

    def _univariate_terms(poly):
        if hasattr(poly, 'dict'):
            return poly.dict().items()
        return [(0, poly)]

    def to_L(poly):
        val = L(0)
        for (i, j), coeff in _poly_terms(poly):
            val += coeff * (t**i) * (alpha_t**j)
        return val

    try:
        val_L = to_L(num_xy) / to_L(den_xy)
    except ZeroDivisionError:
        return r"\infty"

    coeffs = val_L.list()
    denoms = [c.denominator() for c in coeffs]
    common_poly_den = _primitive_integer_polynomial(lcm(denoms))
    scaled_coeffs = [c * common_poly_den for c in coeffs]

    all_polys = scaled_coeffs
    scalar_denoms = []
    for poly in all_polys:
        p_obj = poly.numerator()
        scalar_denoms.extend([c.denominator() for c in p_obj.coefficients()])

    scalar_lcm = lcm(scalar_denoms)
    final_num_coeffs = [(p * scalar_lcm).numerator() for p in scaled_coeffs]
    final_den_poly = common_poly_den * scalar_lcm

    poly_num = R_disp(0)
    for i, coeff_poly in enumerate(final_num_coeffs):
        c_int = 0
        for deg, c_val in _univariate_terms(coeff_poly):
            c_int += ZZ(c_val) * (t_d**deg)
        poly_num += c_int * (a_d**i)

    poly_den = R_disp(0)
    for deg, c_val in _univariate_terms(final_den_poly):
        poly_den += ZZ(c_val) * (t_d**deg)

    shared_coeffs = [ZZ(c) for c in poly_num.coefficients() if c != 0]
    shared_coeffs.extend(ZZ(c) for c in poly_den.coefficients() if c != 0)
    if shared_coeffs:
        shared_content = gcd(shared_coeffs)
        if shared_content not in [0, 1]:
            poly_num //= shared_content
            poly_den //= shared_content

    try:
        lc = final_den_poly.leading_coefficient()
        if lc < 0:
            poly_num = -poly_num
            poly_den = -poly_den
    except Exception:
        pass

    if factor_output:
        num_obj = poly_num.factor()
        den_obj = poly_den.factor()
    else:
        num_obj = poly_num
        den_obj = poly_den

    if poly_den == 1:
        return str(latex(num_obj))
    return r"\frac{" + str(latex(num_obj)) + "}{" + str(latex(den_obj)) + "}"


def _family_argument_signature(model_data):
    """Return the argument list used in displayed formulas for one family type."""
    family_type = model_data.get('quadratic_family_type', 'general')
    parameter_var = model_data.get('family_parameter_var', 't')
    extension_var = model_data.get('family_extension_var', 'alpha_t')
    if family_type == 'elliptic':
        return parameter_var
    return f"{parameter_var},{extension_var}"


def _build_family_context(model_data):
    """Construct the function-field and display context used for final report generation."""
    parameter_var = model_data.get('family_parameter_var', 't')
    extension_var = model_data.get('family_extension_var', 'alpha_t')
    poly_xy_func = model_data['poly_xy']

    R_temp = PolynomialRing(QQ, names=['x', 'y'])
    x_temp, y_temp = R_temp.gens()
    F_xy = poly_xy_func(x_temp, y_temp)

    K = FunctionField(QQ, names=(parameter_var,))
    t = K.gen()
    R_K = PolynomialRing(K, names=('Y_poly',))
    Y_poly = R_K.gen()
    F_tY = R_K(0)
    for (i, j), coeff in F_xy.dict().items():
        F_tY += coeff * (t**i) * (Y_poly**j)

    try:
        L = K.extension(F_tY, names=(extension_var,))
        alpha_t = L.gen()
    except ValueError as exc:
        raise ValueError("Curve equation is reducible over QQ(t).") from exc

    R_disp = PolynomialRing(ZZ, names=(parameter_var, extension_var))
    t_d, a_d = R_disp.gens()

    coeffs_F = F_xy.coefficients()
    denoms = [c.denominator() for c in coeffs_F] if coeffs_F else [1]
    lcm_F = lcm(denoms)
    F_xy_int = (F_xy * lcm_F).change_ring(ZZ)
    F_disp = F_xy_int(t_d, a_d)

    C = Curve(F_xy)
    K_curve = C.function_field()
    if K_curve.ngens() >= 2:
        x_K, y_K = K_curve.gens()[:2]
    else:
        y_K = K_curve.gen()
        x_K = K_curve(K_curve.base_ring().gen())

    return {
        'poly_xy_func': poly_xy_func,
        'F_xy': F_xy,
        'x_temp': x_temp,
        'y_temp': y_temp,
        'parameter_var': parameter_var,
        'extension_var': extension_var,
        'K_base': K,
        't': t,
        'L': L,
        'alpha_t': alpha_t,
        'R_disp': R_disp,
        't_d': t_d,
        'a_d': a_d,
        'F_disp': F_disp,
        'curve': C,
        'curve_function_field': K_curve,
        'x_K': x_K,
        'y_K': y_K,
    }


def _primitive_integer_polynomial(poly):
    """Normalize a rational polynomial to a primitive integer polynomial."""
    coeffs = [c for c in poly.coefficients() if c != 0]
    if not coeffs:
        return poly
    common_den = lcm([c.denominator() for c in coeffs])
    poly_int = (poly * common_den).change_ring(ZZ)
    nz_coeffs = [ZZ(c) for c in poly_int.coefficients() if c != 0]
    if nz_coeffs:
        content = gcd(nz_coeffs)
        if content not in [0, 1]:
            poly_int //= content
    if poly_int.leading_coefficient() < 0:
        poly_int = -poly_int
    return poly_int


def _poly_to_symbolic_expr(poly, var_map):
    """Translate a polynomial into a symbolic expression using the supplied variable map."""
    expr = SR(0)
    gens = poly.parent().gens()
    for exponents, coeff in poly.dict().items():
        term = SR(coeff)
        for gen, exponent in zip(gens, exponents):
            if exponent:
                term *= var_map[gen] ** exponent
        expr += term
    return expr


def _format_weierstrass_equation_latex(E, x_name='x_E', y_name='y_E'):
    """Format an elliptic curve model as a LaTeX Weierstrass equation."""
    x_var = SR.var(x_name)
    y_var = SR.var(y_name)
    a1, a2, a3, a4, a6 = E.a_invariants()
    lhs = y_var**2 + a1 * x_var * y_var + a3 * y_var
    rhs = x_var**3 + a2 * x_var**2 + a4 * x_var + a6
    return f"{latex(lhs)} = {latex(rhs)}"


def _hyperelliptic_discriminant_latex(family_ctx, factor_output=True):
    """Compute the displayed discriminant polynomial for a hyperelliptic family."""
    F_xy = family_ctx['F_xy']
    y_temp = family_ctx['y_temp']
    parameter_var = family_ctx['parameter_var']

    if F_xy.degree(y_temp) != 2:
        raise ValueError(
            f"Hyperelliptic sanity check failed: degree in {family_ctx['extension_var']} is {F_xy.degree(y_temp)}, expected 2."
        )

    disc_xy = F_xy.discriminant(y_temp)
    R_t = PolynomialRing(ZZ, names=(parameter_var,))
    t_d = R_t.gen()
    disc_t = PolynomialRing(QQ, names=(parameter_var,))(0)
    t_q = disc_t.parent().gen()
    for exponents, coeff in disc_xy.dict().items():
        exp_tuple = tuple(exponents) if hasattr(exponents, '__iter__') else (exponents,)
        if len(exp_tuple) >= 2 and exp_tuple[1] != 0:
            raise ValueError("Hyperelliptic discriminant still depends on the extension variable.")
        exp_t = exp_tuple[0]
        disc_t += coeff * (t_q ** exp_t)
    disc_int = _primitive_integer_polynomial(disc_t)
    disc_disp = R_t(disc_int)
    if factor_output:
        disc_obj = disc_disp.factor()
    else:
        disc_obj = disc_disp
    return str(latex(disc_obj))


def _verify_rat_on_target_qexp(runtime, rat, target_q, verification_order=None):
    """Verify a rational function against an explicit target q-series."""
    if rat is None:
        return False
    order = runtime['verify_order_full_prec'] if verification_order is None else int(verification_order)
    try:
        full_ctx = runtime['full_ctx']
        diff = rat(x=full_ctx['x0'], y=full_ctx['y0']) - target_q
        return diff.valuation() >= order
    except Exception:
        return False


def recover_rational_function_from_qexp(target_q, runtime, label,
                                        start_deg=1, max_deg=16, step=1,
                                        max_deg_y_plan=None, verification_order=None,
                                        progress_cb=None):
    """Headless helper that reconstructs one rational function from a target q-expansion."""
    P_xy = runtime['P_xy']
    start_deg = _adjust_start_degree(runtime['N'], start_deg)
    if verification_order is None:
        verification_order = runtime['verify_order_full_prec']

    ctx = runtime['full_ctx']
    next_min_by_y = {}
    for deg in range(start_deg, max_deg + 1, step):
        y_candidates = _max_deg_y_candidates(
            runtime, deg,
            explicit_plan=max_deg_y_plan
        )
        for max_deg_y in y_candidates:
            min_total_degree = next_min_by_y.get(max_deg_y, deg)
            y_label = str(max_deg_y)
            if progress_cb is not None:
                progress_cb(label, deg, max_deg, y_label, runtime['prec'])

            local_cache = get_solver_cache_for_context(ctx, max_deg_y)
            sols = find_minimal_robust_solution(
                target_q, ctx['x0'], ctx['y0'], P_xy,
                min_total_degree=min_total_degree,
                max_total_degree=deg,
                max_deg_y=max_deg_y,
                L0_margin=runtime['solver_L0_margin'],
                Delta=runtime['solver_Delta'],
                verification_order=verification_order,
                stop_at_first_total_degree=True,
                use_cache=True,
                precomputed=local_cache,
                include_q_series=False,
            )
            if sols:
                rat = AB_ratio(sols[0])
                if rat is not None and _verify_rat_on_target_qexp(runtime, rat, target_q, verification_order):
                    return rat, sols[0]
        for max_deg_y in y_candidates:
            next_min_by_y[max_deg_y] = deg + 1

    return None, None


def get_quotient_power_series(label, prec):
    """Return the q-expansions of the quotient elliptic curve coordinates for a Cremona label."""
    E = EllipticCurve(str(label).lower())
    try:
        E = E.optimal_curve()
    except Exception:
        pass
    phi = E.modular_parametrization()
    x_q, y_q = phi.power_series(prec=int(prec))
    R_q = LaurentSeriesRing(QQ, 'q', default_prec=int(prec))
    return E, R_q(x_q), R_q(y_q)


def recover_bielliptic_quotient_map(result, family_ctx=None, progress_cb=None):
    """Recover the elliptic quotient map x_E(X,Y) for a bielliptic level."""
    runtime = result.get('runtime')
    if runtime is None:
        raise ValueError("Bielliptic quotient recovery requires runtime context.")

    model_data = result['model_data']
    quotient_label = model_data.get('quotient_label')
    if not quotient_label:
        raise ValueError("Bielliptic metadata is missing `quotient_label`.")

    E, xE_q, yE_q = get_quotient_power_series(quotient_label, runtime['prec'])
    x_cap = model_data.get('bielliptic_x_degree_cap', 12)
    x_max_deg_y_plan = model_data.get('bielliptic_x_max_deg_y_plan')

    x_rat, x_sol = recover_rational_function_from_qexp(
        xE_q, runtime, 'x_E',
        start_deg=1, max_deg=x_cap,
        max_deg_y_plan=x_max_deg_y_plan,
        verification_order=runtime['verify_order_full_prec'],
        progress_cb=progress_cb,
    )
    if x_rat is None:
        raise ValueError(f"Failed to recover x_E(X,Y) for quotient {quotient_label}.")

    return {
        'label': quotient_label,
        'E': E,
        'x_q': xE_q,
        'y_q': yE_q,
        'x_rat': x_rat,
        'x_solution': x_sol,
    }


def _poly_expr_with_names(poly, var_names):
    """Convert a polynomial to a symbolic expression with explicit display variable names."""
    return _poly_to_symbolic_expr(
        poly,
        {gen: SR.var(name) for gen, name in zip(poly.parent().gens(), var_names)}
    )


def _find_zero_relation_over_quotient(base_x_q, base_y_q, term_series, P_uv,
                                      max_total_degree=6, start_degree=1, max_deg_y=1,
                                      L0_margin=5, Delta=30, verification_order=100,
                                      required_blocks=None, progress_cb=None, label=None):
    """Find a polynomial relation over the quotient coordinates by q-expansion linear algebra."""
    if required_blocks is None:
        required_blocks = []

    cache = _build_solver_precomputed(
        base_x_q, base_y_q, max_total_degree, max_deg_y, precomputed={}
    )
    basis_by_degree = cache['basis_by_degree']
    exponents_by_degree = cache['exponents_by_degree']

    for total_degree in range(max(0, start_degree), max_total_degree + 1):
        basis = basis_by_degree[total_degree]
        exponents = exponents_by_degree[total_degree]
        if not basis:
            continue

        series_list = []
        block_slices = {}
        for block_name, term_q in term_series:
            start = len(series_list)
            series_list.extend([basis_q * term_q for basis_q in basis])
            block_slices[block_name] = (start, len(series_list))

        if progress_cb is not None and label is not None:
            progress_cb(label, total_degree, max_total_degree, str(max_deg_y), 0)

        L0 = len(series_list) + L0_margin
        split_scan = _scan_series_list(series_list)
        split_ok, _ = _quick_split_feasible(split_scan, L0, verification_order)
        if not split_ok:
            continue

        M_L0, k_min, rows_eff = assemble_matrix_generic(
            series_list, L0, verbose=False, pre_scan=split_scan
        )
        if M_L0 is None or rows_eff <= 0:
            continue

        K_L0 = M_L0.right_kernel()
        if K_L0.dimension() == 0:
            continue

        min_required_valuation = max(k_min + rows_eff, verification_order)
        M_ext = None
        ext_has_new_rows = None
        L_ext_target = rows_eff + max(0, Delta)

        for candidate_vec in K_L0.basis():
            block_ok = True
            for block_name in required_blocks:
                i0, i1 = block_slices[block_name]
                if not any(candidate_vec[i] != 0 for i in range(i0, i1)):
                    block_ok = False
                    break
            if not block_ok:
                continue

            is_valid = True
            if Delta > 0:
                if ext_has_new_rows is None:
                    M_ext, _, rows_ext = assemble_matrix_generic(
                        series_list, L_ext_target, verbose=False, pre_scan=split_scan
                    )
                    ext_has_new_rows = (M_ext is not None and rows_ext > rows_eff)
                if ext_has_new_rows and not (M_ext * candidate_vec).is_zero():
                    is_valid = False

            if is_valid:
                S = sum(candidate_vec[i] * series_list[i] for i in range(len(series_list)))
                if S.valuation() < min_required_valuation:
                    is_valid = False

            if not is_valid:
                continue

            final_coeffs = normalize_vector_to_ints(candidate_vec)
            block_polys = {}
            for block_name, (i0, i1) in block_slices.items():
                block_polys[block_name] = reconstruct_polynomial(
                    final_coeffs[i0:i1], total_degree, max_deg_y, P_uv, exponents
                )

            return {
                'degree': total_degree,
                'max_deg_y': max_deg_y,
                'polys': block_polys,
            }

    return None


def _recover_quadratic_coordinate_family(result, quotient_map, progress_cb=None):
    """Recover the quadratic and linear lift relations above a bielliptic quotient."""
    runtime = result['runtime']
    full_ctx = runtime['full_ctx']
    model_data = result['model_data']

    P_uv = PolynomialRing(QQ, names=('x_E', 'y_E'))
    xE_q = quotient_map['x_q']
    yE_q = quotient_map['y_q']
    quad_cap = int(model_data.get('bielliptic_relation_degree_cap', 8))
    lin_cap = int(model_data.get('bielliptic_linear_degree_cap', quad_cap))

    candidates = [
        ('x', full_ctx['x0'], full_ctx['y0']),
        ('y', full_ctx['y0'], full_ctx['x0']),
    ]

    for primitive_var, primitive_q, other_q in candidates:
        quad_rel = _find_zero_relation_over_quotient(
            xE_q, yE_q,
            [('quad', primitive_q**2), ('lin', primitive_q), ('const', primitive_q.parent()(1))],
            P_uv,
            max_total_degree=quad_cap,
            start_degree=1,
            max_deg_y=1,
            L0_margin=runtime['solver_L0_margin'],
            Delta=runtime['solver_Delta'],
            verification_order=runtime['verify_order_full_prec'],
            required_blocks=['quad'],
            progress_cb=progress_cb,
            label=f'{primitive_var}_quad',
        )
        if quad_rel is None or quad_rel['polys']['quad'] == 0:
            continue

        other_var = 'y' if primitive_var == 'x' else 'x'
        lin_rel = _find_zero_relation_over_quotient(
            xE_q, yE_q,
            [('target', other_q), ('aux', primitive_q), ('const', primitive_q.parent()(1))],
            P_uv,
            max_total_degree=lin_cap,
            start_degree=0,
            max_deg_y=1,
            L0_margin=runtime['solver_L0_margin'],
            Delta=runtime['solver_Delta'],
            verification_order=runtime['verify_order_full_prec'],
            required_blocks=['target'],
            progress_cb=progress_cb,
            label=f'{other_var}_lin',
        )
        if lin_rel is None or lin_rel['polys']['target'] == 0:
            continue

        return {
            'primitive_var': primitive_var,
            'other_var': other_var,
            'quadratic_relation': {
                'quad_poly': quad_rel['polys']['quad'],
                'lin_poly': quad_rel['polys']['lin'],
                'const_poly': quad_rel['polys']['const'],
                'degree': quad_rel['degree'],
            },
            'linear_relation': {
                'target_poly': lin_rel['polys']['target'],
                'aux_poly': lin_rel['polys']['aux'],
                'const_poly': lin_rel['polys']['const'],
                'degree': lin_rel['degree'],
            },
        }

    raise ValueError("Failed to recover a quadratic family over the elliptic quotient.")


def _format_bielliptic_family_latex(family_data, x_name='x_E', y_name='y_E'):
    """Format the recovered bielliptic lift relations as LaTeX."""
    quad = family_data['quadratic_relation']
    lin = family_data['linear_relation']
    primitive_var = SR.var(family_data['primitive_var'])
    other_var = SR.var(family_data['other_var'])

    quad_expr = (
        _poly_expr_with_names(quad['quad_poly'], (x_name, y_name)) * primitive_var**2
        + _poly_expr_with_names(quad['lin_poly'], (x_name, y_name)) * primitive_var
        + _poly_expr_with_names(quad['const_poly'], (x_name, y_name))
    )
    lin_expr = (
        _poly_expr_with_names(lin['target_poly'], (x_name, y_name)) * other_var
        + _poly_expr_with_names(lin['aux_poly'], (x_name, y_name)) * primitive_var
        + _poly_expr_with_names(lin['const_poly'], (x_name, y_name))
    )
    return latex(quad_expr), latex(lin_expr)


def _roots_of_quadratic_polynomial(poly_T):
    """Enumerate the roots of a linear or quadratic polynomial together with their fields."""
    if poly_T == 0 or poly_T.degree() <= 0:
        return []
    if poly_T.degree() == 1:
        return [(poly_T.roots(QQ, multiplicities=False)[0], QQ)]
    fac = poly_T.factor()
    if all(f.degree() == 1 for f, _ in fac):
        return [(root, QQ) for root in poly_T.roots(QQ, multiplicities=False)]
    K = NumberField(poly_T, names=('alpha',))
    roots = poly_T.change_ring(K).roots(K, multiplicities=False)
    return [(root, K) for root in roots]


def lift_bielliptic_point_to_quadratic_points(bielliptic_data, P):
    """Lift one rational point on the quotient elliptic curve to points on X_0(N)."""
    xP = QQ(P[0])
    yP = QQ(P[1])

    quad = bielliptic_data['family']['quadratic_relation']
    lin = bielliptic_data['family']['linear_relation']
    RT = PolynomialRing(QQ, names=('T',))
    T = RT.gen()
    poly_T = RT(quad['quad_poly'](xP, yP)) * T**2 + RT(quad['lin_poly'](xP, yP)) * T + RT(quad['const_poly'](xP, yP))
    roots = _roots_of_quadratic_polynomial(poly_T)

    target_coeff = QQ(lin['target_poly'](xP, yP))
    aux_coeff = QQ(lin['aux_poly'](xP, yP))
    const_coeff = QQ(lin['const_poly'](xP, yP))
    if target_coeff == 0:
        return []

    points = []
    for primitive_value, field in roots:
        if field is QQ:
            primitive_value = QQ(primitive_value)
            other_value = -(aux_coeff * primitive_value + const_coeff) / target_coeff
            field_degree = 1
        else:
            K = field
            primitive_value = K(primitive_value)
            other_value = -(K(aux_coeff) * primitive_value + K(const_coeff)) / K(target_coeff)
            field_degree = K.degree()

        if bielliptic_data['family']['primitive_var'] == 'x':
            x_coord = primitive_value
            y_coord = other_value
        else:
            x_coord = other_value
            y_coord = primitive_value

        points.append({
            'x': x_coord,
            'y': y_coord,
            'field': field,
            'field_degree': field_degree,
        })
    return points


def _sample_bielliptic_quadratic_points(bielliptic_data, sample_count=3):
    """Collect a few sample quadratic fibers above low multiples of the base generator."""
    E = bielliptic_data['E']
    gens = list(E.gens())
    samples = []
    if not gens:
        return samples

    base_gen = gens[0]
    n = 1
    while len(samples) < sample_count and n <= max(8, 3 * sample_count):
        P = n * base_gen
        if P.is_zero():
            n += 1
            continue
        lifted = lift_bielliptic_point_to_quadratic_points(bielliptic_data, P)
        quadratic_only = [pt for pt in lifted if pt['field_degree'] == 2]
        if quadratic_only:
            samples.append({
                'multiple': n,
                'quotient_point': P,
                'fiber_points': quadratic_only,
            })
        n += 1
    return samples


def build_family_report(result, factor_output=True, progress_cb=None):
    """Convert recovered maps into the structured public report payload."""
    N = result.get('N')
    runtime = result.get('runtime')
    if N is None and runtime is not None:
        N = runtime['N']
    if N is None:
        raise ValueError("Missing level N in result payload.")

    model_data = result['model_data']
    family_type = model_data.get('quadratic_family_type', 'general')
    family_ctx = _build_family_context(model_data)
    arg_sig = _family_argument_signature(model_data)

    report = {
        'N': N,
        'family_type': family_type,
        'base_curve_note': (
            f"\\textbf{{Base Curve Equation:}} "
            f"\\text{{Parameters }} {model_data.get('family_parameter_var', 't')}, "
            f"{model_data.get('family_extension_var', 'alpha_t')} \\text{{ satisfy:}}"
        ),
        'base_eq': f"{latex(family_ctx['F_disp'])} = 0",
        'domain_coeffs': [
            (f"a_4({arg_sig})", _ratfunc_to_latex(result['maps']['u'], family_ctx['L'], family_ctx['t'], family_ctx['alpha_t'], family_ctx['R_disp'], family_ctx['t_d'], family_ctx['a_d'], factor_output=factor_output)),
            (f"a_6({arg_sig})", _ratfunc_to_latex(result['maps']['v'], family_ctx['L'], family_ctx['t'], family_ctx['alpha_t'], family_ctx['R_disp'], family_ctx['t_d'], family_ctx['a_d'], factor_output=factor_output)),
        ],
        'codomain_coeffs': [
            (f"a'_4({arg_sig})", _ratfunc_to_latex(result['maps']['uN'], family_ctx['L'], family_ctx['t'], family_ctx['alpha_t'], family_ctx['R_disp'], family_ctx['t_d'], family_ctx['a_d'], factor_output=factor_output)),
            (f"a'_6({arg_sig})", _ratfunc_to_latex(result['maps']['vN'], family_ctx['L'], family_ctx['t'], family_ctx['alpha_t'], family_ctx['R_disp'], family_ctx['t_d'], family_ctx['a_d'], factor_output=factor_output)),
        ],
        'family_ctx': family_ctx,
    }

    if family_type == 'hyperelliptic':
        extension_var = model_data.get('family_extension_var', 'alpha_t')
        parameter_var = model_data.get('family_parameter_var', 't')
        report['hyperelliptic'] = {
            'discriminant_label': f"\\Delta({parameter_var})",
            'discriminant': _hyperelliptic_discriminant_latex(family_ctx, factor_output=factor_output),
            'note': (
                f"\\text{{For }} {parameter_var} \\in \\mathbf{{Q}}, "
                f"\\text{{if }} \\Delta({parameter_var}) \\text{{ is not a square, then }} "
                f"{family_ctx['extension_var']} \\text{{ lies in a quadratic field.}}"
            ),
        }

    if family_type == 'bielliptic':
        quotient_map = recover_bielliptic_quotient_map(
            result, family_ctx=family_ctx, progress_cb=progress_cb
        )
        family_data = _recover_quadratic_coordinate_family(
            result, quotient_map, progress_cb=progress_cb
        )
        quad_latex, lin_latex = _format_bielliptic_family_latex(family_data)
        seq_quad_latex, seq_lin_latex = _format_bielliptic_family_latex(
            family_data, x_name='x_n', y_name='y_n'
        )
        rank = quotient_map['E'].rank()
        generators = list(quotient_map['E'].gens())
        distinguished_generator = generators[0] if generators else None
        bielliptic_data = {
            'label': quotient_map['label'],
            'E': quotient_map['E'],
            'equation': _format_weierstrass_equation_latex(quotient_map['E']),
            'rank': rank,
            'generators': generators,
            'distinguished_generator': distinguished_generator,
            'x_map': _ratfunc_to_latex(quotient_map['x_rat'], family_ctx['L'], family_ctx['t'], family_ctx['alpha_t'], family_ctx['R_disp'], family_ctx['t_d'], family_ctx['a_d'], factor_output=factor_output),
            'family': family_data,
            'quadratic_relation': quad_latex,
            'linear_relation': lin_latex,
            'sequence_quadratic_relation': seq_quad_latex,
            'sequence_linear_relation': seq_lin_latex,
        }
        bielliptic_data['sample_points'] = _sample_bielliptic_quadratic_points(
            bielliptic_data,
            sample_count=int(model_data.get('bielliptic_sample_count', 3))
        )
        report['bielliptic'] = {
            'label': bielliptic_data['label'],
            'equation': bielliptic_data['equation'],
            'rank': bielliptic_data['rank'],
            'generators': bielliptic_data['generators'],
            'distinguished_generator': bielliptic_data['distinguished_generator'],
            'x_map': bielliptic_data['x_map'],
            'primitive_var': bielliptic_data['family']['primitive_var'],
            'other_var': bielliptic_data['family']['other_var'],
            'quadratic_relation': bielliptic_data['quadratic_relation'],
            'linear_relation': bielliptic_data['linear_relation'],
            'sequence_quadratic_relation': bielliptic_data['sequence_quadratic_relation'],
            'sequence_linear_relation': bielliptic_data['sequence_linear_relation'],
            'sample_points': bielliptic_data['sample_points'],
        }

    return report

# ==============================================================================
# 4. Public Batch / Report Helpers
# ==============================================================================

def solve_isogeny_result(N, prec=500, progress_cb=None, strict_degree_chain=None):
    """Run the core map recovery workflow for a single level."""
    return _solve_maps_for_level(
        N,
        prec=prec,
        progress_cb=progress_cb,
        strict_degree_chain=strict_degree_chain,
    )


def solve_isogeny_report(N, prec=500, strict_degree_chain=None,
                        factor_output=True, progress_cb=None):
    """Recover a single level and build the corresponding report payload."""
    result = solve_isogeny_result(
        N,
        prec=prec,
        progress_cb=progress_cb,
        strict_degree_chain=strict_degree_chain,
    )
    if result is None:
        return None
    return build_family_report(result, factor_output=factor_output, progress_cb=progress_cb)


def solve_single_case_report(N, strict_degree_chain=None, prec=500, factor_output=True):
    """Headless helper used for batch generation and quick command-line checks."""
    def headless_progress(label, deg, max_deg, y_label, work_prec):
        print(f"  [{label}] deg <= {deg}, max_deg_y={y_label}, prec={work_prec}...", end="\r")

    print(f"Computing N={N}...")
    try:
        result = solve_isogeny_result(
            N,
            prec=prec,
            progress_cb=headless_progress,
            strict_degree_chain=strict_degree_chain,
        )
    except NotImplementedError:
        return None

    if result is None:
        print("  [maps] Failed.                                 ")
        return None

    deg_u = result["degrees"]["u"]
    deg_v = result["degrees"]["v"]
    print(f"  [u] Found at degree {deg_u}.                        ")
    print(f"  [v] Found at degree {deg_v}.                        ")

    try:
        return build_family_report(result, factor_output=factor_output, progress_cb=headless_progress)
    except ValueError:
        return None


def _write_report_block(f_out, report):
    """Append one report block to the canonical text output file."""
    f_out.write(f"=== N = {report['N']} ===\n")
    f_out.write(f"Family Type: {report['family_type']}\n")
    f_out.write(f"Base Curve: {report['base_eq']}\n")
    for lhs, rhs in report["domain_coeffs"]:
        f_out.write(f"{lhs}: {rhs}\n")
    for lhs, rhs in report["codomain_coeffs"]:
        f_out.write(f"{lhs}: {rhs}\n")

    if report["family_type"] == "hyperelliptic":
        hyper = report["hyperelliptic"]
        f_out.write(f"{hyper['discriminant_label']}: {hyper['discriminant']}\n")
        f_out.write(f"Hyperelliptic Note: {hyper['note']}\n")

    if report["family_type"] == "bielliptic":
        bio = report["bielliptic"]
        f_out.write(f"Quotient Curve Label: {bio['label']}\n")
        f_out.write(f"Quotient Curve Equation: {bio['equation']}\n")
        f_out.write(f"Quotient Curve Rank: {bio['rank']}\n")
        if bio["generators"]:
            f_out.write("Quotient Generators: " + ", ".join([str(P) for P in bio["generators"]]) + "\n")
        if bio["distinguished_generator"] is not None:
            f_out.write(f"Distinguished Generator P1: {bio['distinguished_generator']}\n")
            f_out.write("Quotient Sequence: P_n = n*P1 = (x_n,y_n)\n")
        f_out.write(f"Quotient Map x_E(t,alpha_t): {bio['x_map']}\n")
        f_out.write(f"Lift Primitive Variable on X_0(N): {bio['primitive_var']}\n")
        f_out.write(f"Lift Other Variable on X_0(N): {bio['other_var']}\n")
        f_out.write(f"Lift Quadratic Relation over Q(x_n,y_n): {bio['sequence_quadratic_relation']}\n")
        f_out.write(f"Lift Linear Relation over Q(x_n,y_n): {bio['sequence_linear_relation']}\n")
        for sample in bio["sample_points"]:
            f_out.write(f"Sample Multiple n={sample['multiple']}: P={sample['quotient_point']}\n")
            for idx, pt in enumerate(sample["fiber_points"], start=1):
                field_desc = "QQ"
                if hasattr(pt["field"], "defining_polynomial"):
                    field_desc = str(pt["field"].defining_polynomial())
                f_out.write(
                    f"  Fiber {idx}: deg={pt['field_degree']}, "
                    f"x={pt['x']}, y={pt['y']}, field={field_desc}\n"
                )

    f_out.write("\n" + "-" * 40 + "\n\n")


def run_batch_and_save(targets, filename="isogeny_results.txt", prec=500,
                       strict_degree_chain=None, factor_output=True):
    """Compute a list of levels and write a single canonical text results file."""
    targets = list(targets)
    print(f"Starting batch process for {len(targets)} levels...")
    failed_list = []

    with open(filename, "w") as f_out:
        f_out.write("=== Cyclic Isogeny Construction Results ===\n")
        f_out.write("Format: LaTeX\n\n")

        for n_idx, n_val in enumerate(targets):
            try:
                report = solve_single_case_report(
                    n_val,
                    strict_degree_chain=strict_degree_chain,
                    prec=prec,
                    factor_output=factor_output,
                )
                if report:
                    _write_report_block(f_out, report)
                    f_out.flush()
                    print(f"[{n_idx + 1}/{len(targets)}] Saved N={n_val}.          ")
                else:
                    print(f"[{n_idx + 1}/{len(targets)}] Skipped N={n_val} (Not implemented/Failed).     ")
                    failed_list.append(n_val)
            except Exception as e:
                print(f"[{n_idx + 1}/{len(targets)}] Error N={n_val}: {e}     ")
                failed_list.append(n_val)

    print(f"\nBatch processing complete. Results saved to {filename}")
    if failed_list:
        print(f"Failed/Skipped N: {failed_list}")

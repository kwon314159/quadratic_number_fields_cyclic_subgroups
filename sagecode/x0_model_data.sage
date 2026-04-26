# x0_model_data.sage
# Data repository for explicit X0(N) generators and plane models.
#
# This file is loaded by solver_core.sage and keeps the lightweight
# metadata and helper layer. The heavy q-expansion table is lazy-loaded from
# chunked generator files on demand.

import threading

_YANG_GENERATOR_CACHE = {}


def _generator_prec_from_q(q, fallback=500):
    prec = q.parent().default_prec()
    if prec == infinity:
        return int(fallback)
    return int(prec)


def _coerce_series_pair_to_parent(x0, y0, q, prec):
    parent = q.parent()
    return parent(x0).add_bigoh(prec), parent(y0).add_bigoh(prec)


def _generator_work_prec(prec, margin=32):
    return int(prec) + int(margin)


def _cached_eta_generators(N, eta_x, eta_y, prec):
    key = ("eta", int(N), int(prec))
    cached = _YANG_GENERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    work_prec = _generator_work_prec(prec)
    R_q = LaurentSeriesRing(QQ, "q", default_prec=work_prec)
    x0 = R_q(EtaProduct(N, eta_x).q_expansion(work_prec)).add_bigoh(prec)
    y0 = R_q(EtaProduct(N, eta_y).q_expansion(work_prec)).add_bigoh(prec)
    _YANG_GENERATOR_CACHE[key] = (x0, y0)
    return x0, y0


def _normalize_eta_def(eta_def):
    return tuple(sorted((int(d), int(e)) for d, e in eta_def.items()))


def _normalize_comp_list(comp_list):
    return tuple((int(g), int(e)) for g, e in comp_list)


def _cached_eta_series(N, eta_def, prec):
    eta_key = _normalize_eta_def(eta_def)
    key = ("eta_series", int(N), eta_key, int(prec))
    cached = _YANG_GENERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    work_prec = _generator_work_prec(prec)
    R_q = LaurentSeriesRing(QQ, "q", default_prec=work_prec)
    series = R_q(EtaProduct(N, dict(eta_key)).q_expansion(work_prec)).add_bigoh(prec)
    _YANG_GENERATOR_CACHE[key] = series
    return series


def _cached_yang_coset_reps(N):
    key = ("yang_cosets", int(N))
    cached = _YANG_GENERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    reps = []
    for a in range(1, int(N) // 2 + 1):
        if gcd(a, N) == 1:
            _, s, t = xgcd(a, N)
            reps.append((Integer(a), Integer(-t), Integer(N), Integer(s)))

    reps = tuple(reps)
    _YANG_GENERATOR_CACHE[key] = reps
    return reps


def _cached_yang_trace_series(N, comp_list, prec):
    comp_key = _normalize_comp_list(comp_list)
    key = ("yang_trace", int(N), comp_key, int(prec))
    cached = _YANG_GENERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    work_prec = _generator_work_prec(prec)
    field_order = lcm(12, 2 * int(N))
    K = CyclotomicField(field_order)
    R_cyc = LaurentSeriesRing(K, "q", default_prec=work_prec)
    q_cyc = R_cyc.gen()
    memo_Eg_unit = {}

    def get_Eg_val_and_unit(g):
        g = int(g) % int(N)
        cached_unit = memo_Eg_unit.get(g)
        if cached_unit is not None:
            return cached_unit

        val = QQ(g) / QQ(N)
        exp_val = QQ(N) * (val^2 - val + QQ(1) / 6) / 2

        term = R_cyc(1)
        m_limit = int(work_prec / N) + 2
        for m in range(1, m_limit + 1):
            e1 = (m - 1) * N + g
            e2 = m * N - g
            if e1 < work_prec:
                term *= (1 - q_cyc^e1)
            if e2 < work_prec:
                term *= (1 - q_cyc^e2)

        memo_Eg_unit[g] = (exp_val, term)
        return memo_Eg_unit[g]

    total_sum = R_cyc(0)
    for a, b, c, d in _cached_yang_coset_reps(N):
        term_coeff = K(1)
        term_exp = QQ(0)
        term_series = R_cyc(1)

        for old_g, e in comp_key:
            new_g = (a * old_g) % N
            sign_s = (-1)^((a * old_g) // N)

            if c % 2 != 0:
                val_e1 = b * d * (1 - c^2) + c * (a + d - 3)
                factor_e1 = K.zeta(12)^val_e1
            else:
                val_e1 = a * c * (1 - d^2) + d * (b - c + 3)
                factor_e1 = K.zeta(4)^(-1) * K.zeta(12)^val_e1

            num = old_g^2 * a * b - old_g * b * N
            factor_e2 = K.zeta(2 * N)^num

            term_coeff *= (sign_s * factor_e1 * factor_e2)^e
            eg_val, eg_unit = get_Eg_val_and_unit(new_g)
            term_exp += e * eg_val
            term_series *= eg_unit^e

        total_sum += term_coeff * q_cyc^Integer(term_exp) * term_series

    R_q = LaurentSeriesRing(QQ, "q", default_prec=work_prec)
    trace = _rationalize_series_to_parent(total_sum, R_q, prec)
    _YANG_GENERATOR_CACHE[key] = trace
    return trace


def _cached_yang_generators(N, prec):
    key = ("yang_pair", int(N), int(prec))
    cached = _YANG_GENERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    if N == 14:
        x0 = _cached_eta_series(14, {2: 1, 7: 7, 1: -1, 14: -7}, prec) + 1
        y0 = _cached_eta_series(14, {2: 8, 7: 4, 1: -4, 14: -8}, prec) - 3 * x0 + 1
    elif N == 15:
        x0 = _cached_eta_series(15, {3: 1, 5: 5, 1: -1, 15: -5}, prec) - 1
        y0 = (
            _cached_eta_series(15, {3: 9, 5: 3, 1: -3, 15: -9}, prec)
            - _cached_eta_series(15, {3: 2, 5: 10, 1: -2, 15: -10}, prec)
            - 3 * x0
            - 2
        )
    elif N == 17:
        x0 = _cached_yang_trace_series(17, [(3, 1), (8, 1), (1, -1), (2, -1)], prec) - 4
        y0 = _cached_yang_trace_series(17, [(6, 2), (8, 1), (2, -2), (3, -1)], prec) + 1
    elif N == 19:
        x0 = _cached_yang_trace_series(19, [(7, 1), (8, 1), (1, -1), (6, -1)], prec) - 3
        y0 = _cached_yang_trace_series(19, [(6, 2), (8, 1), (2, -1), (3, -2)], prec) + x0 - 6
    elif N == 21:
        x0 = _cached_eta_series(21, {3: 3, 7: 1, 1: -1, 21: -3}, prec) - 2
        y0 = (
            _cached_eta_series(21, {3: 6, 7: 2, 1: -2, 21: -6}, prec)
            - _cached_eta_series(21, {3: 1, 7: 7, 1: -1, 21: -7}, prec)
            - 2 * x0
            - 4
        )
    elif N == 27:
        x0 = _cached_eta_series(27, {9: 4, 3: -1, 27: -3}, prec)
        y0 = _cached_eta_series(27, {3: 3, 27: -3}, prec)
    elif N == 37:
        x0 = _cached_eta_series(37, {1: 2, 37: -2}, prec) + 37
        y0 = _cached_yang_trace_series(37, [(6, 1), (8, 1), (14, 1), (3, -1), (4, -1), (7, -1)], prec) / 3 - 5 * x0 + 174
    elif N == 43:
        x0 = _cached_yang_trace_series(
            43,
            [(5, 1), (8, 1), (13, 1), (1, -1), (6, -1), (7, -1)],
            prec,
        ) / (3 * 43) - QQ(15) / 43
        y0 = _cached_yang_trace_series(
            43,
            [(2, 1), (9, 1), (11, 1), (12, 1), (14, 1), (20, 1), (1, -1), (4, -1), (6, -1), (7, -1), (15, -1), (19, -1)],
            prec,
        ) / (3 * 43) - QQ(9) / 43
    else:
        raise NotImplementedError(f"Dynamic Yang generators for N={N} are not configured.")

    x0 = x0.add_bigoh(prec)
    y0 = y0.add_bigoh(prec)
    _YANG_GENERATOR_CACHE[key] = (x0, y0)
    return x0, y0


def _yang_split_eta_unit(N, g, prec):
    g = g % N
    val = QQ(g) / QQ(N)
    exp_val = QQ(N) * (val^2 - val + QQ(1) / 6) / 2

    K = CyclotomicField(2 * N)
    R_q = LaurentSeriesRing(K, "q", default_prec=prec)
    q = R_q.gen()
    term = R_q(1)

    for m in range(1, int(prec / N) + 2):
        e1 = (m - 1) * N + g
        e2 = m * N - g
        if e1 < prec:
            term *= (1 - q^e1)
        if e2 < prec:
            term *= (1 - q^e2)

    return exp_val, term


def _rationalize_series_to_parent(series, parent, prec):
    q = parent.gen()
    out = parent(0)
    if series == 0:
        return out.add_bigoh(prec)

    val = series.valuation()
    for i, coeff in enumerate(series.list()):
        if coeff == 0:
            continue
        try:
            rat_coeff = QQ(coeff)
        except Exception:
            rat_coeff = QQ(CC(coeff).real())
        out += rat_coeff * q^(val + i)

    return out.add_bigoh(prec)


def _cached_yang_49_generators(prec):
    key = ("yang49", int(prec))
    cached = _YANG_GENERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    work_prec = _generator_work_prec(prec)
    R_q = LaurentSeriesRing(QQ, "q", default_prec=work_prec)
    q = R_q.gen()

    E49 = EllipticCurve("49a1")
    phi49 = E49.modular_parametrization()
    x_ser, _ = phi49.power_series(work_prec)
    x0 = R_q(x_ser).add_bigoh(prec)

    exp7, ser7 = _yang_split_eta_unit(49, 7, work_prec)
    exp14, ser14 = _yang_split_eta_unit(49, 14, work_prec)
    exp21, ser21 = _yang_split_eta_unit(49, 21, work_prec)

    q_cyc = ser7.parent().gen()
    t_yang = (
        q_cyc^Integer(exp21 - exp7) * (ser21 / ser7)
        + q_cyc^Integer(exp7 - exp14) * (ser7 / ser14)
        - q_cyc^Integer(exp14 - exp21) * (ser14 / ser21)
    )
    x0_cyc = x0.change_ring(ser7.parent().base_ring())
    y0 = _rationalize_series_to_parent(t_yang - 2 * x0_cyc + 1, R_q, prec)

    _YANG_GENERATOR_CACHE[key] = (x0, y0)
    return x0, y0


_GENERATOR_CHUNK_LEVELS = {
    "startup": (11, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26, 27),
    "mid": (28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 46, 47, 48, 49, 50),
    "high": (53, 59, 61, 65, 67, 71, 79, 83, 89, 101, 131, 163),
}

_GENERATOR_CHUNK_FILES = {
    "startup": "sagecode/x0_generator_startup.sage",
    "mid": "sagecode/x0_generator_mid.sage",
    "high": "sagecode/x0_generator_high.sage",
}

_GENERATOR_CHUNK_FUNCTIONS = {
    "startup": "_get_X0_generators_chunk_startup",
    "mid": "_get_X0_generators_chunk_mid",
    "high": "_get_X0_generators_chunk_high",
}

_GENERATOR_CHUNK_FOR_LEVEL = {}
for _chunk_name, _chunk_levels in _GENERATOR_CHUNK_LEVELS.items():
    for _level in _chunk_levels:
        _GENERATOR_CHUNK_FOR_LEVEL[int(_level)] = _chunk_name

_GENERATOR_CHUNK_LOADED = {name: False for name in _GENERATOR_CHUNK_FILES}
_GENERATOR_CHUNK_LOAD_LOCK = threading.Lock()
_GENERATOR_PREFETCH_LOCK = threading.Lock()
_GENERATOR_PREFETCH_ACTIVE = False


def _load_generator_chunk(chunk_name):
    if _GENERATOR_CHUNK_LOADED[chunk_name]:
        return

    with _GENERATOR_CHUNK_LOAD_LOCK:
        if _GENERATOR_CHUNK_LOADED[chunk_name]:
            return

        load(_GENERATOR_CHUNK_FILES[chunk_name])
        _GENERATOR_CHUNK_LOADED[chunk_name] = True


def preload_generator_chunks_async(chunk_names=None):
    """
    Warm up generator chunks in a background thread.
    The default order prioritizes small-N cases so the notebook becomes
    responsive quickly while the larger chunks continue loading.
    """
    global _GENERATOR_PREFETCH_ACTIVE

    names = tuple(chunk_names) if chunk_names is not None else ("startup", "mid", "high")

    with _GENERATOR_PREFETCH_LOCK:
        if _GENERATOR_PREFETCH_ACTIVE:
            return False
        _GENERATOR_PREFETCH_ACTIVE = True

    def _worker():
        global _GENERATOR_PREFETCH_ACTIVE
        try:
            for chunk_name in names:
                _load_generator_chunk(chunk_name)
        finally:
            _GENERATOR_PREFETCH_ACTIVE = False

    thread = threading.Thread(
        target=_worker,
        name="x0-generator-prefetch",
        daemon=True,
    )
    thread.start()
    return True


def get_X0_generators(N, q):
    """
    Return the generators (x0, y0) for the function field of X0(N) as
    q-expansions. Chunked generator tables are loaded on demand.
    """
    chunk_name = _GENERATOR_CHUNK_FOR_LEVEL.get(int(N))
    if chunk_name is None:
        raise NotImplementedError(f"Generators for N={N} are not yet implemented in this library.")

    _load_generator_chunk(chunk_name)
    fn_name = _GENERATOR_CHUNK_FUNCTIONS[chunk_name]
    return globals()[fn_name](N, q)


def get_model_data(N):
    """
    Return the curve data for X0(N), including the curve type, equation,
    generators, and rational test points.
    """
    models = {
        # [N=11] Genus 1 (Elliptic)
        11: {
            'type': 'elliptic',
            'curve_coeffs': [0, -1, 1, -10, -20],
            'poly_xy': lambda x, y: -20 - y - 10*x - y^2 - x^2 + x^3,
            'generators': lambda q: get_X0_generators(11, q),
        },

        # [N=14] Genus 1 (Elliptic)
        14: {
            'type': 'elliptic',
            'curve_coeffs': [1, 0, 1, 4, -6],
            'poly_xy': lambda x, y: -6 - y + 4*x - y^2 - x*y + x^3,
            'generators': lambda q: get_X0_generators(14, q),
        },

        # [N=15] Genus 1 (Elliptic)
        15: {
            'type': 'elliptic',
            'curve_coeffs': [1, 1, 1, -10, -10],
            'poly_xy': lambda x, y: -10 - y - 10*x - y^2 - x*y + x^2 + x^3,
            'generators': lambda q: get_X0_generators(15, q),
        },

        # [N=17] Genus 1 (Elliptic)
        17: {
            'type': 'elliptic',
            'curve_coeffs': [1, -1, 1, -1, -14],
            'poly_xy': lambda x, y: -14 - y - x - y^2 - x*y - x^2 + x^3,
            'generators': lambda q: get_X0_generators(17, q),
        },

        # [N=19] Genus 1 (Elliptic)
        19: {
            'type': 'elliptic',
            'curve_coeffs': [0, 1, -1, -9, -15],
            'poly_xy': lambda x, y: -15 + y - 9*x - y^2 + x^2 + x^3,
            'generators': lambda q: get_X0_generators(19, q),
        },
        # [N=20] Genus 1 (Elliptic)
        20: {
            'type': 'elliptic',
            'curve_coeffs': [0, 1, 0, 4, 4],
            'poly_xy': lambda x, y: 4 - y^2 + x^3 + x^2 + 4*x,
            'generators': lambda q: get_X0_generators(20, q),
        },

        # [N=21] Genus 1 (Elliptic)
        21: {
            'type': 'elliptic',
            'curve_coeffs': [1, 0, 0, -4, -1],
            'poly_xy': lambda x, y: -1 - 4*x - y^2 - x*y + x^3,
            'generators': lambda q: get_X0_generators(21, q),
        },
        22: {
            'type': 'hyperelliptic',
            'quadratic_family_type': 'hyperelliptic',
            'poly_xy': lambda x, y: x^6 + 6*x^5 + 11*x^4 + 24*x^3 + 11*x^2 - y^2 + 18*x - 7,
            'generators': lambda q: get_X0_generators(22, q),
        },

        23: {
            'type': 'hyperelliptic',
            'quadratic_family_type': 'hyperelliptic',
            'poly_xy': lambda x, y: x^6 - 8*x^5 + 2*x^4 + 2*x^3 - 11*x^2 - y^2 + 10*x - 7,
            'generators': lambda q: get_X0_generators(23, q),
        },

        26: {
            'type': 'hyperelliptic',
            'quadratic_family_type': 'hyperelliptic',
            'poly_xy': lambda x, y: x^6 - 8*x^5 + 8*x^4 - 18*x^3 + 8*x^2 - y^2 - 8*x + 1,
            'generators': lambda q: get_X0_generators(26, q),
        },

        # [N=24] Genus 1 (Elliptic)
        24: {
            'type': 'elliptic',
            'curve_coeffs': [0, -1, 0, -4, 4],
            'poly_xy': lambda x, y: 4 - 4*x - x^2 + x^3 - y^2,
            'generators': lambda q: get_X0_generators(24, q),
        },
        # [N=27] Genus 1 (Elliptic)
        27: {
            'type': 'elliptic',
            'curve_coeffs': [0, 0, 9, 0, -27],
            'poly_xy': lambda x, y: -27 - 9*y - y^2 + x^3,
            'generators': lambda q: get_X0_generators(27, q),
        },
        # [N=32] Genus 1 (Elliptic)
        32: {
            'type': 'elliptic',
            'curve_coeffs': [0, 0, 0, 4, 0],
            'poly_xy': lambda x, y: x^3 + 4*x - y^2,
            'generators': lambda q: get_X0_generators(32, q),
        },
        # [N=36] Genus 1 (Elliptic)
        36: {
            'type': 'elliptic',
            'curve_coeffs': [0, 0, 0, 0, 1],
            'poly_xy': lambda x, y: x^3 + 1 - y^2,
            'generators': lambda q: get_X0_generators(36, q),
        },
        # [N=37] Genus 2 (Hyperelliptic)
        # y.yang's model for X0(37)
        # 37: {
        #     'type': 'general',
        #     'poly_xy': lambda x, y: 259*y^2 - 259*x*y + 1332*x^2 - y^3 - 7*x*y^2 + 7*x^2*y - 73*x^3 + x^4,
        #     'generators': lambda q: get_X0_generators(37, q),
        #     'base_points': [[0, 0, 1], [0, 1, 0], [0, 259, 1], [36, 0, 1], [37, 0, 1]] # Rational points on X0(37)(bound=1000)
        # },
        # Galbraith model for X0(37)
        37: {
            'type': 'general',
            'poly_xy': lambda x, y: x^6 + 14*x^5 + 35*x^4 + 48*x^3 + 35*x^2 + 14*x + 1 -y^2, 
            'generators': lambda q: get_X0_generators(37, q),
            'base_points': [[0, 1, 1], [0, -1, 1], [1, 1, 0],[1,-1,0]]
        },

        # [N=43] Genus 3 (General)
        43: {
            'type': 'general',
            'poly_xy': lambda x, y: x**4 + 2*x**2*y**2 - 3*y**4 + 8*x**2*y + 8*y**3 + 16*x**2 + 16*y**2 + 48*y + 64,
            'generators': lambda q: get_X0_generators(43, q),
            'base_points': [[1, 1, 0], [1, -1, 0], [0, 4, -3]]
        },
        # [N=49] Genus 1 (Elliptic)
        49: {
            'type': 'elliptic',
            'curve_coeffs': [1, -1, 0, -2, -1],
            'poly_xy': lambda x, y: x^3 - x^2 - 2*x - 1 - y^2 - x*y,
            'generators': lambda q: get_X0_generators(49, q),
        },
        # [N=53] Genus 4 (Bielliptic)
        53: {
            'type': 'general',
            # Plane model computed in 53.ipynb using the triangular cuspform basis
            # X = x_2 / x_0, Y = x_3 / x_0.
            'poly_xy': lambda x, y: (
                25*x^6 - 40*x^5*y + 90*x^4*y^2 - 55*x^3*y^3 + 34*x^2*y^4
                - 24*x*y^5 + 6*y^6 + 20*x^5 + 13*x^4*y - 18*x^3*y^2
                + 46*x^2*y^3 - 28*x*y^4 + 3*y^5 + 10*x^4 - 58*x^3*y
                + 42*x^2*y^2 - 31*x*y^3 + 10*y^4 - 9*x^3 - 12*x^2*y
                - 3*x*y^2 + 6*y^3 + 9*y^2
            ),
            'generators': lambda q: get_X0_generators(53, q),
        },

    61: {
        'type': 'bielliptic',
        'quadratic_family_type': 'bielliptic',
        'quotient_label': '61a1',
        'poly_xy': lambda x, y: 8*x^6 + 27*x^5*y + 68*x^4*y^2 + 64*x^3*y^3 + 28*x^2*y^4 + 2*x*y^5 + 2*y^6 - 5*x^5 - 2*x^4*y + 18*x^3*y^2 + 32*x^2*y^3 + 40*x*y^4 + 2*y^5 + 10*x^4 + 56*x^3*y + 60*x^2*y^2 + 12*x*y^3 + 4*y^4 + 6*x^3 - 8*x^2*y - 24*x*y^2 + 4*y^3 + 2*x*y - 6*y^2 - 6*y,
        'generators': lambda q: get_X0_generators(61, q),
    },

    65: {
        'type': 'bielliptic',
        'quadratic_family_type': 'bielliptic',
        'quotient_label': '65a1',
        'poly_xy': lambda x, y: 16*x^7 + 10*x^6*y + 41*x^5*y^2 - 33*x^4*y^3 + 27*x^3*y^4 - 41*x^2*y^5 + 17*x*y^6 - y^7 + 26*x^6 + 4*x^5*y + 37*x^4*y^2 - 6*x^3*y^3 + 68*x^2*y^4 - 73*x*y^5 + 16*y^6 + 11*x^5 - 70*x^4*y + 44*x^3*y^2 - 11*x^2*y^3 + 61*x*y^4 - 26*y^5 - 12*x^4 - 48*x^3*y + 36*x^2*y^2 - 51*x*y^3 + 30*y^4 - 9*x^3 + 18*x*y^2 - 18*y^3 + 9*y^2,
        'generators': lambda q: get_X0_generators(65, q),
    },

    79: {
        'type': 'bielliptic',
        'quadratic_family_type': 'bielliptic',
        'quotient_label': '79a1',
        'bielliptic_x_degree_cap': 18,
        'bielliptic_relation_degree_cap': 12,
        'bielliptic_linear_degree_cap': 12,
        'poly_xy': lambda x, y: 3*x^10 + 2*x^9*y + 75*x^8*y^2 + 88*x^7*y^3 + 614*x^6*y^4 + 1036*x^5*y^5 + 2638*x^4*y^6 + 4056*x^3*y^7 + 10231*x^2*y^8 + 9154*x*y^9 + 4871*y^10 + 22*x^9 + 30*x^8*y + 384*x^7*y^2 + 880*x^6*y^3 + 2492*x^5*y^4 + 6908*x^4*y^5 + 11696*x^3*y^6 + 19776*x^2*y^7 + 19198*x*y^8 + 12342*y^9 + 73*x^8 + 136*x^7*y + 1044*x^6*y^2 + 2536*x^5*y^3 + 5886*x^4*y^4 + 11480*x^3*y^5 + 17060*x^2*y^6 + 10808*x*y^7 + 8321*y^8 + 138*x^7 + 318*x^6*y + 1498*x^5*y^2 + 2974*x^4*y^3 + 4430*x^3*y^4 + 7082*x^2*y^5 - 818*x*y^6 - 3334*y^7 + 156*x^6 + 344*x^5*y + 1068*x^4*y^2 + 560*x^3*y^3 + 308*x^2*y^4 - 2312*x*y^5 - 6268*y^6 + 100*x^5 + 136*x^4*y + 88*x^3*y^2 - 1168*x^2*y^3 - 956*x*y^4 - 2296*y^5 + 32*x^4 - 96*x^3*y - 424*x^2*y^2 - 816*x*y^3 - 232*y^4 - 140*x^2*y - 408*x*y^2 - 220*y^3 - 100*x*y - 156*y^2 - 32*y,
        'generators': lambda q: get_X0_generators(79, q),
    },

    83: {
        'type': 'bielliptic',
        'quadratic_family_type': 'bielliptic',
        'quotient_label': '83a1',
        'bielliptic_x_degree_cap': 18,
        'bielliptic_relation_degree_cap': 12,
        'bielliptic_linear_degree_cap': 12,
        'poly_xy': lambda x, y: 12*x^12 - 132*x^11*y + 1104*x^10*y^2 - 4596*x^9*y^3 + 14612*x^8*y^4 - 16584*x^7*y^5 + 15776*x^6*y^6 - 23880*x^5*y^7 + 18260*x^4*y^8 - 5300*x^3*y^9 + 10832*x^2*y^10 - 2756*x*y^11 + 844*y^12 + 6*x^11 + 418*x^10*y - 2126*x^9*y^2 + 13478*x^8*y^3 - 3812*x^7*y^4 + 4212*x^6*y^5 + 1220*x^5*y^6 + 492*x^4*y^7 + 3646*x^3*y^8 + 8714*x^2*y^9 + 7210*x*y^10 + 3406*y^11 + 131*x^10 - 294*x^9*y + 7447*x^8*y^2 + 9688*x^7*y^3 - 746*x^6*y^4 + 44156*x^5*y^5 + 1334*x^4*y^6 + 13848*x^3*y^7 - 6553*x^2*y^8 + 18106*x*y^9 + 9139*y^10 + 140*x^9 + 2372*x^8*y + 9760*x^7*y^2 + 5184*x^6*y^3 + 47448*x^5*y^4 + 29928*x^4*y^5 + 38080*x^3*y^6 + 8032*x^2*y^7 + 15676*x*y^8 + 6196*y^9 + 467*x^8 + 3648*x^7*y + 6748*x^6*y^2 + 25840*x^5*y^3 + 29082*x^4*y^4 + 46240*x^3*y^5 + 46444*x^2*y^6 + 18864*x*y^7 - 181*y^8 + 612*x^7 + 2748*x^6*y + 9524*x^5*y^2 + 11196*x^4*y^3 + 18220*x^3*y^4 + 47572*x^2*y^5 + 22204*x*y^6 + 5172*y^7 + 440*x^6 + 2272*x^5*y + 2816*x^4*y^2 - 3040*x^3*y^3 + 15384*x^2*y^4 + 8512*x*y^5 + 8176*y^6 + 268*x^5 + 736*x^4*y - 3544*x^3*y^2 - 2864*x^2*y^3 - 5844*x*y^4 - 144*y^5 + 116*x^4 - 612*x^3*y - 2732*x^2*y^2 - 6588*x*y^3 - 5896*y^4 - 440*x^2*y - 2240*x*y^2 - 4024*y^3 - 268*x*y - 1116*y^2 - 116*y,
        'generators': lambda q: get_X0_generators(83, q),
    },

    89: {
        'type': 'bielliptic',
        'quadratic_family_type': 'bielliptic',
        'quotient_label': '89a1',
        'bielliptic_x_degree_cap': 20,
        'bielliptic_x_max_deg_y_plan': [8],
        'bielliptic_relation_degree_cap': 12,
        'bielliptic_linear_degree_cap': 12,
        'poly_xy': lambda x, y: 619583*x^12 + 5222610*x^11*y + 32394101*x^10*y^2 + 103756263*x^9*y^3 + 173971721*x^8*y^4 - 8706354*x^7*y^5 - 453395507*x^6*y^6 - 538173550*x^5*y^7 + 728281962*x^4*y^8 + 1965189737*x^3*y^9 + 1624231727*x^2*y^10 + 583001497*x*y^11 + 76704706*y^12 + 1280730*x^11 + 4291418*x^10*y + 19136818*x^9*y^2 + 37994920*x^8*y^3 + 115694621*x^7*y^4 + 107812368*x^6*y^5 - 175935656*x^5*y^6 - 1605479500*x^4*y^7 - 2031733746*x^3*y^8 - 661174712*x^2*y^9 + 141683076*x*y^10 + 70778591*y^11 + 1905758*x^10 + 6083563*x^9*y + 32888190*x^8*y^2 + 38500747*x^7*y^3 + 27415928*x^6*y^4 - 4401525*x^5*y^5 + 789807934*x^4*y^6 + 682522736*x^3*y^7 - 171659106*x^2*y^8 - 155199589*x*y^9 - 749756*y^10 + 2009345*x^9 + 3236347*x^8*y + 26381802*x^7*y^2 + 29948120*x^6*y^3 - 4344917*x^5*y^4 - 371137167*x^4*y^5 - 19387546*x^3*y^6 + 189723685*x^2*y^7 - 17261499*x*y^8 - 18677130*y^9 + 1940099*x^8 + 853369*x^7*y + 11798757*x^6*y^2 + 7307591*x^5*y^3 + 64913374*x^4*y^4 - 105681468*x^3*y^5 + 23153147*x^2*y^6 + 50549392*x*y^7 + 671799*y^8 + 1183445*x^7 + 1065427*x^6*y + 6575775*x^5*y^2 - 8605153*x^4*y^3 + 9803125*x^3*y^4 - 44039490*x^2*y^5 + 4790855*x*y^6 + 4956877*y^7 + 295245*x^6 + 656100*x^5*y + 3375270*x^4*y^2 + 1093500*x^3*y^3 + 3761640*x^2*y^4 - 5737230*x*y^5 + 1102248*y^6 - 295245*y^5,
        'generators': lambda q: get_X0_generators(89, q),
    },

    101: {
        'type': 'bielliptic',
        'quadratic_family_type': 'bielliptic',
        'quotient_label': '101a1',
        'bielliptic_x_degree_cap': 22,
        'bielliptic_x_max_deg_y_plan': [9],
        'bielliptic_relation_degree_cap': 14,
        'bielliptic_linear_degree_cap': 14,
        'poly_xy': lambda x, y: 1591843521252500*x^14 - 3055467566470720*x^13*y + 6051982783840271*x^12*y^2 + 492097965730450*x^11*y^3 - 7840595803363054*x^10*y^4 + 6830339639224345*x^9*y^5 + 1446100795684752*x^8*y^6 - 6953586315617682*x^7*y^7 + 3585523782483297*x^6*y^8 - 426481606214676*x^5*y^9 - 1257572630380914*x^4*y^10 + 1089584146714383*x^3*y^11 - 1862936618161980*x^13 + 2619875898030009*x^12*y - 25807391144544807*x^11*y^2 + 25620592320597249*x^10*y^3 - 19281971974118508*x^9*y^4 - 17160217689302019*x^8*y^5 + 49992256815394581*x^7*y^6 - 21715584163979265*x^6*y^7 - 4040555086297692*x^5*y^8 + 17042387128790202*x^4*y^9 - 10044368054476776*x^3*y^10 + 426481606214676*x^2*y^11 + 1257572630380914*x*y^12 - 1089584146714383*y^13 + 7035315347835990*x^12 + 408201126521985*x^11*y + 14029624047861288*x^10*y^2 + 14706893098559106*x^9*y^3 + 52632818797912461*x^8*y^4 - 114927010266997548*x^7*y^5 + 51866657835375483*x^6*y^6 + 20338732841639535*x^5*y^7 - 74932351641265842*x^4*y^8 + 37571185859368383*x^3*y^9 + 2836584100993914*x^2*y^10 - 10088800813172520*x*y^11 + 6458844271993479*y^12 - 15923627888515866*x^11 + 32597453375539920*x^10*y - 96132841910134749*x^9*y^2 - 21318118586035272*x^8*y^3 + 146903555087139465*x^7*y^4 - 137760745321982457*x^6*y^5 - 32725598004066060*x^5*y^6 + 165865688634270213*x^4*y^7 - 101231997936473502*x^3*y^8 - 7005844028674980*x^2*y^9 + 35970137164738953*x*y^10 - 22803982045555239*y^11 + 12419404511376168*x^10 + 6693795939069090*x^9*y + 102500169394099248*x^8*y^2 - 204638514449846022*x^7*y^3 + 298014449930730684*x^6*y^4 + 41964212277421461*x^5*y^5 - 237543620230190637*x^4*y^6 + 237358289003783814*x^3*y^7 + 12849732642222378*x^2*y^8 - 81374813734347837*x*y^9 + 60570013136091264*y^10 - 45541143892609200*x^9 + 39203372078274816*x^8*y + 24547561955868663*x^7*y^2 - 319605691037076918*x^6*y^3 - 57295154455951302*x^5*y^4 + 193604684017024152*x^4*y^5 - 409479807905532432*x^3*y^6 - 6392052205644888*x^2*y^7 + 122443058382693273*x*y^8 - 124901410526841033*y^9 + 37010237253059916*x^8 - 83892461832148122*x^7*y + 402593870472687837*x^6*y^2 - 72188997312885813*x^5*y^3 - 33416766550831536*x^4*y^4 + 485345994686788563*x^3*y^5 - 56255281000971381*x^2*y^6 - 128329757340111144*x*y^7 + 207107528956263072*y^8 - 50572002006829944*x^7 - 104417240169244581*x^6*y + 41127874822627938*x^5*y^2 - 28770181140104289*x^4*y^3 - 375105275075973033*x^3*y^4 + 145497853852603668*x^2*y^5 + 62856811375504659*x*y^6 - 274773339405277956*y^7 + 98973692275963482*x^6 - 151692724775550726*x^5*y + 163255952849899338*x^4*y^2 + 123826467964307850*x^3*y^3 - 148288046508320391*x^2*y^4 + 68842033695900999*x*y^5 + 299245942613768886*y^6 - 24828101634526236*x^5 - 74628044327836170*x^4*y - 59803280079195960*x^3*y^2 + 26030753087851521*x^2*y^3 - 201563591155700679*x*y^4 - 258842010670683048*y^5 + 107147308581932328*x^4 - 18048422073834576*x^3*y + 126833289643219860*x^2*y^2 + 225755346136606254*x*y^3 + 177158017858372551*y^4 - 29977007570004168*x^3 - 90397802941825941*x^2*y - 139299483309656382*x*y^2 - 86811309655339122*y^3 + 62305255419623442*x^2 + 37897758411769854*x*y + 36432581341967586*y^2 - 12585234558152268*x - 15354741052925343*y + 10271089177784586,
        'generators': lambda q: get_X0_generators(101, q),
    },

    131: {
        # The public notebook uses a dedicated fixed-degree mod-p search path
        # for this level to avoid changing the default solver behavior elsewhere.
        'type': 'bielliptic',
        'quadratic_family_type': 'bielliptic',
        'quotient_label': '131a1',
        'bielliptic_x_degree_cap': 22,
        'bielliptic_relation_degree_cap': 18,
        'bielliptic_linear_degree_cap': 18,
        'poly_xy': lambda x, y: 1440000*x^20*y^4 - 120000*x^20*y^3 - 8400000*x^19*y^4 + 2500*x^20*y^2 + 950000*x^19*y^3 + 32170000*x^18*y^4 - 7715000*x^19*y^2 + 4900000*x^18*y^3 - 92220000*x^17*y^4 + 120000*x^19*y + 51562500*x^18*y^2 - 45110000*x^17*y^3 + 218090000*x^16*y^4 - 2500*x^19 - 832500*x^18*y - 193585000*x^17*y^2 + 177995000*x^16*y^3 - 440720000*x^15*y^4 + 6272500*x^18 - 782500*x^17*y + 529115000*x^16*y^2 - 500510000*x^15*y^3 + 778069200*x^14*y^4 - 38212500*x^17 + 16105000*x^16*y - 1147025000*x^15*y^2 + 1079089600*x^14*y^3 - 1212749600*x^13*y^4 + 134395000*x^16 - 65900000*x^15*y + 2103511100*x^14*y^2 - 1953732800*x^13*y^3 + 1685860000*x^12*y^4 - 348977500*x^15 + 195155200*x^14*y - 3342078300*x^13*y^2 + 2994643200*x^12*y^3 - 2095036000*x^11*y^4 + 722352000*x^14 - 414925350*x^13*y + 4608669600*x^12*y^2 - 3937873800*x^11*y^3 + 2333492000*x^10*y^4 - 1300549000*x^13 + 712162100*x^12*y - 5482547900*x^11*y^2 + 4441402600*x^10*y^3 - 2316708000*x^9*y^4 + 2049775700*x^12 - 1003055900*x^11*y + 5573688300*x^10*y^2 - 4235801600*x^9*y^3 + 2048096464*x^8*y^4 - 2826627050*x^11 + 1136886800*x^10*y - 4667501800*x^9*y^2 + 3389691856*x^8*y^3 - 1597203344*x^7*y^4 + 3401577850*x^10 - 1108274550*x^9*y + 3058746784*x^8*y^2 - 2132551176*x^7*y^3 + 1094260704*x^6*y^4 - 3535256975*x^9 + 737553856*x^8*y - 1118145964*x^7*y^2 + 905347016*x^6*y^3 - 642865760*x^5*y^4 + 3245793089*x^8 - 188045526*x^7*y - 396948176*x^6*y^2 - 89188640*x^5*y^3 + 327210064*x^4*y^4 - 2509153644*x^7 - 356738334*x^6*y + 985496140*x^5*y^2 - 193826944*x^4*y^3 - 150107392*x^3*y^4 + 1486843029*x^6 + 526657660*x^5*y - 774926216*x^4*y^2 + 139456032*x^3*y^3 + 61336672*x^2*y^4 - 503208235*x^5 - 358875094*x^4*y + 267679948*x^3*y^2 - 53912912*x^2*y^3 - 21220336*x*y^4 + 55951614*x^4 + 86062232*x^3*y - 17301268*x^2*y^2 + 9557056*x*y^3 + 4355472*y^4 + 10680083*x^3 + 11045388*x^2*y - 15804316*x*y^2 + 1373688*y^3 + 20518947*x^2 + 4495106*x*y + 9030732*y^2 + 3029939*x + 799788*y + 506022,
        'generators': lambda q: get_X0_generators(131, q),
    },


    28: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^6 + 10*x^4 + 25*x^2 - y^2 + 28,
        'generators': lambda q: get_X0_generators(28, q),
    },

    29: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^6 - 4*x^5 - 12*x^4 + 2*x^3 + 8*x^2 - y^2 + 8*x - 7,
        'generators': lambda q: get_X0_generators(29, q),
    },

    30: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^8 + 6*x^7 + 9*x^6 + 6*x^5 - 4*x^4 - 6*x^3 + 9*x^2 - y^2 - 6*x + 1,
        'generators': lambda q: get_X0_generators(30, q),
    },

    31: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^6 - 8*x^5 + 6*x^4 + 18*x^3 - 11*x^2 - y^2 - 14*x - 3,
        'generators': lambda q: get_X0_generators(31, q),
    },

    33: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^8 + 10*x^6 - 8*x^5 + 47*x^4 - 40*x^3 + 82*x^2 - y^2 - 44*x + 33,
        'generators': lambda q: get_X0_generators(33, q),
    },

    35: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^8 - 4*x^7 - 6*x^6 - 4*x^5 - 9*x^4 + 4*x^3 - 6*x^2 - y^2 + 4*x + 1,
        'generators': lambda q: get_X0_generators(35, q),
    },

    39: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^8 - 6*x^7 + 3*x^6 + 12*x^5 - 23*x^4 + 12*x^3 + 3*x^2 - y^2 - 6*x + 1,
        'generators': lambda q: get_X0_generators(39, q),
    },

    40: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^8 + 8*x^6 - 2*x^4 + 8*x^2 - y^2 + 1,
        'generators': lambda q: get_X0_generators(40, q),
    },

    41: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^8 - 4*x^7 - 8*x^6 + 10*x^5 + 20*x^4 + 8*x^3 - 15*x^2 - y^2 - 20*x - 8,
        'generators': lambda q: get_X0_generators(41, q),
    },

    46: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^12 - 2*x^11 + 5*x^10 + 6*x^9 - 26*x^8 + 84*x^7 - 113*x^6 + 134*x^5 - 64*x^4 + 26*x^3 + 12*x^2 - y^2 + 8*x - 7,
        'generators': lambda q: get_X0_generators(46, q),
    },

    47: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^10 - 6*x^9 + 11*x^8 - 24*x^7 + 19*x^6 - 16*x^5 - 13*x^4 + 30*x^3 - 38*x^2 - y^2 + 28*x - 11,
        'generators': lambda q: get_X0_generators(47, q),
    },

    48: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^8 + 14*x^4 - y^2 + 1,
        'generators': lambda q: get_X0_generators(48, q),
    },

    50: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^6 - 4*x^5 - 10*x^3 - y^2 - 4*x + 1,
        'generators': lambda q: get_X0_generators(50, q),
    },

    59: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^12 - 8*x^11 + 22*x^10 - 28*x^9 + 3*x^8 + 40*x^7 - 62*x^6 + 40*x^5 - 3*x^4 - 24*x^3 + 20*x^2 - y^2 - 4*x - 8,
        'generators': lambda q: get_X0_generators(59, q),
    },

    71: {
        'type': 'hyperelliptic',
        'quadratic_family_type': 'hyperelliptic',
        'poly_xy': lambda x, y: x^14 - 10*x^13 + 37*x^12 - 66*x^11 + 66*x^10 - 48*x^9 + 15*x^8 + 40*x^7 - 66*x^6 + 66*x^5 - 58*x^4 + 40*x^3 - 27*x^2 - y^2 + 6*x - 7,
        'generators': lambda q: get_X0_generators(71, q),
    },

        # [N=67] Genus 7 (General)
        67: {
            'type': 'general',
            # Galbraith model for X0(67)
            #eq1 = x^3 + 18296/12705*x^2 - 150013/16940*x*y + 13961/10164*y^2 + 187/56*x + 8875/20328*y + 4243/4840
            #eq2 = x^2*y + 1071/605*x^2 - 133289/2420*x*y + 4531/484*y^2 + 23/8*x + 5105/968*y + 29253/4840
            #eq3 = x*y^2 + 5271/605*x^2 - 651839/2420*x*y + 22181/484*y^2 + 113/8*x + 24987/968*y + 142903/4840
            #eq4 = y^3 + 236397/5566*x^2 - 14681699/11132*x*y + 2501287/11132*y^2 + 12685/184*x + 2817499/22264*y + 3206487/22264
            'poly_xy': lambda x, y:  4840*x*y^2 + 42168*x^2 - 1303678*x*y + 221810*y^2 + 68365*x + 124935*y + 142903,
            'generators': lambda q: get_X0_generators(67, q),
            'base_points': [[2, 9,3]]
        },
        163: {
            'type': 'general',
            #'poly_xy': lambda x, y: 27*x^14*y^2 + 22*x^13*y^3 + 102*x^12*y^4 + 121*x^11*y^5 + 173*x^10*y^6 + 192*x^9*y^7 + 156*x^8*y^8 + 109*x^7*y^9 + 66*x^6*y^10 + 16*x^5*y^11 + 8*x^4*y^12 - 186*x^13*y^2 - 121*x^12*y^3 - 633*x^11*y^4 - 558*x^10*y^5 - 884*x^9*y^6 - 842*x^8*y^7 - 637*x^7*y^8 - 542*x^6*y^9 - 266*x^5*y^10 - 153*x^4*y^11 - 74*x^3*y^12 - 16*x^2*y^13 - 8*x*y^14 - 18*x^13*y + 629*x^12*y^2 + 398*x^11*y^3 + 1939*x^10*y^4 + 1612*x^9*y^5 + 2611*x^8*y^6 + 2137*x^7*y^7 + 1797*x^6*y^8 + 1312*x^5*y^9 + 681*x^4*y^10 + 376*x^3*y^11 + 126*x^2*y^12 + 36*x*y^13 + 8*y^14 + 12*x^12*y - 1456*x^11*y^2 - 1150*x^10*y^3 - 4073*x^9*y^4 - 3510*x^8*y^5 - 4664*x^7*y^6 - 3598*x^6*y^7 - 2721*x^5*y^8 - 1599*x^4*y^9 - 750*x^3*y^10 - 271*x^2*y^11 - 66*x*y^12 - 12*y^13 + 27*x^12 + 211*x^11*y + 2444*x^10*y^2 + 2496*x^9*y^3 + 5879*x^8*y^4 + 5058*x^7*y^5 + 5649*x^6*y^6 + 3883*x^5*y^7 + 2455*x^4*y^8 + 1163*x^3*y^9 + 415*x^2*y^10 + 97*x*y^11 + 18*y^12 - 126*x^11 - 647*x^10*y - 2988*x^9*y^2 - 3547*x^8*y^3 - 5929*x^7*y^4 - 5019*x^6*y^5 - 4441*x^5*y^6 - 2521*x^4*y^7 - 1277*x^3*y^8 - 410*x^2*y^9 - 96*x*y^10 - 13*y^11 + 255*x^10 + 936*x^9*y + 2720*x^8*y^2 + 3350*x^7*y^3 + 4221*x^6*y^4 + 3195*x^5*y^5 + 2165*x^4*y^6 + 988*x^3*y^7 + 353*x^2*y^8 + 62*x*y^9 + 11*y^10 - 300*x^9 - 858*x^8*y - 1804*x^7*y^2 - 2062*x^6*y^3 - 1952*x^5*y^4 - 1202*x^4*y^5 - 607*x^3*y^6 - 190*x^2*y^7 - 33*x*y^8 - 4*y^9 + 229*x^8 + 513*x^7*y + 789*x^6*y^2 + 748*x^5*y^3 + 529*x^4*y^4 + 247*x^3*y^5 + 84*x^2*y^6 + 10*x*y^7 + y^8 - 110*x^7 - 169*x^6*y - 194*x^5*y^2 - 140*x^4*y^3 - 65*x^3*y^4 - 22*x^2*y^5 - 2*x*y^6 + 25*x^6 + 20*x^5*y + 24*x^4*y^2 + 8*x^3*y^3 + 4*x^2*y^4,
            'poly_xy': lambda x, y: -1600*y^2 + 17230*y + 31110*x - 5019,
            'generators': lambda q: get_X0_generators(163, q),
            'base_points': [[9,-12,10]]
        }
    }
    
    if N not in models:
        raise NotImplementedError(f"Model data for N={N} is not yet implemented in the library.")

    model = dict(models[N])
    model.update(get_quadratic_family_metadata(N))
    return model

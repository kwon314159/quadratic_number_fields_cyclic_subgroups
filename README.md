# Families of Elliptic Curves Over Quadratic Number Fields With Prescribed Cyclic N-Subgroups

This repository contains SageMath code accompanying the paper:

> **Families of Elliptic Curves over Quadratic Number Fields with Prescribed Cyclic N-Subgroups**  
> *Daeyeol Jeon and Yongjae Kwon*

The code computes explicit quadratic-point families on selected modular curves
`X_0(N)`. The resulting formulas give families of elliptic curves over
quadratic number fields with prescribed cyclic `N`-subgroups.

It includes:

- an interactive notebook for running the computations,
- the supporting Sage source files,
- one canonical results file `isogeny_results.txt`,
- and a read-only notebook viewer for that results file.

The bundled results file contains verified formulas for selected levels up to
and including `N = 131`.

## Key Features

- `Families` Explicit elliptic, hyperelliptic, and bielliptic quadratic-point families.
- `Maps` Rational-map recovery for the Weierstrass coefficient maps `a_4`, `a_6`, `a'_4`, and `a'_6`.
- `Solver` Degree- and precision-controlled q-expansion linear algebra over `QQ`.
- `Notebook UI` Interactive level selection and one-click computation.
- `Stored Results` A canonical text output file and a notebook viewer for browsing the verified formulas.
- `N=131 Path` Dedicated fixed-degree mod-p screening path for the largest bundled level.

## Supported Levels

The main notebook and bundled results file cover:

```text
N = 11, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26, 27,
    28, 29, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41,
    43, 46, 47, 48, 49, 50, 53, 59, 61, 65, 71, 79,
    83, 89, 101, 131.
```

## Output Convention

The solver recovers four rational functions on `X_0(N)`:

```text
u(x_0(q),  y_0(q))  = f(q)
u_N(x_0(q), y_0(q)) = f_N(q)
v(x_0(q),  y_0(q))  = g(q)
v_N(x_0(q), y_0(q)) = g_N(q)
```

These are identified with the paper's Weierstrass coefficients by:

```text
u   = a_4,     v   = a_6,
u_N = a'_4,    v_N = a'_6.
```

The normalization follows the modular-form definitions in
`setup_modular_forms` inside `sagecode/solver_core.sage`.

## Repository Contents

- `X0_main.ipynb`  
  Main interactive notebook for computing the families.
- `sagecode/solver_core.sage`  
  Core search, verification, and report-construction routines.
- `sagecode/x0_model_data.sage`  
  Curve metadata, plane models, and lazy-loading helpers for generator tables.
- `sagecode/x0_generator_startup.sage`, `sagecode/x0_generator_mid.sage`, `sagecode/x0_generator_high.sage`  
  Chunked tables of explicit generator `q`-expansions.
- `isogeny_results.txt`  
  Canonical stored output for the recovered families.
- `isogeny_results_viewer.ipynb`  
  Read-only browser for `isogeny_results.txt`.

## Execution Flow

```text
input level N
    |
    v
get_model_data(N)
    |
    +--> plane model for X_0(N)
    +--> generator supplier x_0(q), y_0(q)
    +--> family metadata: elliptic / hyperelliptic / bielliptic
    |
    v
setup_modular_forms(N, prec)
    |
    +--> target q-series f, g, f_N, g_N
    |
    v
_build_search_runtime(...)
    |
    +--> precision, valuation, and verification thresholds
    +--> lazy-loaded q-expansion context
    |
    v
_solve_maps_for_level(...)
    |
    +--> recover u, u_N, v, v_N
    |
    v
build_family_report(...)
    |
    +--> elliptic output
    +--> hyperelliptic output
    +--> bielliptic quotient / lift data
    |
    v
final notebook report or stored text section
```

## Main Notebook Behavior

When `N = 131` is selected in `X0_main.ipynb`, the notebook uses a dedicated
fixed-degree mod-p search path inside `sagecode/solver_core.sage`. Other levels
continue to use the default public solver path.

This is still a computation path, not a lookup from `isogeny_results.txt`. The
special handling is used because the plain public search routine is much less
effective on `N = 131`: the relevant rational maps are recovered at the fixed
degrees `u = 36`, `u_N = 36`, `v = 39`, `v_N = 39`, and the code first applies
mod-p kernel screening before lifting exact candidates over `QQ`.

## Notes on Bielliptic Output

Bielliptic levels include extra data beyond the four coefficient maps.
Depending on the level, the stored section may also contain:

- the elliptic quotient curve,
- the quotient map `x_E(t,\alpha_t)`,
- a chosen lift primitive variable on `X_0(N)`,
- a quadratic relation for that primitive variable over the quotient coordinates,
- a linear relation recovering the other coordinate,
- and sample quadratic fibers above selected quotient points.

The labels `Lift Primitive Variable on X_0(N)` and
`Lift Other Variable on X_0(N)` are reconstruction choices made by the code.
They are not canonical and may differ from one bielliptic level to another.

For very large formulas, especially in some bielliptic cases, the notebooks may
skip full MathJax rendering and show raw LaTeX instead, to keep the interface
responsive.

## Requirements

- SageMath 10.6+ with Jupyter notebook support
- `ipywidgets` enabled in the notebook environment

Run the notebooks from the repository root so that the relative file paths
resolve correctly.

## Quick Start

### Run the main notebook

1. Start Jupyter through SageMath, for example:

   ```bash
   sage -n jupyter
   ```

2. Open `X0_main.ipynb`.
3. Run the setup cell.
4. Choose a level `N` in the notebook UI and launch the computation.

### Browse the stored results

1. Keep `isogeny_results_viewer.ipynb` and `isogeny_results.txt` in the same directory.
2. Open `isogeny_results_viewer.ipynb`.
3. Run the notebook and choose a level from the dropdown.

## AI-Assisted Development Note

Parts of implementation, refactoring, and code-organization tasks were assisted
by OpenAI Codex. All mathematical definitions, derivations, and final computed
results were checked and validated by the authors.

## Citation

If you use this code in research, please cite:

```bibtex
@article{jeon_kwon_quadratic_cyclic_subgroups,
  title={Families of elliptic curves over quadratic number fields with prescribed cyclic {N}-subgroups},
  author={Jeon, Daeyeol and Kwon, Yongjae},
  note={Manuscript},
  year={2026}
}
```

## License

MIT License. See `LICENSE`.

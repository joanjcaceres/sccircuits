# BBQ Validation

This page explains the code-backed checks that make the `BBQ` matrix-to-modes
workflow auditable. The examples here are not separate benchmark scripts; they
are documented versions of behavior covered by `tests/test_bbq.py`.

Run the full validation suite with:

```bash
pixi run -e sccircuits test
```

Build the rendered documentation with:

```bash
pixi run -e sccircuits docs-build
```

## Single-Mode LC Frequency and Units

Test:

```text
test_single_mode_lc_frequency_and_units
```

For a one-coordinate LC oscillator with capacitance `C` and inductance `L`,
`BBQ` should report

$$
\omega = \frac{1}{\sqrt{LC}},
\qquad
f_{\mathrm{GHz}} = \frac{\omega}{2\pi\cdot 10^9}.
$$

The same test verifies the one-branch phase zero-point fluctuation:

$$
\varphi_{\mathrm{zpf}}
=
\frac{1}{\varphi_0\sqrt{C}}
\sqrt{\frac{\hbar}{2\omega}}.
$$

This check protects the unit convention: angular frequencies are in rad/s,
ordinary frequencies are in GHz, and `branch_phase_zpfs` is dimensionless.

## Generalized Eigenproblem

Test:

```text
test_generalized_solver_matches_scipy_reference
```

For a nonsingular two-node circuit, the result should match SciPy's generalized
eigenproblem directly:

$$
\mathbf{K}\mathbf{v}_k
=
\omega_k^2\mathbf{C}\mathbf{v}_k.
$$

The test also checks the normalization and diagonalization identities:

$$
\mathbf{U}^{T}\mathbf{C}\mathbf{U} = \mathbb{1},
\qquad
\mathbf{U}^{T}\mathbf{K}\mathbf{U} = \mathbf{\Omega}^2.
$$

This is the baseline case: when no frozen, null-capacitance, or free
directions are present, `BBQ` reduces to the standard generalized
eigenproblem.

## Frozen Coordinate Schur Reduction

Test:

```text
test_si_scale_frozen_coordinate_uses_schur_reduction_and_reconstruction
```

This test uses a two-coordinate matrix where the second coordinate has zero
capacitance. `BBQ` eliminates it by minimizing the potential. For the specific
stiffness matrix in the test, the effective stiffness is

```text
6.0e9 - (2.0e9 * 2.0e9 / 4.0e9)
```

and the reconstructed frozen coordinate is `-0.5` times the active coordinate.
The test verifies both the frequency and the reconstructed branch ZPF row. This
is the check that eliminated coordinates still contribute correctly to branch
quantities in the original node basis.

The companion failure test is:

```text
test_frozen_coordinate_requires_positive_definite_stiffness_block
```

It verifies that `BBQ` rejects a frozen block that cannot define a unique
potential minimum.

## Free or Zero-Potential Direction

Test:

```text
test_singular_inverse_inductance_drops_dc_mode
```

This test uses two finite-capacitance coordinates connected by a singular
stiffness matrix. One direction is a floating/DC mode and should not become a
finite oscillator. `BBQ` separates the zero-potential direction, uses the
charge-sector Schur complement for the oscillator capacitance, and retains one
positive-frequency mode.

The test verifies:

- only one positive-frequency mode remains;
- `U.T @ C @ U` is the identity;
- `U.T @ K @ U` gives the retained `omega^2`;
- the branch ZPF matches the analytic branch-amplitude expression.

This guards against the common mistake of discarding free variables before the
Legendre-transform logic has been accounted for.

## Null Capacitance Direction

Test:

```text
test_exact_singular_capacitance_matrix_uses_positive_subspace
```

This case uses a capacitance matrix with one exact null direction. `BBQ` keeps
the positive capacitance eigenvector, projects the stiffness matrix into that
subspace, and reconstructs a full node-basis mode whose entries sum to zero.

The expected retained capacitance is `2 * capacitance`, and only one physical
mode remains. The branch ZPF is then computed from the reconstructed
differential mode.

## Branch Direction and cQEDraw Row Order

Tests:

```text
test_branch_reversal_flips_phase_zpf_only
test_multiple_nonlinear_branches_return_branch_by_mode_zpfs
test_junction_records_outputs_follow_input_row_order
test_junction_records_phase_reversal_flips_only_selected_junction
```

The branch convention is directional. Reversing a branch flips the sign of the
corresponding phase ZPF row but does not change the mode frequencies.

For cQEDraw-style `junctions`, the row order is the contract. The tests verify
that:

- `branch_phase_nodes` follows input junction order;
- `branch_incidence_matrix` rows follow the same order;
- `branch_phase_zpfs` rows follow the same order;
- `josephson_energies_ghz` follows the same order when exported.

This is the behavior drawing tools should rely on when mapping computed
modal quantities back onto drawn Josephson junctions.

## Documentation Launch Checklist

The repository is configured to publish the MkDocs site to GitHub Pages, but
the public URL should be treated as launched only after it is verified.

Before calling the documentation site public, confirm:

- the documentation deployment workflow has been merged to `main`;
- GitHub Pages is enabled for the repository;
- `https://joanjcaceres.github.io/sccircuits/` returns `200 OK`;
- the GitHub repository sidebar website field points to that URL;
- `README.md` and `pyproject.toml` point to the same documentation URL.

If the URL returns `404` after merge, the likely blocker is repository Pages
configuration rather than MkDocs content.


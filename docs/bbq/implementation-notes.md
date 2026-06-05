# BBQ Implementation Notes

This page maps the matrix quantization derivation to the current
`sccircuits.BBQ` implementation. It is meant for researchers who want to audit
the code path from input matrices to frequencies and branch phase zero-point
fluctuations.

For the full derivation, read the
[Mathematical Reference](../theory/circuit-matrix-quantization.md). This page is
the code-facing companion to that derivation.

## Symbols and Code Names

| Mathematical object | Meaning | BBQ code name |
| --- | --- | --- |
| $\mathbf{C}$ | capacitance matrix | `capacitance_matrix` |
| $\mathbf{K} = \mathbf{L}^{-1}$ | inverse-inductance or stiffness matrix | `inverse_inductance_matrix`, then `stiffness_matrix` |
| $\mathbf{B}$ | branch-by-node incidence matrix | `branch_incidence_matrix` |
| $\mathbf{U}$ | reconstructed capacitance-normalized mode matrix | `normal_mode_vectors` |
| $\omega_k^2$ | oscillator eigenvalues | `angular_frequencies_squared` |
| $\omega_k / 2\pi$ | ordinary frequencies | `frequencies_ghz` |
| $\varphi_{\mathrm{zpf}}$ | branch-by-mode phase ZPF matrix | `branch_phase_zpfs` |

The linear solve starts in `_linear_mode_solution(...)`. That method is a
thin sequence of reduction stages followed by one final generalized
eigenproblem.

## Stage 1: Frozen Variables

Code stage:

```python
_remove_frozen_variables_by_potential_minimization(...)
```

Frozen coordinates have no capacitance row or column. They carry no kinetic
energy, so `BBQ` does not treat them as zero-frequency oscillators. It solves
their algebraic potential-minimization equation and substitutes the result back
into the active coordinates.

With active coordinates `a` and frozen coordinates `f`, the code forms the
matrix blocks `K_aa`, `K_af`, `K_fa`, and `K_ff`. The frozen-coordinate block
must be positive definite. The effective stiffness is the Schur complement:

$$
\mathbf{K}_{\mathrm{eff}}
=
\mathbf{K}_{aa}
-
\mathbf{K}_{af}\mathbf{K}_{ff}^{-1}\mathbf{K}_{fa}.
$$

The same solve also constructs `original_from_active`, the reconstruction that
maps reduced active coordinates back into the original node basis. This is
why branch quantities can still be reported for the original cQEDraw nodes even
when a coordinate was eliminated internally.

Tests that cover this behavior:

- `test_si_scale_frozen_coordinate_uses_schur_reduction_and_reconstruction`
- `test_frozen_coordinate_requires_positive_definite_stiffness_block`

## Stage 2: Positive Capacitance Subspace

Code stage:

```python
_project_positive_capacitance_subspace(...)
```

After frozen coordinates are removed, the remaining capacitance matrix can
still have null directions. `BBQ` diagonalizes the reduced capacitance matrix
and keeps only positive capacitance eigenvalues:

$$
\mathbf{C}_{\mathrm{eff}}\mathbf{Q}
=
\mathbf{Q}\Lambda_C.
$$

The stiffness matrix is projected into the same basis:

$$
\mathbf{K}_{C>0}
=
\mathbf{Q}_{C>0}^{T}\mathbf{K}_{\mathrm{eff}}\mathbf{Q}_{C>0}.
$$

The code rejects negative capacitance eigenvalues beyond tolerance because
they do not define a physical kinetic energy. Tiny positive or null
capacitance directions are removed before solving for oscillator modes.

Tests that cover this behavior:

- `test_tiny_capacitance_direction_is_excluded`
- `test_exact_singular_capacitance_matrix_uses_positive_subspace`

## Stage 3: Free Variables

Code stage:

```python
_remove_free_variables_by_charge_sector(...)
```

The projected stiffness matrix can still have a nullspace. These are free or
zero-potential coordinates. They are cyclic coordinates, so their conjugate
charges are conserved. `BBQ` works in the neutral free-charge sector and uses
the effective oscillator capacitance

$$
\widetilde{\mathbf{C}}
=
\mathbf{C}_{oo}
-
\mathbf{C}_{oz}\mathbf{C}_{zz}^{-1}\mathbf{C}_{zo}.
$$

This is important: the dynamic capacitance is not always just
`C_oo`. The code also reconstructs the oscillator displacement through the
free-coordinate subspace:

$$
\mathbf{R}_{\mathrm{osc}}
=
\mathbf{R}_o
-
\mathbf{R}_z\mathbf{C}_{zz}^{-1}\mathbf{C}_{zo}.
$$

That reconstruction is stored as `positive_capacitance_from_oscillator` and is
used before mapping modes back to the original node basis.

Tests that cover this behavior:

- `test_singular_inverse_inductance_drops_dc_mode`
- `test_positive_modes_do_not_require_legacy_sign_crossing`

## Stage 4: Frequency Eigenproblem

Code stage:

```python
_solve_frequency_eigenproblem(...)
```

Only after the previous reductions does `BBQ` solve the oscillator generalized
eigenproblem:

$$
\mathbf{K}_{oo}\mathbf{W}
=
\widetilde{\mathbf{C}}\mathbf{W}\mathbf{\Omega}^2.
$$

`scipy.linalg.eigh` returns eigenvectors normalized by the reduced capacitance
matrix. After reconstruction, the full mode matrix `normal_mode_vectors`
satisfies the expected capacitance normalization:

$$
\mathbf{U}^{T}\mathbf{C}\mathbf{U}
=
\mathbb{1}.
$$

The eigenvalues are stored as `angular_frequencies_squared`; `angular_frequencies`
is the square root, and `frequencies_ghz` converts $\omega_k / 2\pi$ to GHz.

Tests that cover this behavior:

- `test_single_mode_lc_frequency_and_units`
- `test_generalized_solver_matches_scipy_reference`

## Stage 5: Branch Phase ZPFs

Code stage:

```python
_branch_phase_zpfs()
```

Branch phases are computed after the mode vectors have been reconstructed to
the original node basis. For each branch row, `B @ U` gives the branch flux per
unit normal coordinate. The phase ZPF matrix is then

$$
\varphi_{\mathrm{zpf}}^{r,k}
=
\frac{(\mathbf{B}\mathbf{U})_{r,k}}{\varphi_0}
\sqrt{\frac{\hbar}{2\omega_k}},
\qquad
\varphi_0 = \frac{\hbar}{2e}.
$$

The branch direction fixes the sign. Reversing a branch flips the sign of that
branch row without changing the mode frequencies.

Tests that cover this behavior:

- `test_branch_reversal_flips_phase_zpf_only`
- `test_multiple_nonlinear_branches_return_branch_by_mode_zpfs`
- `test_junction_records_outputs_follow_input_row_order`

## cQEDraw Contract

When `junctions=` is used, `BBQ` treats cQEDraw records as numerical branch
definitions. It does not retain web-specific identifiers, but it does preserve
row order:

- `branch_phase_nodes[i]`
- `branch_incidence_matrix[i]`
- `branch_phase_zpfs[i]`
- `josephson_energies_ghz[i]`, when present

all correspond to `junctions[i]`. Callers that need to map results back to a
drawing should keep the original `junctions` list.


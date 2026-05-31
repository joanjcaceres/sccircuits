# Circuit Matrix Quantization

This page describes the linear circuit calculation used by `sccircuits.BBQ`.
The goal is to turn a capacitance matrix `C` and an inverse inductance matrix
`L_inv` into the quantities needed by the SCCircuits Hamiltonian model:
normal-mode frequencies and phase zero-point fluctuations across a nonlinear
branch.

The code names intentionally follow the symbols below: `capacitance_matrix`
is $\mathbf{C}$, `inverse_inductance_matrix` is $\mathbf{L}^{-1}$,
`normal_mode_vectors` is $\mathbf{U}$, `branch_incidence_matrix` is
$\mathbf{B}$, and `branch_phase_zpfs` is the branch-by-mode matrix
$\varphi_{\mathrm{zpf}}$.

## Calculation Boundary

`BBQ` starts after a circuit has already been converted into matrix form. It
does not parse a circuit graph, choose independent loop fluxes, or derive a
symbolic Lagrangian from elements. Those steps belong to a graph or
lumped-circuit layer.

Today, the companion cQEDraw workflow owns drawing and graph export. A future
SCCircuits graph layer should own graph-to-Lagrangian construction, external
loop-flux handling, and topology-level variable classification. `BBQ` then acts
as the matrix-to-modes backend: it performs numerical reductions needed for
singular supplied matrices, solves the oscillator eigenproblem, computes branch
phase zero-point fluctuations, and builds dense Hamiltonian matrices from those
modal quantities.

The reductions below are numerical matrix reductions. They are not a complete
symbolic treatment of all graph constraints, external fluxes, or gauge choices.

## Linear Circuit Matrices

Use node fluxes $\Phi$. The linearized Lagrangian is

$$
\mathcal{L} =
\frac{1}{2}\dot{\Phi}^T \mathbf{C}\dot{\Phi} -
\frac{1}{2}\Phi^T \mathbf{L}^{-1}\Phi.
$$

The equations of motion are

$$
\mathbf{C}\ddot{\Phi}+\mathbf{L}^{-1}\Phi=0.
$$

With the normal-mode ansatz
$\Phi(t)=\mathbf{v}_k e^{i\omega_k t}$, the modes solve

$$
\mathbf{L}^{-1}\mathbf{v}_k=\omega_k^2 \mathbf{C}\mathbf{v}_k.
$$

This is the formal generalized eigenvalue problem for an already reduced,
positive-definite oscillator basis. When the supplied matrices contain frozen
coordinates, null capacitance directions, or zero-potential directions, `BBQ`
does not solve this equation directly. It first performs the reductions below
and then solves the final oscillator problem in the reduced basis.

## Numerical Reduction Workflow

`BBQ` first reduces singular or nearly singular matrix directions before the
final oscillator solve. All thresholds are relative to the scale of the matrix
or eigenvalues being tested; they do not use `max(1, scale)`, so SI-size
capacitances such as femtofarads are not treated as zero merely because they are
small in absolute units.

### Frozen coordinates

A coordinate is frozen when its entire row and column in $\mathbf{C}$ are zero
within the capacitance tolerance. Split dynamic coordinates $d$ from frozen
coordinates $f$:

$$
\mathbf{C} =
\begin{bmatrix}
\mathbf{C}_{dd} & 0 \\
0 & 0
\end{bmatrix},
\qquad
\mathbf{L}^{-1} =
\begin{bmatrix}
\mathbf{K}_{dd} & \mathbf{K}_{df} \\
\mathbf{K}_{fd} & \mathbf{K}_{ff}
\end{bmatrix}.
$$

The frozen coordinates carry no kinetic energy. `BBQ` minimizes the quadratic
potential with respect to them. This requires the frozen-coordinate stiffness
block $\mathbf{K}_{ff}$ to be positive definite, so the frozen coordinates are
uniquely constrained:

$$
\Phi_f = -\mathbf{K}_{ff}^{-1}\mathbf{K}_{fd}\Phi_d,
$$

which gives

$$
\begin{aligned}
\mathbf{C}_{\mathrm{eff}} &= \mathbf{C}_{dd}, \\
\mathbf{K}_{\mathrm{eff}} &= \mathbf{K}_{dd}
{}- \mathbf{K}_{df}\mathbf{K}_{ff}^{-1}\mathbf{K}_{fd}.
\end{aligned}
$$

The reconstruction

$$
\Phi =
\begin{bmatrix}
\mathbb{1} \\
-\mathbf{K}_{ff}^{-1}\mathbf{K}_{fd}
\end{bmatrix}
\Phi_d
$$

is retained so branch phases that touch an eliminated coordinate are still
computed in the original node basis.

### Null capacitance directions

After frozen-coordinate reduction, `BBQ` diagonalizes the remaining capacitance
matrix and keeps only positive capacitance eigenvalues:

$$
\mathbf{C}_{\mathrm{eff}}\mathbf{Q} = \mathbf{Q}\Lambda_C.
$$

Eigenvalues no larger than
$10^{-12}\max_i |\lambda_i(\mathbf{C}_{\mathrm{eff}})|$ are treated as null.
Negative capacitance eigenvalues beyond this tolerance are rejected. The
stiffness matrix is projected into this positive capacitance subspace.

### Zero-potential modes

The projected stiffness matrix may still have null directions, corresponding to
DC or floating modes. `BBQ` diagonalizes the projected stiffness and splits
oscillatory directions $o$ from zero-potential directions $z$. In that basis,
the capacitance matrix is partitioned as

$$
\mathbf{C}' =
\begin{bmatrix}
\mathbf{C}_{oo} & \mathbf{C}_{oz} \\
\mathbf{C}_{zo} & \mathbf{C}_{zz}
\end{bmatrix}.
$$

At fixed zero-mode charge, the oscillator capacitance is the Schur complement

$$
\mathbf{C}_{\mathrm{osc}} =
\mathbf{C}_{oo} - \mathbf{C}_{oz}\mathbf{C}_{zz}^{-1}\mathbf{C}_{zo}.
$$

The oscillator reconstruction also includes the zero-mode displacement induced
by this constraint:

$$
\mathbf{R}_{\mathrm{osc}} =
\mathbf{R}_o - \mathbf{R}_z\mathbf{C}_{zz}^{-1}\mathbf{C}_{zo}.
$$

The final oscillator problem is

$$
\mathbf{K}_{oo}\psi_k = \omega_k^2\mathbf{C}_{\mathrm{osc}}\psi_k.
$$

Only positive finite $\omega_k^2$ values are retained.

## Normalization

Each positive-frequency solution of the generalized eigenvalue problem gives
one normal-mode vector $\mathbf{v}_k$ after reconstruction to the original node
basis. `BBQ` collects the remaining physical mode vectors into a matrix
$\mathbf{U}$, one mode per column. It normalizes those columns with the
capacitance metric:

$$
\mathbf{U}^T\mathbf{C}\mathbf{U}=\mathbb{1},
\qquad
\mathbf{U}^T\mathbf{L}^{-1}\mathbf{U}=\mathbf{\Omega}^2.
$$

The diagonal entries of $\mathbf{\Omega}^2$ are $\omega_k^2$. Define the normal
coordinates $\eta$ by

$$
\Phi=\mathbf{U}\eta.
$$

These coordinates produce decoupled harmonic oscillators.

## Quantization

The normal coordinate operators are

$$
\begin{aligned}
\hat{\eta}_k &=
\sqrt{\frac{\hbar}{2\omega_k}}
(\hat{a}_k^\dagger+\hat{a}_k),
\\
\hat{\pi}_k &=
i\sqrt{\frac{\hbar\omega_k}{2}}
(\hat{a}_k^\dagger-\hat{a}_k).
\end{aligned}
$$

The linear Hamiltonian is

$$
\hat{H}_{\mathrm{lin}} =
\sum_k \hbar\omega_k
\left(\hat{a}_k^\dagger\hat{a}_k+\frac{1}{2}\right).
$$

`BBQ.hamiltonian_linear()` reports this harmonic energy in GHz and includes
the zero-point offset. For transition frequencies, subtracting the ground-state
energy removes the common offset.

## Branch Phase ZPF

The node flux operator is

$$
\hat{\Phi}_j =
\sum_k U_{jk}
\sqrt{\frac{\hbar}{2\omega_k}}
(\hat{a}_k^\dagger+\hat{a}_k).
$$

For a nonlinear branch between `node_a` and `node_b`, SCCircuits uses branch
phase $\Phi_b - \Phi_a$. The dimensionless phase is

$$
\hat{\varphi}_{ab} =
\frac{\hat{\Phi}_{b}-\hat{\Phi}_{a}}{\varphi_0},
\qquad
\varphi_0=\frac{\hbar}{2e}.
$$

The phase zero-point fluctuation of mode $k$ is therefore

$$
\varphi_{\mathrm{zpf}}^{ab,k} =
\frac{U_{b k}-U_{a k}}{\varphi_0}
\sqrt{\frac{\hbar}{2\omega_k}}.
$$

Reversing the branch direction flips the sign of every
$\varphi_{\mathrm{zpf}}$ value.

For multiple nonlinear branches, collect the branch directions into a matrix
$\mathbf{B}$. Each row is one branch. For branch `(node_a, node_b)`,
$B_{r,node_a}=-1$ and $B_{r,node_b}=1$. For branch `(node,)`, the row contains
$B_{r,node}=1$. Then the branch-by-mode zero-point fluctuation matrix is

$$
\varphi_{\mathrm{zpf}}^{r,k} =
\frac{(\mathbf{B}\mathbf{U})_{r,k}}{\varphi_0}
\sqrt{\frac{\hbar}{2\omega_k}}.
$$

In `BBQ`, this matrix is available as `branch_phase_zpfs` with shape
`(number_of_branches, number_of_modes)`. For a single nonlinear branch, the
first axis has length one, so `branch_phase_zpfs[0, k]` is the phase ZPF of
mode `k` on that branch.

## Nonlinear Hamiltonian

For each nonlinear branch $r$, `BBQ.hamiltonian_nonlinear()` uses the branch
phase operator $\hat{\varphi}_r$ built from the selected modes. It returns
energies in GHz with the convention

$$
\hat{H}_{\mathrm{nl}} =
-\sum_r E_{J,r}
\left[
s_r \cos(\hat{\varphi}_r+\varphi_{\mathrm{ext},r})
\,+\,
\frac{1}{2}\hat{\varphi}_r^2
\right].
$$

Here $s_r$ is the Josephson-energy suppression factor from modes omitted from
the truncated Hilbert space. The quadratic term avoids double-counting the
linearized Josephson inductance already included in the supplied
$\mathbf{L}^{-1}$ matrix.

The values passed as `external_phases` are per-branch, gauge-fixed phase
offsets in radians. They are ordered like `branch_phase_zpfs`. They are not,
by themselves, guaranteed to be independent physical loop fluxes. The number
of independent external fluxes is determined by circuit topology and belongs
to the graph or lumped-circuit layer that prepares the matrices and branch
offsets before calling `BBQ`.

## Units

- `angular_frequencies` stores angular frequencies $\omega_k$ in rad/s.
- `frequencies_ghz` stores $\omega_k/(2\pi)$ in GHz.
- `branch_phase_zpfs` is dimensionless.
- `hamiltonian_linear()` and `hamiltonian_nonlinear()` return dense Hamiltonian
  matrices in GHz.

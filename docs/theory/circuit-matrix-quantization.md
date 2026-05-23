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

## Linear Circuit

Use node fluxes $\Phi$. The linearized Lagrangian is

$$
\mathcal{L}
=
\frac{1}{2}\dot{\Phi}^T
\mathbf{C}
\dot{\Phi}
-
\frac{1}{2}\Phi^T
\mathbf{L}^{-1}
\Phi.
$$

The equations of motion are

$$
\mathbf{C}\ddot{\Phi}
+
\mathbf{L}^{-1}\Phi
=0.
$$

With the normal-mode ansatz
$\Phi(t)=\mathbf{v}_k e^{i\omega_k t}$, the modes solve

$$
\mathbf{L}^{-1}\mathbf{v}_k
=
\omega_k^2 \mathbf{C}\mathbf{v}_k.
$$

This generalized eigenvalue problem is the main calculation in `BBQ`.

## Normalization

Each positive-frequency solution of the generalized eigenvalue problem gives
one normal-mode vector $\mathbf{v}_k$. After removing null capacitance
directions and zero-frequency modes, `BBQ` collects the remaining physical mode
vectors into a matrix $\mathbf{U}$, one mode per column. It normalizes those
columns with the capacitance metric:

$$
\mathbf{U}^T\mathbf{C}\mathbf{U}=\mathbb{1},
\qquad
\mathbf{U}^T\mathbf{L}^{-1}\mathbf{U}=\mathbf{\Omega}^2.
$$

The diagonal entries of $\mathbf{\Omega}^2$ are $\omega_k^2$. The normal
coordinates $\eta$, defined by

$$
\Phi=\mathbf{U}\eta,
$$

then produce decoupled harmonic oscillators.

If the capacitance matrix has null or numerically tiny directions, `BBQ`
projects them out before solving the eigenproblem. Concretely, it diagonalizes
`C` and keeps only capacitance eigenvalues larger than

$$
10^{-12}\max_i |\lambda_i(C)|.
$$

Eigenvalues below this threshold are treated as zero for the purpose of the
normal-mode calculation. This keeps the calculation on the physical
capacitance subspace while avoiding numerical divisions by nearly zero
capacitances.

## Quantization

The normal coordinate operators are

$$
\hat{\eta}_k
=
\sqrt{\frac{\hbar}{2\omega_k}}
(\hat{a}_k^\dagger+\hat{a}_k),
\qquad
\hat{\pi}_k
=
i\sqrt{\frac{\hbar\omega_k}{2}}
(\hat{a}_k^\dagger-\hat{a}_k).
$$

The linear Hamiltonian is

$$
\hat{H}_{\mathrm{lin}}
=
\sum_k \hbar\omega_k
\left(\hat{a}_k^\dagger\hat{a}_k+\frac{1}{2}\right).
$$

`BBQ.hamiltonian_linear()` reports this harmonic energy in GHz and includes
the zero-point offset. For transition frequencies, subtracting the ground-state
energy removes the common offset.

## Branch Phase ZPF

The node flux operator is

$$
\hat{\Phi}_j
=
\sum_k U_{jk}
\sqrt{\frac{\hbar}{2\omega_k}}
(\hat{a}_k^\dagger+\hat{a}_k).
$$

For a nonlinear branch between `node_a` and `node_b`, SCCircuits uses branch
phase $\Phi_b - \Phi_a$. The dimensionless phase is

$$
\hat{\varphi}_{ab}
=
\frac{\hat{\Phi}_{b}-\hat{\Phi}_{a}}{\varphi_0},
\qquad
\varphi_0=\frac{\hbar}{2e}.
$$

The phase zero-point fluctuation of mode $k$ is therefore

$$
\varphi_{\mathrm{zpf}}^{ab,k}
=
\frac{U_{b k}-U_{a k}}{\varphi_0}
\sqrt{\frac{\hbar}{2\omega_k}}.
$$

Reversing the branch direction flips the sign of every
$\varphi_{\mathrm{zpf}}$ value and leaves the frequencies unchanged.

For multiple nonlinear branches, collect the branch directions into a matrix
$\mathbf{B}$. Each row is one branch. For branch `(node_a, node_b)`,
$B_{r,node_a}=-1$ and $B_{r,node_b}=1$. For branch `(node,)`, the row contains
$B_{r,node}=1$. Then the branch-by-mode zero-point fluctuation matrix is

$$
\varphi_{\mathrm{zpf}}^{r,k}
=
\frac{(\mathbf{B}\mathbf{U})_{r,k}}{\varphi_0}
\sqrt{\frac{\hbar}{2\omega_k}}.
$$

In `BBQ`, this matrix is available as `branch_phase_zpfs` with shape
`(number_of_branches, number_of_modes)`. For a single nonlinear branch, the
first axis has length one, so `branch_phase_zpfs[0, k]` is the phase ZPF of
mode `k` on that branch.

## Units

- `angular_frequencies` stores angular frequencies $\omega_k$ in rad/s.
- `frequencies_ghz` stores $\omega_k/(2\pi)$ in GHz.
- `branch_phase_zpfs` is dimensionless.
- `hamiltonian_linear()` and `hamiltonian_nonlinear()` return dense Hamiltonian
  matrices in GHz.

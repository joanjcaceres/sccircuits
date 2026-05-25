# Graph-To-Matrix Quantization

This page describes the mathematical layer that prepares the reduced matrices
and branch data consumed by `sccircuits.BBQ`. It is guidance for a future graph
or lumped-circuit implementation. It does not describe additional behavior that
`BBQ` performs today.

The graph layer starts from a circuit drawing or netlist and produces the
matrix-to-modes input contract:

- a reduced `capacitance_matrix`;
- a reduced `inverse_inductance_matrix`;
- Josephson junction records ordered by nonlinear branch;
- per-junction external phase offsets in the same row order.

`BBQ` then solves the generalized eigenproblem, computes normal-mode vectors,
computes branch phase zero-point fluctuations, and assembles Hamiltonian
matrices from those modal quantities.

## Branch Incidence

After choosing a ground node, collect the remaining node fluxes in
$\Phi \in \mathbb{R}^n$. Each branch is assigned an orientation. For branch
$r$, let $p$ be the positive node and $m$ be the negative node. The branch
incidence row is

$$
B_{rj} =
\begin{cases}
+1, & j=p,\\
-1, & j=m,\\
0, & \text{otherwise}.
\end{cases}
$$

If one end of the branch is ground, the grounded endpoint simply has no column
in the reduced node-flux vector. The oriented branch flux is

$$
\phi_r=(B\Phi)_r=\Phi_p-\Phi_m.
$$

Separate element classes should use separate incidence matrices:

- $B_C$ for capacitive branches;
- $B_L$ for linear inductive branches;
- $B_J$ for Josephson branches.

The row orientation fixes the sign convention for branch phases and external
phase offsets. It does not change observable spectra when the rest of the
description is transformed consistently.

## Matrix Construction

Let $C_b$ be the diagonal matrix of branch capacitances. The node capacitance
matrix is

$$
C = B_C^T C_b B_C.
$$

Let $Y_L$ be the diagonal matrix of inverse linear inductances. With oriented
external branch flux offsets $\phi_{L,\mathrm{ext}}$, the linear inductive
energy is

$$
U_L =
\frac{1}{2}
(B_L\Phi-\phi_{L,\mathrm{ext}})^T
Y_L
(B_L\Phi-\phi_{L,\mathrm{ext}}).
$$

Expanding this expression gives the linear matrix and source term

$$
L_{\mathrm{inv}} = B_L^T Y_L B_L,
\qquad
I_{\mathrm{ext}} = B_L^T Y_L \phi_{L,\mathrm{ext}}.
$$

The corresponding linear equations of motion have the form

$$
C\ddot{\Phi}+L_{\mathrm{inv}}\Phi-I_{\mathrm{ext}}=0.
$$

For a Josephson junction with energy $E_J$, the linearized Josephson inductance
is

$$
L_J=\frac{\varphi_0^2}{E_J},
\qquad
\varphi_0=\frac{\hbar}{2e}.
$$

If the graph layer includes this linearized junction contribution in
`inverse_inductance_matrix`, the nonlinear Hamiltonian must subtract the
quadratic part already counted by the linear model. This is the convention used
by `BBQ.hamiltonian_nonlinear()`.

## External Flux Matrix

Independent external flux coordinates belong to the graph layer. Write those
coordinates as $f \in \mathbb{R}^{n_f}$. A gauge choice maps them to branch
phase offsets through a matrix $S$:

$$
\theta_{J,\mathrm{ext}} = S f.
$$

The Josephson potential then has the form

$$
U_J =
-\sum_r E_{J,r}
\cos\left(
\frac{(B_J\Phi)_r}{\varphi_0}
+(\theta_{J,\mathrm{ext}})_r
\right).
$$

The matrix $S$ is gauge dependent. Different choices can describe the same
physical circuit if the branch orientations and phase offsets are transformed
consistently.

`BBQ.hamiltonian_nonlinear()` receives the gauge-fixed branch offsets
`external_phases`, ordered like the nonlinear branch rows. These values are not
guaranteed to be independent physical loop fluxes. The number of independent
fluxes is a graph-topology question outside `BBQ`.

## Variable Classes

The graph layer must classify variables before handing matrices to `BBQ`.

- Frozen variables are algebraically constrained variables that should be
  eliminated before modal analysis.
- Free variables are cyclic variables absent from the potential. Their
  conjugate charges are conserved, so a charge sector must be selected or
  carried into a Routhian reduction.
- Periodic variables are compact phase variables, typically associated with
  Josephson degrees of freedom and charge-like bases.
- Extended variables are real-valued flux variables, typically associated with
  oscillator-like modes.

The null-capacitance projection inside `BBQ` is only a numerical safeguard for
the supplied matrices. It is not a substitute for graph-level variable
classification and reduction.

## Frozen Variable Elimination

For exact linear constraints

$$
A\Phi=b,
$$

choose coordinates $q$ that satisfy the constraint by writing

$$
\Phi=\Phi_0+Tq.
$$

The reduced quadratic matrices are projected as

$$
C_q=T^T C T,
\qquad
L_{\mathrm{inv},q}=T^T L_{\mathrm{inv}}T.
$$

For algebraic variables that enter the potential but have no dynamics, split
the variables into kept coordinates $x$ and eliminated coordinates $y$. A
quadratic potential can be written as

$$
U(x,y)=
\frac{1}{2}
\begin{bmatrix}
x\\
y
\end{bmatrix}^T
\begin{bmatrix}
K_{xx} & K_{xy}\\
K_{yx} & K_{yy}
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}
-\begin{bmatrix}
I_x\\
I_y
\end{bmatrix}^T
\begin{bmatrix}
x\\
y
\end{bmatrix}.
$$

When $K_{yy}$ is invertible, minimizing over $y$ gives the Schur-complement
reduction

$$
K_{\mathrm{eff}}=K_{xx}-K_{xy}K_{yy}^{-1}K_{yx},
\qquad
I_{\mathrm{eff}}=I_x-K_{xy}K_{yy}^{-1}I_y.
$$

The graph layer should apply this reduction before constructing the final
`inverse_inductance_matrix` passed to `BBQ`.

## Free Variable Reduction

For a free coordinate $y$, the Lagrangian does not depend on $y$ itself. Its
conjugate charge

$$
Q_y=\frac{\partial \mathcal{L}}{\partial \dot{y}}
$$

is conserved. The graph layer should choose a conserved-charge sector or keep
the charge as a parameter in a Routhian reduction.

For a purely capacitive block with dynamic coordinates $x$ and free coordinate
$y$, eliminating $\dot{y}$ in a fixed charge sector produces the effective
capacitance block

$$
C_{\mathrm{eff}}=C_{xx}-C_{xy}C_{yy}^{-1}C_{yx}.
$$

Charge-sector constants and bias terms belong to the graph-layer Hamiltonian
construction. They should not appear as extra oscillator modes in `BBQ`.

## Output To BBQ

After incidence construction, gauge choice, variable classification, and
reduction, the graph layer hands `BBQ` a reduced matrix problem:

- `capacitance_matrix` is the final reduced capacitance matrix in SI units.
- `inverse_inductance_matrix` is the final reduced inverse-inductance matrix in
  SI units and in the same coordinate basis.
- `junctions` has one record per Josephson branch whose phase zero-point
  fluctuation is needed.
- `external_phases` has one gauge-fixed phase offset per Josephson branch, in
  the same row order as `junctions`.

For node-coordinate reductions, each junction record can use
`phase_positive_index` and `phase_negative_index` to identify the branch
orientation in the reduced coordinate basis. If a graph reduction changes to a
coordinate basis that no longer corresponds to literal node indices, the graph
layer must provide the equivalent reduced branch incidence information instead
of labeling transformed coordinates as nodes.

`BBQ` does not receive $B_C$, $B_L$, $B_J$, $S$, the original graph nodes, loop
bases, or variable classifications. Its contract starts from the reduced
matrices and nonlinear branch rows.

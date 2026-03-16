# Star and Chain Modeling Background

This page summarizes the modeling ideas behind SCCircuits. It is adapted from
the chapter `3-sccircuits.tex` in the companion thesis source and rewritten here
for package users rather than as a thesis derivation.

## Starting Point: Harmonic Modes Plus Josephson Nonlinearity

After linearizing a superconducting circuit, the harmonic part of the
Hamiltonian can be written as

$$
H_{\mathrm{lin}} = \sum_{k=1}^{M} \hbar \omega_k a_k^\dagger a_k.
$$

For a Josephson junction, the phase operator entering the cosine is a weighted
sum of those normal-mode coordinates:

$$
\phi = \sum_{k=1}^{M} \phi_{\mathrm{zpf},k} (a_k + a_k^\dagger).
$$

That is the key observation behind the package design: the cosine does not care
about every linear mode independently. It only depends on a small number of
collective directions in mode space.

## Collective Nonlinear Coordinate

For a single junction, define a collective annihilation operator aligned with
the phase zero-point fluctuation vector:

$$
b = \frac{1}{\lVert \phi_{\mathrm{zpf}} \rVert}
\sum_{k=1}^{M} \phi_{\mathrm{zpf},k} a_k.
$$

This isolates one nonlinear coordinate while the remaining orthogonal
combinations stay harmonic. In SCCircuits:

- `Circuit.from_harmonic_modes()` starts from the harmonic frequencies and phase
  zero-point fluctuations
- a Lanczos transform turns that problem into the chain representation used by
  `Circuit`

## Star Representation

In the star view, the nonlinear collective mode couples directly to every
remaining linear mode:

$$
H = H_b + \sum_k \hbar \omega_k' d_k^\dagger d_k
    + \sum_k \hbar g_k (b^\dagger d_k + b d_k^\dagger).
$$

This basis is usually the clearest one for physical interpretation because each
mode remains an independent oscillator coupled to the nonlinear core.

SCCircuits exposes this view through:

```python
star = circuit.star_representation()
```

## Chain Representation

The chain view tridiagonalizes the linear sector:

$$
H = H_b
  + \hbar t_1 (b^\dagger c_1 + b c_1^\dagger)
  + \sum_k \hbar \omega_k'' c_k^\dagger c_k
  + \sum_k \hbar t_{k+1} (c_k^\dagger c_{k+1} + c_k c_{k+1}^\dagger).
$$

This is the representation used internally by `Circuit` because it is well
suited to iterative diagonalization. Only nearest neighbors couple, so the
Hamiltonian can be built one mode at a time.

## Iterative Solution

`IterativeHamiltonianDiagonalizer` follows the chain structure directly:

1. diagonalize the nonlinear core
2. attach the first chain site
3. truncate to the low-energy subspace
4. continue site by site

This is why the `Circuit.eigensystem()` interface asks for a truncation schedule.
The method keeps the lowest-energy subspace after each stage instead of forming
the full tensor-product space at once.

## Two-Junction Extension

The same logic extends to circuits with a small number of nonlinear elements.
With two junctions, the nonlinear core becomes two collective coordinates rather
than one. The remaining linear sector can still be organized as either:

- a star-like bath coupled to that nonlinear core, or
- a chain or block-chain structure for iterative treatment

SCCircuits currently exposes the single-core chain construction in its public
API, but the theoretical picture remains useful when interpreting multimode
devices and deciding which modes to retain.

## Why This Matters for the Package

The package connects three layers that often appear separately in practice:

- circuit matrices from `BBQ` or the designer app
- harmonic-mode data from linear analysis
- low-energy nonlinear spectra used in fitting

The star/chain mapping is what makes those layers compatible inside one
workflow.

#!/usr/bin/env python3
"""
Example script demonstrating the basic usage of SCCircuits package.

This script shows how to:
1. Create a simple superconducting circuit
2. Calculate its eigenenergies
3. Show basic analysis capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from sccircuits import Circuit, get_info

def main():
    """Run the example demonstrating SCCircuits functionality."""
    
    # Print package information
    print("=" * 60)
    print("SCCircuits - Example Usage")
    print("=" * 60)
    info = get_info()
    print(f"Package: {info['name']} v{info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print()
    
    # Example 1: Single-mode transmon-like circuit
    print("1. Single-mode circuit analysis:")
    print("-" * 30)
    
    frequencies = [5.2]      # GHz - single mode frequency
    phase_zpf = [0.15]       # radians - phase zero-point fluctuation
    dimensions = [50]        # Hilbert space truncation
    Ej = 1.2                 # GHz - Josephson energy
    
    # Create circuit
    circuit = Circuit(
        frequencies=frequencies,
        phase_zpf=phase_zpf, 
        dimensions=dimensions,
        Ej=Ej,
        phase_ext=0.0
    )
    
    print(f"Circuit modes: {circuit.modes}")
    print(f"Collective frequency: {circuit.collective_frequency:.3f} GHz")
    print(f"Nonlinear frequency: {circuit.non_linear_frequency:.3f} GHz")
    print(f"Charging energy (Ec): {circuit.Ec:.3f} GHz")
    print(f"Inductive energy (El): {circuit.El:.3f} GHz")
    
    # Calculate eigenenergies
    truncation = 10
    evals, evecs = circuit.eigensystem(truncation=truncation)
    
    print(f"\nFirst {truncation} energy levels (GHz):")
    for i, energy in enumerate(evals):
        print(f"  Level {i}: {energy:.6f}")
    
    # Calculate transition frequencies
    transitions_01 = evals[1] - evals[0]
    transitions_12 = evals[2] - evals[1] 
    anharmonicity = transitions_12 - transitions_01
    
    print("\nTransition frequencies:")
    print(f"  |0⟩ → |1⟩: {transitions_01:.6f} GHz")
    print(f"  |1⟩ → |2⟩: {transitions_12:.6f} GHz") 
    print(f"  Anharmonicity: {anharmonicity:.6f} GHz")
    
    # Example 2: Multi-mode circuit
    print("\n" + "=" * 60)
    print("2. Multi-mode circuit analysis:")
    print("-" * 30)
    
    frequencies_multi = [5.0, 6.5, 7.8]  # Three modes
    phase_zpf_multi = [0.1, 0.15, 0.05]
    dimensions_multi = [30, 20, 15]
    Ej_multi = 0.8
    
    circuit_multi = Circuit(
        frequencies=frequencies_multi,
        phase_zpf=phase_zpf_multi,
        dimensions=dimensions_multi, 
        Ej=Ej_multi
    )
    
    print(f"Multi-mode circuit with {circuit_multi.modes} modes")
    print(f"Linear frequencies: {circuit_multi.linear_frequencies}")
    print(f"Linear coupling: {circuit_multi.linear_coupling}")
    
    # Example 3: External flux sweep
    print("\n" + "=" * 60)
    print("3. External flux dependence:")
    print("-" * 30)
    
    # Sweep external flux
    phi_ext_values = np.linspace(0, 2*np.pi, 50)
    transition_01_vs_flux = []
    
    for phi_ext in phi_ext_values:
        evals_flux, _ = circuit.eigensystem(truncation=5, phase_ext=phi_ext)
        transition_01_vs_flux.append(evals_flux[1] - evals_flux[0])
    
    transition_01_vs_flux = np.array(transition_01_vs_flux)
    
    print(f"Flux sweep completed over {len(phi_ext_values)} points")
    print(f"Transition frequency range: {transition_01_vs_flux.min():.3f} - {transition_01_vs_flux.max():.3f} GHz")
    print(f"Flux sensitivity: {(transition_01_vs_flux.max() - transition_01_vs_flux.min()):.3f} GHz")
    
    # Simple visualization (if matplotlib is available)
    try:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(truncation), evals, alpha=0.7, color='skyblue')
        plt.xlabel('Energy Level')
        plt.ylabel('Energy (GHz)')
        plt.title('Energy Spectrum')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(phi_ext_values/(2*np.pi), transition_01_vs_flux, 'r-', linewidth=2)
        plt.xlabel('External Flux (Φ/Φ₀)')
        plt.ylabel('|0⟩→|1⟩ Frequency (GHz)')
        plt.title('Flux Dependence')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("\nPlots generated successfully!")
        
    except Exception as e:
        print(f"\nNote: Plotting failed ({e}), but analysis completed successfully.")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
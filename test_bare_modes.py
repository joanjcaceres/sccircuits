#!/usr/bin/env python3
"""
Test script para verificar que el property bare_modes funciona correctamente
como la inversa de harmonic_modes_to_physical.
"""

import numpy as np
from sccircuits.circuit import Circuit

def test_bare_modes_reconstruction():
    """Prueba que bare_modes reconstruye correctamente los parámetros originales."""
    
    # Parámetros originales de prueba
    original_frequencies = np.array([5.0, 6.0, 7.8])
    original_phase_zpf = np.array([0.1, 0.2, 0.01])
    dimensions = [50, 10, 3]
    Ej = 1.0
    
    print("=== Test de reconstrucción de bare_modes ===")
    print(f"Frecuencias originales: {original_frequencies}")
    print(f"Phase ZPF originales: {original_phase_zpf}")
    
    # Crear el circuit usando from_harmonic_modes
    circuit = Circuit.from_harmonic_modes(
        frequencies=original_frequencies,
        phase_zpf=original_phase_zpf,
        dimensions=dimensions,
        Ej=Ej,
        phase_ext=0.0,
    )
    
    print(f"\nParámetros físicos resultantes:")
    print(f"Non-linear frequency: {circuit.non_linear_frequency:.6f} GHz")
    print(f"Non-linear phase ZPF: {circuit.non_linear_phase_zpf:.6f}")
    print(f"Linear frequencies: {circuit.linear_frequencies}")
    print(f"Linear couplings: {circuit.linear_coupling}")
    
    # Reconstruir los parámetros bare
    try:
        bare_modes = circuit.bare_modes
        reconstructed_frequencies = bare_modes['frequencies']
        reconstructed_phase_zpf = bare_modes['phase_zpf']
        
        print(f"\nParámetros reconstruidos:")
        print(f"Frecuencias reconstruidas: {reconstructed_frequencies}")
        print(f"Phase ZPF reconstruidos: {reconstructed_phase_zpf}")
        
        # Verificar que son equivalentes (dentro de la precisión numérica)
        freq_match = np.allclose(original_frequencies, reconstructed_frequencies, rtol=1e-10)
        zpf_match = np.allclose(original_phase_zpf, reconstructed_phase_zpf, rtol=1e-10)
        
        print(f"\n=== Resultados de la verificación ===")
        print(f"Frecuencias coinciden: {freq_match}")
        print(f"Phase ZPF coinciden: {zpf_match}")
        
        if freq_match and zpf_match:
            print("✅ SUCCESS: La reconstrucción es correcta!")
        else:
            print("❌ ERROR: Hay diferencias significativas")
            print(f"Diferencia máxima en frecuencias: {np.max(np.abs(original_frequencies - reconstructed_frequencies))}")
            print(f"Diferencia máxima en phase ZPF: {np.max(np.abs(original_phase_zpf - reconstructed_phase_zpf))}")
            
    except Exception as e:
        print(f"❌ ERROR al acceder a bare_modes: {e}")

def test_error_cases():
    """Prueba los casos de error cuando no se puede reconstruir."""
    
    print(f"\n=== Test de casos de error ===")
    
    # Crear un circuit de forma estándar (sin from_harmonic_modes)
    try:
        circuit = Circuit(
            non_linear_frequency=5.0,
            non_linear_phase_zpf=0.1,
            linear_frequencies=[6.0],
            linear_couplings=[0.1],
            dimensions=[50, 10],
            Ej=1.0
        )
        
        # Intentar acceder a bare_modes debería fallar
        bare_modes = circuit.bare_modes
        print("❌ ERROR: Debería haber fallado para Circuit creado sin from_harmonic_modes")
        
    except ValueError as e:
        print(f"✅ SUCCESS: Error esperado capturado correctamente: {e}")
    except Exception as e:
        print(f"❌ ERROR: Excepción inesperada: {e}")

if __name__ == "__main__":
    test_bare_modes_reconstruction()
    test_error_cases()
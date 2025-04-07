from quasiq import Circuit
import numpy as np


def grover_search(target: str):
    """Run Grover's search on **two qubits** to find `target` (a 2‑bit string).

    For a database of size N = 4, one Grover iteration is optimal.  The circuit
    implements:

    1.  Uniform superposition via Hadamards
    2.  Oracle that flips the phase of the marked state
    3.  Diffusion operator (inversion about the mean)

    Parameters
    ----------
    target : str
        Bit‑string to search for ("00", "01", "10" or "11").

    Returns
    -------
    np.ndarray
        Measurement results of shape (shots, 2) with qubit‑order `[q0, q1]`.
    """

    assert len(target) == 2 and set(target) <= {"0", "1"}, "target must be a 2‑bit binary string"

    n = 2
    circuit = Circuit(n, n)

    # ── 1. Initialise |++> ──
    for q in range(n):
        circuit.h(q)

    # ── Grover iteration count (π/4 * √N) ──
    iterations = 1  # N = 4 → 1 iteration

    for _ in range(iterations):
        # --- Oracle: phase flip |target⟩ ---
        # Map |target⟩ → |11⟩, apply CZ, then uncompute
        for q, bit in enumerate(target):
            if bit == "0":
                circuit.x(q)
        circuit.cz(0, 1)
        for q, bit in enumerate(target):
            if bit == "0":
                circuit.x(q)

        # --- Diffusion operator (inversion about mean) ---
        for q in range(n):
            circuit.h(q)
            circuit.x(q)
        circuit.cz(0, 1)  # phase‑flip |00⟩
        for q in range(n):
            circuit.x(q)
            circuit.x(q)

    # ── Measurement ──
    for q in range(n):
        circuit.measure(q, q)

    circuit.print_circuit()
    return circuit.execute(shots=1024, visualize=True)


if __name__ == "__main__":
    target_state = "11"  # change as desired
    results = grover_search(target_state)

    # Simple accuracy check
    decimal_results = results[:, 0] * 2 + results[:, 1]
    target_decimal = int(target_state, 2)
    counts = np.bincount(decimal_results, minlength=4)
    success_prob = counts[target_decimal] / counts.sum()
    print(f"Success probability for |{target_state}⟩: {success_prob:.2%}")

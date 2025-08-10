# Scientific Papers Repository

A collection of original scientific papers exploring mathematical and physical phenomena through rigorous analysis and computational verification.

## Papers

### 1. Golden Ratio from Energy Minimization and Self-Similarity
**Location:** [`/golden_ratio/`](./golden_ratio/)

A complete mathematical framework demonstrating that the golden ratio φ = (1+√5)/2 emerges naturally from energy minimization in hierarchical braided configurations of physical structures like vortices, magnetic flux tubes, and optical beams.

**Key Contributions:**
- Rigorous proof that φ arises from fundamental physical constraints rather than being externally imposed
- Energy functional `E(x) = ½(x-1)² - ln(x)` with unique minimum at φ
- Quantitative robustness bounds and experimental predictions
- Connections to Fibonacci sequences, continued fractions, and optimal irrationality

**Repository Structure:**
- `/doc/` - LaTeX source for the academic paper
- `/calculations/` - Interactive demonstrations and visualizations
- `/derivations/` - Complete mathematical verification suite (40+ tests)

## Getting Started

Each paper is self-contained with its own documentation and code. Navigate to the specific paper directory for detailed instructions.

### General Prerequisites
```bash
pip install numpy scipy matplotlib sympy
```

## Author

Trevor Norris  
trev.norris@gmail.com

## License

This work is licensed under a Creative Commons Attribution 4.0 International License.

## Citation

If you use any of these papers in your research, please cite the specific paper. See individual paper directories for citation information.

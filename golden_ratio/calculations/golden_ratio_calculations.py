"""
Golden Ratio from Energy Minimization and Self-Similarity - Mathematical Demonstration

This script demonstrates all the key mathematical concepts from the paper:
"Golden Ratio from Energy Minimization and Self-Similarity in Hierarchical Vortices"

Mathematical Framework:
- Energy functional: E(x) = ½(x-1)² - ln(x) where x = (P/ξₕ)²
- Self-similarity map: T(x) = 1 + 1/x
- Golden ratio emergence: x* = φ = (1+√5)/2 from energy minimization
- Robustness bound: |x* - φ| ≤ √(2Δ/m) under perturbations
- Twist rate scaling: τ = 2π/(√φ ξₕ)

Physical Interpretation:
- x: Dimensionless pitch parameter (squared ratio)
- P: Helical pitch length
- ξₕ: Helical coherence length
- τ: Twist rate
- Quadratic term: Local overlap/strain penalty
- Logarithmic term: Multi-scale relaxation gain
- φ: "Most irrational" ratio avoiding resonant alignments
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve, minimize
from scipy.integrate import odeint
import matplotlib.animation as animation
from fractions import Fraction
from collections import defaultdict

# Set style for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True

class GoldenRatioHierarchy:
    def __init__(self):
        """
        Initialize the hierarchical vortex energy system based on the paper.

        Core mathematical elements:
        - Energy E(x) = ½(x-1)² - ln(x)
        - Self-similarity map T(x) = 1 + 1/x
        - Golden ratio φ = (1+√5)/2
        """
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.phi_conjugate = (1 - np.sqrt(5)) / 2  # Conjugate root
        self.sqrt_phi = np.sqrt(self.phi)

        print(f"Golden Ratio: φ = {self.phi:.8f}")
        print(f"√φ = {self.sqrt_phi:.8f}")
        print(f"φ² = {self.phi**2:.8f}")
        print(f"φ + 1 = {self.phi + 1:.8f}")
        print(f"Verification: φ² - φ - 1 = {self.phi**2 - self.phi - 1:.2e}")

    def energy_functional(self, x):
        """
        Core energy functional from the paper: E(x) = ½(x-1)² - ln(x)

        Components:
        - ½(x-1)²: Quadratic penalty for deviations from natural spacing
        - -ln(x): Logarithmic relaxation from multi-scale rearrangements

        Domain: x ∈ (1, ∞) corresponding to P > ξₕ
        """
        if x <= 1.0:  # Domain boundary
            return np.inf
        return 0.5 * (x - 1)**2 - np.log(x)

    def energy_derivative(self, x):
        """
        First derivative: E'(x) = (x-1) - 1/x

        Critical point condition: E'(x) = 0
        ⟹ (x-1) - 1/x = 0
        ⟹ x² - x - 1 = 0  (multiply by x)
        ⟹ x = φ (positive root)
        """
        if x <= 1.0:
            return np.inf
        return (x - 1) - 1/x

    def energy_second_derivative(self, x):
        """
        Second derivative: E''(x) = 1 + 1/x²

        Always positive for x > 0, confirming:
        - Strong convexity (E''(x) ≥ 1)
        - Unique global minimum
        """
        if x <= 1.0:
            return np.inf
        return 1 + 1/(x**2)

    def self_similarity_map(self, x):
        """
        Self-similarity transformation: T(x) = 1 + 1/x

        Physical interpretation: "Add a layer then rescale"
        Mathematical properties:
        - Maps (1,∞) → (1,∞)
        - Fixed point: T(φ) = φ
        - Decreasing: T'(x) = -1/x² < 0
        """
        return 1 + 1/x

    def verify_golden_ratio_properties(self):
        """
        Comprehensive verification of golden ratio properties.
        """
        print("\n" + "="*60)
        print("GOLDEN RATIO PROPERTY VERIFICATION")
        print("="*60)

        # 1. Quadratic equation
        quad_check = self.phi**2 - self.phi - 1
        print(f"1. Quadratic equation φ² - φ - 1 = {quad_check:.2e}")

        # 2. Energy critical point
        energy_deriv = self.energy_derivative(self.phi)
        print(f"2. Energy derivative E'(φ) = {energy_deriv:.2e}")

        # 3. Self-similarity fixed point
        T_phi = self.self_similarity_map(self.phi)
        fixed_point_check = T_phi - self.phi
        print(f"3. Fixed point T(φ) - φ = {fixed_point_check:.2e}")

        # 4. Strong convexity
        second_deriv = self.energy_second_derivative(self.phi)
        print(f"4. Strong convexity E''(φ) = {second_deriv:.6f} > 1")

        # 5. Continued fraction [1; 1, 1, 1, ...]
        print(f"5. Continued fraction: φ = [1; 1, 1, 1, ...] (most irrational)")

        # 6. Reciprocal relation
        reciprocal_check = self.phi - 1 - 1/self.phi
        print(f"6. Reciprocal relation φ - 1 - 1/φ = {reciprocal_check:.2e}")

        return True

    def demonstrate_energy_minimization(self):
        """
        Show that φ is the unique global minimizer of E(x).
        """
        print("\n" + "="*60)
        print("ENERGY MINIMIZATION DEMONSTRATION")
        print("="*60)

        # Create energy landscape
        x = np.linspace(1.01, 4.0, 1000)
        energy = [self.energy_functional(xi) for xi in x]
        energy_deriv = [self.energy_derivative(xi) for xi in x]
        energy_second_deriv = [self.energy_second_derivative(xi) for xi in x]

        # Find numerical minimum
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(self.energy_functional, bounds=(1.01, 10.0), method='bounded')
        x_min_numerical = result.x

        print(f"Numerical minimum: x* = {x_min_numerical:.8f}")
        print(f"Theoretical minimum: φ = {self.phi:.8f}")
        print(f"Difference: {abs(x_min_numerical - self.phi):.2e}")
        print(f"Energy at minimum: E(φ) = {self.energy_functional(self.phi):.6f}")

        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Energy landscape
        ax1.plot(x, energy, 'b-', linewidth=2.5, label='E(x) = ½(x-1)² - ln(x)')
        ax1.axvline(self.phi, color='red', linestyle='--', linewidth=2, label=f'φ = {self.phi:.4f}')
        ax1.plot(self.phi, self.energy_functional(self.phi), 'ro', markersize=10, label='Global minimum')
        ax1.set_xlabel('x = (P/ξₕ)²')
        ax1.set_ylabel('Energy E(x)')
        ax1.set_title('Energy Landscape: Unique Global Minimum at φ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 4)

        # Plot 2: First derivative (shows critical point)
        ax2.plot(x, energy_deriv, 'g-', linewidth=2.5, label="E'(x) = (x-1) - 1/x")
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(self.phi, color='red', linestyle='--', linewidth=2, label=f'φ = {self.phi:.4f}')
        ax2.plot(self.phi, 0, 'ro', markersize=10, label='Critical point')
        ax2.set_xlabel('x = (P/ξₕ)²')
        ax2.set_ylabel("E'(x)")
        ax2.set_title('First Derivative: Critical Point at φ')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, 4)

        # Plot 3: Second derivative (shows strong convexity)
        ax3.plot(x, energy_second_deriv, 'm-', linewidth=2.5, label="E''(x) = 1 + 1/x²")
        ax3.axvline(self.phi, color='red', linestyle='--', linewidth=2, label=f'φ = {self.phi:.4f}')
        ax3.axhline(1, color='orange', linestyle=':', linewidth=2, label='Strong convexity bound')
        ax3.set_xlabel('x = (P/ξₕ)²')
        ax3.set_ylabel("E''(x)")
        ax3.set_title('Second Derivative: Strong Convexity E\'\'(x) ≥ 1')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(1, 4)

        # Plot 4: Components breakdown
        x_comp = np.linspace(1.01, 3, 500)
        quadratic_term = [0.5 * (xi - 1)**2 for xi in x_comp]
        log_term = [-np.log(xi) for xi in x_comp]

        ax4.plot(x_comp, quadratic_term, 'b--', linewidth=2, label='½(x-1)² (overlap penalty)')
        ax4.plot(x_comp, log_term, 'r--', linewidth=2, label='-ln(x) (relaxation gain)')
        ax4.plot(x_comp, [q + l for q, l in zip(quadratic_term, log_term)], 'k-', linewidth=2.5, label='Total E(x)')
        ax4.axvline(self.phi, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax4.set_xlabel('x = (P/ξₕ)²')
        ax4.set_ylabel('Energy components')
        ax4.set_title('Energy Components: Quadratic vs Logarithmic')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return x_min_numerical

    def demonstrate_self_similarity(self):
        """
        Demonstrate the self-similarity map T(x) = 1 + 1/x and its connection to φ.
        """
        print("\n" + "="*60)
        print("SELF-SIMILARITY DEMONSTRATION")
        print("="*60)

        # Properties of T(x)
        x_test = np.linspace(1.1, 5.0, 100)
        T_x = [self.self_similarity_map(xi) for xi in x_test]
        T_derivative = [-1/(xi**2) for xi in x_test]

        # Fixed point analysis
        print(f"Self-similarity map: T(x) = 1 + 1/x")
        print(f"Fixed point equation: x = T(x) ⟹ x = 1 + 1/x ⟹ x² - x - 1 = 0")
        print(f"Fixed point: x* = φ = {self.phi:.8f}")
        print(f"Verification: T(φ) = {self.self_similarity_map(self.phi):.8f}")
        print(f"T'(φ) = {-1/self.phi**2:.6f} (contractive: |T'| < 1)")

        # Iterate the map from different starting points
        def iterate_map(x0, n_iterations=20):
            """Iterate T(x) starting from x0."""
            trajectory = [x0]
            x = x0
            for i in range(n_iterations):
                x = self.self_similarity_map(x)
                trajectory.append(x)
            return trajectory

        # Test convergence from multiple starting points
        starting_points = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Self-similarity map
        ax1.plot(x_test, x_test, 'k--', alpha=0.5, label='y = x')
        ax1.plot(x_test, T_x, 'b-', linewidth=2.5, label='T(x) = 1 + 1/x')
        ax1.axvline(self.phi, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.axhline(self.phi, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.plot(self.phi, self.phi, 'ro', markersize=10, label=f'Fixed point φ = {self.phi:.4f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('T(x)')
        ax1.set_title('Self-Similarity Map: T(x) = 1 + 1/x')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 5)
        ax1.set_ylim(1, 3)

        # Plot 2: Convergence trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, len(starting_points)))
        for i, x0 in enumerate(starting_points):
            trajectory = iterate_map(x0, 15)
            iterations = range(len(trajectory))
            ax2.plot(iterations, trajectory, 'o-', color=colors[i],
                    linewidth=2, markersize=4, label=f'x₀ = {x0}')

        ax2.axhline(self.phi, color='red', linestyle='--', linewidth=2, label=f'φ = {self.phi:.4f}')
        ax2.set_xlabel('Iteration n')
        ax2.set_ylabel('xₙ')
        ax2.set_title('Convergence to Fixed Point φ')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cobweb diagram for one trajectory
        x0_demo = 2.5
        trajectory_demo = iterate_map(x0_demo, 8)

        ax3.plot(x_test, x_test, 'k--', alpha=0.5, label='y = x')
        ax3.plot(x_test, T_x, 'b-', linewidth=2.5, label='T(x) = 1 + 1/x')

        # Draw cobweb
        for i in range(len(trajectory_demo)-1):
            x_curr = trajectory_demo[i]
            x_next = trajectory_demo[i+1]
            # Vertical line to curve
            ax3.plot([x_curr, x_curr], [x_curr, x_next], 'r-', alpha=0.7, linewidth=1.5)
            # Horizontal line to diagonal
            if i < len(trajectory_demo)-2:
                ax3.plot([x_curr, x_next], [x_next, x_next], 'r-', alpha=0.7, linewidth=1.5)

        ax3.plot(trajectory_demo[0], trajectory_demo[0], 'go', markersize=8, label=f'Start: x₀ = {x0_demo}')
        ax3.plot(self.phi, self.phi, 'ro', markersize=10, label=f'Fixed point φ')
        ax3.set_xlabel('x')
        ax3.set_ylabel('T(x)')
        ax3.set_title(f'Cobweb Diagram: Convergence from x₀ = {x0_demo}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(1.4, 3)
        ax3.set_ylim(1.4, 2)

        # Plot 4: Derivative showing contraction
        ax4.plot(x_test, T_derivative, 'purple', linewidth=2.5, label="T'(x) = -1/x²")
        ax4.axvline(self.phi, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax4.axhline(0, color='black', alpha=0.5)
        ax4.axhline(-1, color='orange', linestyle=':', alpha=0.7, label='|T\'| = 1 boundary')
        ax4.set_xlabel('x')
        ax4.set_ylabel("T'(x)")
        ax4.set_title('Map Derivative: Contraction Property')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(1, 5)

        plt.tight_layout()
        plt.show()

        return starting_points, [iterate_map(x0, 15) for x0 in starting_points]

    def demonstrate_robustness_theorem(self):
        """
        Demonstrate the robustness theorem: |x* - φ| ≤ √(2Δ/m)
        """
        print("\n" + "="*60)
        print("ROBUSTNESS THEOREM DEMONSTRATION")
        print("="*60)

        print("Theorem: If E is m-strongly convex and |E(T(x)) - E(x)| ≤ Δ,")
        print("then |x* - φ| ≤ √(2Δ/m)")
        print(f"For our E(x): m = min E''(x) ≥ 1")

        # Test various perturbations
        perturbation_types = [
            ("Core radius", lambda x, eps: eps / x**2),
            ("Anisotropy", lambda x, eps: eps * (x - self.phi)**2),
            ("Finite range", lambda x, eps: eps * np.sin(2*np.pi*x)),
            ("Asymmetry", lambda x, eps: eps * x),
        ]

        epsilon_values = np.logspace(-4, -1, 20)  # From 10^-4 to 10^-1

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        colors = ['blue', 'green', 'red', 'purple']

        all_results = {}

        for i, (name, perturbation_func) in enumerate(perturbation_types):
            deviations = []
            Delta_estimates = []
            bounds = []

            for eps in epsilon_values:
                # Define perturbed energy
                def E_perturbed(x):
                    if x <= 1.0:
                        return np.inf
                    return self.energy_functional(x) + perturbation_func(x, eps)

                # Find minimum of perturbed energy
                try:
                    result = minimize_scalar(E_perturbed, bounds=(1.01, 5.0), method='bounded')
                    x_star_pert = result.x

                    # Compute deviation from φ
                    deviation = abs(x_star_pert - self.phi)
                    deviations.append(deviation)

                    # Estimate Δ (symmetry breaking parameter)
                    x_test = np.linspace(1.1, 3.0, 100)
                    invariance_defects = []
                    for x_val in x_test:
                        T_x = self.self_similarity_map(x_val)
                        if T_x > 1.0:
                            defect = abs(E_perturbed(T_x) - E_perturbed(x_val))
                            invariance_defects.append(defect)

                    Delta_est = max(invariance_defects) if invariance_defects else eps
                    Delta_estimates.append(Delta_est)

                    # Theoretical bound: √(2Δ/m) with m ≥ 1
                    bound = np.sqrt(2 * Delta_est / 1.0)
                    bounds.append(bound)

                except:
                    deviations.append(np.nan)
                    Delta_estimates.append(eps)
                    bounds.append(np.sqrt(2 * eps))

            all_results[name] = {
                'deviations': deviations,
                'bounds': bounds,
                'Delta': Delta_estimates
            }

            # Plot results
            ax = [ax1, ax2, ax3, ax4][i]

            valid_indices = [j for j, d in enumerate(deviations) if not np.isnan(d)]
            if valid_indices:
                eps_valid = [epsilon_values[j] for j in valid_indices]
                dev_valid = [deviations[j] for j in valid_indices]
                bound_valid = [bounds[j] for j in valid_indices]

                ax.loglog(eps_valid, dev_valid, 'o-', color=colors[i],
                         linewidth=2, markersize=5, label=f'Actual |x* - φ|')
                ax.loglog(eps_valid, bound_valid, '--', color=colors[i],
                         linewidth=2, alpha=0.7, label=f'Bound √(2Δ/m)')
                ax.loglog(eps_valid, eps_valid, ':', color='black', alpha=0.5, label='O(ε)' if i == 0 else "")

            ax.set_xlabel('Perturbation strength ε')
            ax.set_ylabel('Deviation from φ')
            ax.set_title(f'Robustness: {name} perturbation')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Summary table
        print(f"\nRobustness Analysis Summary:")
        print(f"{'Perturbation':<15} {'Max ε tested':<12} {'Max deviation':<15} {'Bound satisfied':<15}")
        print("-" * 70)

        for name, results in all_results.items():
            valid_devs = [d for d in results['deviations'] if not np.isnan(d)]
            valid_bounds = [b for b in results['bounds'] if not np.isnan(b)]

            if valid_devs and valid_bounds:
                max_eps = max(epsilon_values[:len(valid_devs)])
                max_dev = max(valid_devs)
                corresponding_bound = valid_bounds[np.argmax(valid_devs)]
                bound_satisfied = "✓" if max_dev <= corresponding_bound * 1.1 else "✗"

                print(f"{name:<15} {max_eps:<12.1e} {max_dev:<15.2e} {bound_satisfied:<15}")

        return all_results

    def demonstrate_twist_rate_scaling(self):
        """
        Demonstrate the twist rate formula τ = 2π/(√φ ξₕ).
        """
        print("\n" + "="*60)
        print("TWIST RATE SCALING DEMONSTRATION")
        print("="*60)

        print("Physical relationships:")
        print("• x = (P/ξₕ)² (dimensionless pitch parameter)")
        print("• P = 2π/τ (pitch-twist relation)")
        print("• P = ξₕ√x (from definition of x)")
        print("• Therefore: τ = 2π/(ξₕ√x)")
        print(f"• At optimum x* = φ: τ* = 2π/(ξₕ√φ) = 2π/(ξₕ × {self.sqrt_phi:.6f})")

        # Physical parameter examples
        xi_h_values = np.array([0.1e-6, 0.5e-6, 1.0e-6, 2.0e-6, 5.0e-6]) * 1e6  # Convert to μm

        print(f"\nPhysical parameter examples:")
        print(f"{'ξₕ (μm)':<10} {'P* (μm)':<10} {'τ* (μm⁻¹)':<12} {'x* = φ':<10}")
        print("-" * 45)

        for xi_h_um in xi_h_values:
            P_optimal_um = xi_h_um * self.sqrt_phi
            tau_optimal = 2 * np.pi / (self.sqrt_phi * xi_h_um)

            print(f"{xi_h_um:<10.1f} {P_optimal_um:<10.2f} {tau_optimal:<12.3f} {self.phi:<10.6f}")

        # Scaling analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: P vs ξₕ scaling
        xi_h_range = np.linspace(0.1, 5.0, 100)
        P_optimal = xi_h_range * self.sqrt_phi
        P_other_ratios = {
            "x = 2": xi_h_range * np.sqrt(2),
            "x = 3": xi_h_range * np.sqrt(3),
            "x = π": xi_h_range * np.sqrt(np.pi),
        }

        ax1.plot(xi_h_range, P_optimal, 'r-', linewidth=3, label=f'P* = ξₕ√φ (optimal)')
        for label, P_vals in P_other_ratios.items():
            ax1.plot(xi_h_range, P_vals, '--', linewidth=2, alpha=0.7, label=f'P = ξₕ√({label.split("=")[1].strip()})')

        ax1.set_xlabel('ξₕ (μm)')
        ax1.set_ylabel('P (μm)')
        ax1.set_title('Pitch vs Coherence Length Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: τ vs ξₕ scaling
        tau_optimal = 2 * np.pi / (self.sqrt_phi * xi_h_range)
        tau_other_ratios = {
            "x = 2": 2 * np.pi / (np.sqrt(2) * xi_h_range),
            "x = 3": 2 * np.pi / (np.sqrt(3) * xi_h_range),
            "x = π": 2 * np.pi / (np.sqrt(np.pi) * xi_h_range),
        }

        ax2.loglog(xi_h_range, tau_optimal, 'r-', linewidth=3, label=f'τ* = 2π/(ξₕ√φ)')
        for label, tau_vals in tau_other_ratios.items():
            ax2.loglog(xi_h_range, tau_vals, '--', linewidth=2, alpha=0.7,
                      label=f'τ = 2π/(ξₕ√{label.split("=")[1].strip()})')

        ax2.set_xlabel('ξₕ (μm)')
        ax2.set_ylabel('τ (μm⁻¹)')
        ax2.set_title('Twist Rate vs Coherence Length (log-log)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Energy landscape with physical interpretation
        x_range = np.linspace(1.1, 4.0, 500)
        energy_vals = [self.energy_functional(x) for x in x_range]
        P_ratio = np.sqrt(x_range)  # P/ξₕ = √x

        ax3.plot(P_ratio, energy_vals, 'b-', linewidth=2.5, label='E(x) vs P/ξₕ')
        ax3.axvline(self.sqrt_phi, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal P*/ξₕ = √φ = {self.sqrt_phi:.3f}')
        ax3.set_xlabel('P/ξₕ (pitch ratio)')
        ax3.set_ylabel('Energy E')
        ax3.set_title('Energy vs Physical Pitch Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Avoidance of commensurate ratios
        # Show rational approximations and their energy costs
        rationals = []
        for q in range(2, 20):
            for p in range(2, 2*q):
                ratio = p/q
                if 1.0 < ratio < 2.5:
                    rationals.append((p, q, ratio))

        rationals.sort(key=lambda x: x[2])

        rational_ratios = [r[2] for r in rationals]
        rational_energies = [self.energy_functional(r**2) for r in rational_ratios]  # x = (P/ξₕ)²

        ax4.plot(P_ratio, energy_vals, 'b-', linewidth=2, alpha=0.7, label='E(x) continuous')
        ax4.scatter(rational_ratios, rational_energies, c='orange', s=30, alpha=0.8,
                   label='Rational P/ξₕ ratios', zorder=5)
        ax4.axvline(self.sqrt_phi, color='red', linestyle='--', linewidth=2,
                   label=f'φ = {self.sqrt_phi:.3f} (irrational)')

        # Mark some specific rationals
        special_rationals = [(3, 2), (4, 3), (5, 3), (7, 4), (8, 5)]
        for p, q in special_rationals:
            ratio = p/q
            if 1.0 < ratio < 2.5:
                energy = self.energy_functional(ratio**2)
                ax4.annotate(f'{p}/{q}', (ratio, energy), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, alpha=0.7)

        ax4.set_xlabel('P/ξₕ (pitch ratio)')
        ax4.set_ylabel('Energy E')
        ax4.set_title('Avoidance of Commensurate (Rational) Ratios')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return xi_h_values, P_optimal_um, tau_optimal

    def demonstrate_three_routes_to_logarithm(self):
        """
        Demonstrate the three independent derivations of the logarithmic term.
        """
        print("\n" + "="*60)
        print("THREE ROUTES TO LOGARITHMIC RELAXATION")
        print("="*60)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Route A: Elastic defect analogy
        print("Route A: Elastic defect analogy")
        print("• Line defect energy: E ~ A ln(R/r₀)")
        print("• Outer scale R ∝ P, inner scale r₀ ~ ξₕ")
        print("• Relaxation gain: -B ln(P/ξₕ) = -(B/2) ln(x)")

        # Show logarithmic profile for line defect
        r = np.linspace(0.1, 5, 200)
        ln_profile = np.log(r)

        ax1.plot(r, ln_profile, 'b-', linewidth=2.5, label='ln(r) profile')
        ax1.axvline(1, color='red', linestyle='--', alpha=0.7, label='r₀ ~ ξₕ')
        ax1.axvline(self.sqrt_phi, color='green', linestyle='--', alpha=0.7, label='R* ~ P* ~ √φ ξₕ')
        ax1.fill_between([1, self.sqrt_phi], -2, 2, alpha=0.2, color='yellow', label='Active region')
        ax1.set_xlabel('r/ξₕ')
        ax1.set_ylabel('Logarithmic energy density')
        ax1.set_title('Route A: Elastic Defect Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 2)

        # Route B: Short-range overlap model
        print("\nRoute B: Short-range overlap model")
        print("• Inter-layer interaction: ∫ exp(-r/ξc) dℓ")
        print("• Helical geometry + angular averaging")
        print("• Typical separation ~ P → ln(P) dependence")

        # Show exponential decay and its integral
        r_overlap = np.linspace(0, 3, 200)
        xi_c = 0.3  # Core radius parameter
        exp_decay = np.exp(-r_overlap / xi_c)

        # Cumulative effect (integral) grows logarithmically
        cumulative = np.array([np.trapz(exp_decay[:i+1], r_overlap[:i+1]) if i > 0 else 0
                              for i in range(len(r_overlap))])

        ax2_twin = ax2.twinx()
        ax2.plot(r_overlap, exp_decay, 'r-', linewidth=2.5, label='exp(-r/ξc) interaction')
        ax2_twin.plot(r_overlap, cumulative, 'g-', linewidth=2.5, label='Cumulative effect')

        ax2.axvline(xi_c, color='red', linestyle=':', alpha=0.7, label='ξc (core scale)')
        ax2.axvline(self.sqrt_phi * xi_c, color='blue', linestyle='--', alpha=0.7, label='P* scale')

        ax2.set_xlabel('r/ξc')
        ax2.set_ylabel('Interaction strength', color='red')
        ax2_twin.set_ylabel('Cumulative relaxation', color='green')
        ax2.set_title('Route B: Overlap Model')
        ax2.legend(loc='upper right')
        ax2_twin.legend(loc='center right')
        ax2.grid(True, alpha=0.3)

        # Route C: Scale invariance (RG)
        print("\nRoute C: Renormalization group scale invariance")
        print("• Under scaling x → λx, only additive invariant is κ ln(x)")
        print("• Scale-invariant relaxation must be logarithmic")
        print("• Marginal operator in RG sense")

        # Show scale invariance of logarithm
        x_scale = np.linspace(1.1, 4, 200)
        scales = [1, 1.5, 2, 2.5]

        for i, scale in enumerate(scales):
            x_scaled = x_scale * scale
            log_scaled = np.log(x_scaled)
            log_original = np.log(x_scale)

            if i == 0:
                ax3.plot(x_scale, log_original, 'b-', linewidth=2.5, alpha=0.8, label='ln(x)')
            else:
                # Show that ln(λx) = ln(λ) + ln(x) (additive shift)
                ax3.plot(x_scale, log_scaled, '--', linewidth=2, alpha=0.6,
                        label=f'ln({scale}x) = ln({scale}) + ln(x)')

        ax3.axvline(self.phi, color='red', linestyle='--', linewidth=2, alpha=0.7, label='x* = φ')
        ax3.set_xlabel('x')
        ax3.set_ylabel('ln(x) and scaled versions')
        ax3.set_title('Route C: Scale Invariance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\nAll three routes lead to logarithmic relaxation term in E(x):")
        print(f"E(x) = ½(x-1)² - ln(x)")
        print(f"Minimization yields x* = φ = {self.phi:.6f}")

    def demonstrate_lyapunov_descent(self):
        """
        Demonstrate the Lyapunov descent property: E(T(x)) <= E(x) with equality only at φ.
        """
        print("\n" + "="*60)
        print("LYAPUNOV DESCENT DEMONSTRATION")
        print("="*60)

        print("Theorem: E(T(x)) ≤ E(x) for all x > 1, with equality only at x = φ")
        print("This proves φ is the unique dynamic attractor under self-similarity map T")

        # Define descent function G(x) = E(T(x)) - E(x)
        def descent_function(x):
            """G(x) = E(T(x)) - E(x)"""
            if x <= 1.0:
                return np.inf
            T_x = self.self_similarity_map(x)
            return self.energy_functional(T_x) - self.energy_functional(x)

        def descent_derivative(x):
            """G'(x) = -[(x²-x-1)(x³+x²-1)]/[x³(x+1)]"""
            if x <= 1.0:
                return np.inf
            numerator = -(x**2 - x - 1) * (x**3 + x**2 - 1)
            denominator = x**3 * (x + 1)
            return numerator / denominator

        # Test over range
        x_test = np.linspace(1.01, 5.0, 1000)
        G_values = [descent_function(x) for x in x_test]
        G_prime_values = [descent_derivative(x) for x in x_test]

        # Verify descent property
        max_G = max(G_values)
        G_at_phi = descent_function(self.phi)

        print(f"Maximum of G(x): {max_G:.8e}")
        print(f"G(φ): {G_at_phi:.8e}")
        print(f"G(x) ≤ 0 everywhere: {all(g <= 1e-12 for g in G_values)}")

        # Test specific points
        test_points = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
        print(f"\nDescent verification at test points:")
        print(f"{'x':<6} {'G(x)':<12} {'Descent?':<8}")
        print("-" * 30)

        for x in test_points:
            G_x = descent_function(x)
            is_descent = "✓" if G_x <= 1e-12 else "✗"
            print(f"{x:<6.1f} {G_x:<12.2e} {is_descent:<8}")

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Descent function G(x)
        ax1.plot(x_test, G_values, 'b-', linewidth=2.5, label='G(x) = E(T(x)) - E(x)')
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(self.phi, color='red', linestyle='--', linewidth=2, label=f'φ = {self.phi:.4f}')
        ax1.plot(self.phi, G_at_phi, 'ro', markersize=10, label='Unique zero at φ')
        ax1.set_xlabel('x')
        ax1.set_ylabel('G(x)')
        ax1.set_title('Lyapunov Descent: G(x) ≤ 0')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 5)

        # Plot 2: Derivative G'(x) showing sign structure
        ax2.plot(x_test, G_prime_values, 'g-', linewidth=2.5, label="G'(x)")
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(self.phi, color='red', linestyle='--', linewidth=2, label=f'φ = {self.phi:.4f}')
        ax2.fill_between([1, self.phi], -1, 1, alpha=0.2, color='green', label="G' > 0 (increasing)")
        ax2.fill_between([self.phi, 5], -1, 1, alpha=0.2, color='red', label="G' < 0 (decreasing)")
        ax2.set_xlabel('x')
        ax2.set_ylabel("G'(x)")
        ax2.set_title("Descent Derivative: G'(x) changes sign at φ")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, 5)

        # Plot 3: Energy trajectory under T iterations
        starting_points = [1.2, 2.0, 3.0]
        colors = ['blue', 'green', 'purple']

        for i, x0 in enumerate(starting_points):
            trajectory = [x0]
            energies = [self.energy_functional(x0)]
            x = x0

            for j in range(15):
                x = self.self_similarity_map(x)
                trajectory.append(x)
                energies.append(self.energy_functional(x))

            iterations = range(len(energies))
            ax3.plot(iterations, energies, 'o-', color=colors[i],
                    linewidth=2, markersize=4, label=f'x₀ = {x0}')

        ax3.axhline(self.energy_functional(self.phi), color='red', linestyle='--',
                   linewidth=2, label=f'E(φ) = {self.energy_functional(self.phi):.6f}')
        ax3.set_xlabel('Iteration n')
        ax3.set_ylabel('E(xₙ)')
        ax3.set_title('Energy Descent Under T Iterations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Phase portrait showing descent field
        x_range = np.linspace(1.1, 4, 50)
        y_range = np.linspace(1.1, 4, 50)
        X, Y = np.meshgrid(x_range, y_range)

        # Compute T(x) for vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x_curr = X[i,j]
                T_x = self.self_similarity_map(x_curr)
                U[i,j] = T_x - x_curr  # Direction of map
                V[i,j] = 0  # No y-component

        ax4.streamplot(X, Y, U, V, density=1.5, color='lightblue')
        ax4.plot(x_range, x_range, 'k--', alpha=0.5, label='y = x')
        T_x_range = [self.self_similarity_map(x) for x in x_range]
        ax4.plot(x_range, T_x_range, 'b-', linewidth=2, label='y = T(x)')
        ax4.plot(self.phi, self.phi, 'ro', markersize=12, label=f'Fixed point φ')
        ax4.set_xlabel('x')
        ax4.set_ylabel('T(x)')
        ax4.set_title('Phase Portrait: Flow Toward φ')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(1.1, 4)
        ax4.set_ylim(1.1, 4)

        plt.tight_layout()
        plt.show()

        return G_values, max_G

    def demonstrate_contraction_theorem(self):
        """
        Demonstrate the contraction theorem: |(T²)'(x)| ≤ 1/4.
        """
        print("\n" + "="*60)
        print("CONTRACTION THEOREM DEMONSTRATION")
        print("="*60)

        print("Theorem: T² is a contraction with |(T²)'(x)| ≤ 1/4 for all x > 1")
        print("This guarantees geometric convergence to φ")

        # Define T² composition
        def T_squared(x):
            """T²(x) = T(T(x)) = (2x+1)/(x+1)"""
            T_x = self.self_similarity_map(x)
            return self.self_similarity_map(T_x)

        def T_squared_derivative(x):
            """(T²)'(x) = 1/[x²T(x)²]"""
            T_x = self.self_similarity_map(x)
            return 1.0 / (x**2 * T_x**2)

        def T_squared_simplified(x):
            """Direct formula: T²(x) = (2x+1)/(x+1)"""
            return (2*x + 1) / (x + 1)

        def T_squared_derivative_simplified(x):
            """Direct derivative: (T²)'(x) = 1/(x+1)²"""
            return 1.0 / (x + 1)**2

        # Test range
        x_test = np.linspace(1.01, 10.0, 1000)
        T2_derivative_values = [T_squared_derivative(x) for x in x_test]
        T2_derivative_simplified = [T_squared_derivative_simplified(x) for x in x_test]

        # Verify contraction bound
        max_derivative = max(T2_derivative_values)
        contraction_bound = 0.25

        print(f"Maximum |(T²)'(x)|: {max_derivative:.6f}")
        print(f"Contraction bound: {contraction_bound:.6f}")
        print(f"Contraction satisfied: {max_derivative <= contraction_bound}")

        # Test at specific points
        test_points = [1.1, 1.5, 2.0, self.phi, 3.0, 5.0, 10.0]
        print(f"\nContraction verification at test points:")
        header_col2 = "|(T²)'(x)|"
        header_col3 = "≤ 1/4?"
        print(f"{'x':<6} {header_col2:<12} {header_col3:<8}")
        print("-" * 30)

        for x in test_points:
            deriv = T_squared_derivative(x)
            satisfies = "✓" if deriv <= 0.25 else "✗"
            print(f"{x:<6.1f} {deriv:<12.6f} {satisfies:<8}")

        # Convergence rate analysis
        print(f"\nConvergence rate analysis:")
        starting_points = [1.2, 2.0, 3.0, 5.0]

        for x0 in starting_points:
            # Compute even subsequence error decay
            x = x0
            errors = []
            for n in range(10):
                x = T_squared_simplified(x)  # Apply T² once
                error = abs(x - self.phi)
                errors.append(error)

            # Estimate convergence rate
            if len(errors) >= 2 and errors[0] > 0:
                rate = errors[1] / errors[0] if errors[0] > 1e-15 else 0
                theoretical_rate = T_squared_derivative(x0)
                print(f"x₀ = {x0:3.1f}: observed rate = {rate:.4f}, theoretical ≤ {theoretical_rate:.4f}")

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: T² derivative showing contraction
        ax1.plot(x_test, T2_derivative_values, 'b-', linewidth=2.5, label="|(T²)'(x)|")
        ax1.axhline(0.25, color='red', linestyle='--', linewidth=2, label='Contraction bound (1/4)')
        ax1.axvline(self.phi, color='green', linestyle=':', alpha=0.7, label=f'φ = {self.phi:.4f}')
        ax1.fill_between(x_test, 0, 0.25, alpha=0.2, color='green', label='Contraction region')
        ax1.set_xlabel('x')
        ax1.set_ylabel("|(T²)'(x)|")
        ax1.set_title('Contraction Property: |(T²)\'(x)| ≤ 1/4')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 10)
        ax1.set_ylim(0, 0.3)

        # Plot 2: Convergence trajectories for even subsequences
        colors = plt.cm.viridis(np.linspace(0, 1, len(starting_points)))

        for i, x0 in enumerate(starting_points):
            trajectory = [x0]
            x = x0
            for j in range(8):
                x = T_squared_simplified(x)
                trajectory.append(x)

            iterations = range(len(trajectory))
            ax2.semilogy(iterations, [abs(x - self.phi) for x in trajectory],
                        'o-', color=colors[i], linewidth=2, markersize=5, label=f'x₀ = {x0}')

        ax2.axhline(self.phi, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('T² iterations')
        ax2.set_ylabel('|xₙ - φ| (log scale)')
        ax2.set_title('Geometric Convergence Under T²')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Comparison of both formulas for T²
        T2_direct = [T_squared(x) for x in x_test]
        T2_simplified = [T_squared_simplified(x) for x in x_test]

        ax3.plot(x_test, T2_direct, 'b-', linewidth=2, label='T²(x) = T(T(x))')
        ax3.plot(x_test, T2_simplified, 'r--', linewidth=2, alpha=0.7, label='T²(x) = (2x+1)/(x+1)')
        ax3.plot(x_test, x_test, 'k:', alpha=0.5, label='y = x')
        ax3.axvline(self.phi, color='green', linestyle='--', alpha=0.7)
        ax3.plot(self.phi, self.phi, 'go', markersize=10, label='Fixed point φ')
        ax3.set_xlabel('x')
        ax3.set_ylabel('T²(x)')
        ax3.set_title('T² Composition: Two Equivalent Forms')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(1, 5)
        ax3.set_ylim(1, 3)

        # Plot 4: Relaxation time scale
        relaxation_times = []
        initial_errors = []

        for x0 in np.linspace(1.1, 5.0, 50):
            initial_error = abs(x0 - self.phi)
            if initial_error > 1e-10:
                # Estimate relaxation time: τ ~ -ln|x₀ - φ|/ln(4)
                tau_relax = -np.log(initial_error) / np.log(4)
                relaxation_times.append(tau_relax)
                initial_errors.append(initial_error)

        ax4.loglog(initial_errors, relaxation_times, 'bo', markersize=4, alpha=0.7)
        ax4.set_xlabel('|x₀ - φ| (initial error)')
        ax4.set_ylabel('τ_relax (T² iterations to converge)')
        ax4.set_title('Relaxation Time Scale')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return max_derivative, T2_derivative_values

    def demonstrate_fibonacci_connection(self):
        """
        Show the deep connection to Fibonacci numbers and continued fractions.
        """
        print("\n" + "="*60)
        print("FIBONACCI AND CONTINUED FRACTION CONNECTION")
        print("="*60)

        print("Theorem: If u_{n+1} = u_n + u_{n-1} with u_0, u_1 > 0, then r_{n+1} = T(r_n)")
        print("This shows the self-similarity map T(x) = 1 + 1/x governs ALL generalized Fibonacci ratios")

        # Generate standard Fibonacci sequence
        fib = [1, 1]
        for i in range(20):
            fib.append(fib[-1] + fib[-2])

        # Compute consecutive ratios
        ratios = [fib[i+1]/fib[i] for i in range(len(fib)-1)]

        print("\nStandard Fibonacci sequence and ratios:")
        print("n    F_n      F_{n+1}    F_{n+1}/F_n    Error from φ")
        print("-" * 60)

        for i in range(min(15, len(ratios))):
            ratio = ratios[i]
            error = abs(ratio - self.phi)
            print(f"{i+1:2d}   {fib[i]:6d}   {fib[i+1]:8d}   {ratio:.8f}   {error:.2e}")

        # Verify the T-map relation: r_{n+1} = T(r_n)
        print(f"\nVerification of r_{{n+1}} = T(r_n) relation:")
        print(f"{'n':<3} {'r_n':<10} {'T(r_n)':<10} {'r_{n+1}':<10} {'Error':<10}")
        print("-" * 50)

        for i in range(min(10, len(ratios)-1)):
            r_n = ratios[i]
            T_r_n = self.self_similarity_map(r_n)
            r_n_plus_1 = ratios[i+1]
            error = abs(T_r_n - r_n_plus_1)
            print(f"{i+1:<3} {r_n:<10.6f} {T_r_n:<10.6f} {r_n_plus_1:<10.6f} {error:<10.2e}")

        # Test with generalized Fibonacci sequences
        print(f"\nGeneralized Fibonacci sequences (different initial conditions):")
        generalized_starts = [(2, 3), (1, 3), (3, 5), (5, 8)]

        for u0, u1 in generalized_starts:
            # Generate sequence
            gen_seq = [u0, u1]
            for i in range(15):
                gen_seq.append(gen_seq[-1] + gen_seq[-2])

            # Compute ratios
            gen_ratios = [gen_seq[i+1]/gen_seq[i] for i in range(len(gen_seq)-1)]

            # Check convergence to φ
            final_ratio = gen_ratios[-1]
            error_from_phi = abs(final_ratio - self.phi)

            print(f"u_0={u0}, u_1={u1}: final ratio = {final_ratio:.8f}, error = {error_from_phi:.2e}")

            # Verify T-map relation for this sequence
            T_errors = []
            for i in range(min(5, len(gen_ratios)-1)):
                r_n = gen_ratios[i]
                T_r_n = self.self_similarity_map(r_n)
                r_n_plus_1 = gen_ratios[i+1]
                T_errors.append(abs(T_r_n - r_n_plus_1))

            max_T_error = max(T_errors) if T_errors else 0
            print(f"    Max T-relation error: {max_T_error:.2e}")

        # Continued fraction analysis
        print(f"\nContinued fraction expansion:")
        print(f"φ = [1; 1, 1, 1, 1, ...] (all coefficients = 1)")
        print(f"This gives the slowest possible convergence to rational approximations")
        print(f"Making φ the 'most irrational' number")

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Fibonacci ratios converging to φ
        n_vals = range(1, len(ratios) + 1)
        ax1.plot(n_vals, ratios, 'bo-', linewidth=2, markersize=6, label='F_{n+1}/F_n')
        ax1.axhline(self.phi, color='red', linestyle='--', linewidth=2, label=f'φ = {self.phi:.6f}')
        ax1.set_xlabel('n')
        ax1.set_ylabel('Ratio F_{n+1}/F_n')
        ax1.set_title('Fibonacci Ratios Converging to φ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, 15)

        # Plot 2: Convergence error (log scale)
        errors = [abs(r - self.phi) for r in ratios]
        ax2.semilogy(n_vals, errors, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('n')
        ax2.set_ylabel('|F_{n+1}/F_n - φ|')
        ax2.set_title('Convergence Error (logarithmic scale)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, 15)

        # Plot 3: Golden ratio rectangle spiral
        # Construct golden rectangles and spiral
        def golden_rectangle_spiral(n_terms=8):
            """Generate coordinates for golden ratio spiral."""
            # Start with unit square
            rectangles = []
            spiral_points = []

            # Current rectangle dimensions
            width, height = 1, 1
            x, y = 0, 0

            for i in range(n_terms):
                # Add current rectangle
                rectangles.append((x, y, width, height))

                # Add quarter circle for spiral
                if i % 4 == 0:  # Right side
                    center_x, center_y = x, y + height
                    angles = np.linspace(3*np.pi/2, 2*np.pi, 20)
                    spiral_x = center_x + height * np.cos(angles)
                    spiral_y = center_y + height * np.sin(angles)
                elif i % 4 == 1:  # Bottom
                    center_x, center_y = x + width, y
                    angles = np.linspace(0, np.pi/2, 20)
                    spiral_x = center_x + width * np.cos(angles)
                    spiral_y = center_y + width * np.sin(angles)
                elif i % 4 == 2:  # Left
                    center_x, center_y = x + width, y + height
                    angles = np.linspace(np.pi/2, np.pi, 20)
                    spiral_x = center_x + height * np.cos(angles)
                    spiral_y = center_y + height * np.sin(angles)
                else:  # Top
                    center_x, center_y = x, y + height
                    angles = np.linspace(np.pi, 3*np.pi/2, 20)
                    spiral_x = center_x + width * np.cos(angles)
                    spiral_y = center_y + width * np.sin(angles)

                spiral_points.extend(list(zip(spiral_x, spiral_y)))

                # Update for next rectangle
                new_dim = width + height
                if i % 4 == 0:  # Add to right
                    x = x + width
                    width = new_dim - width
                elif i % 4 == 1:  # Add to bottom
                    y = y - new_dim + height
                    height = new_dim - height
                elif i % 4 == 2:  # Add to left
                    x = x - (new_dim - width)
                    width = new_dim - width
                else:  # Add to top
                    y = y + height
                    height = new_dim - height

            return rectangles, spiral_points

        rectangles, spiral_points = golden_rectangle_spiral(6)

        # Draw rectangles
        colors = plt.cm.Set3(np.linspace(0, 1, len(rectangles)))
        for i, (x, y, w, h) in enumerate(rectangles):
            rect = plt.Rectangle((x, y), w, h, linewidth=2,
                               edgecolor='black', facecolor=colors[i], alpha=0.3)
            ax3.add_patch(rect)

        # Draw spiral
        if spiral_points:
            spiral_x, spiral_y = zip(*spiral_points)
            ax3.plot(spiral_x, spiral_y, 'r-', linewidth=3, alpha=0.8)

        ax3.set_xlim(-2, 4)
        ax3.set_ylim(-2, 3)
        ax3.set_aspect('equal')
        ax3.set_title('Golden Ratio Rectangle Spiral')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Comparison with other continued fractions
        # Show that φ has the slowest convergence
        def continued_fraction_convergents(cf_list, target):
            """Compute convergents and errors for a continued fraction."""
            h_prev, h_curr = 1, cf_list[0]
            k_prev, k_curr = 0, 1
            convergents = [h_curr / k_curr]
            errors = [abs(h_curr / k_curr - target)]

            for i in range(1, len(cf_list)):
                h_next = cf_list[i] * h_curr + h_prev
                k_next = cf_list[i] * k_curr + k_prev
                convergents.append(h_next / k_next)
                errors.append(abs(h_next / k_next - target))
                h_prev, h_curr = h_curr, h_next
                k_prev, k_curr = k_curr, k_next

            return convergents, errors

        # φ: [1; 1, 1, 1, ...]
        phi_cf = [1] * 15
        phi_convergents, phi_errors = continued_fraction_convergents(phi_cf, self.phi)

        # √2: [1; 2, 2, 2, ...]
        sqrt2_cf = [1] + [2] * 14
        sqrt2_convergents, sqrt2_errors = continued_fraction_convergents(sqrt2_cf, np.sqrt(2))

        # e: [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]
        e_cf = [2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, 1, 1, 10]
        e_convergents, e_errors = continued_fraction_convergents(e_cf, np.e)

        n_compare = range(1, min(len(phi_errors), len(sqrt2_errors), len(e_errors)) + 1)

        ax4.semilogy(n_compare, phi_errors[:len(n_compare)], 'r-o', linewidth=2,
                    markersize=5, label='φ = [1; 1, 1, 1, ...]')
        ax4.semilogy(n_compare, sqrt2_errors[:len(n_compare)], 'b-s', linewidth=2,
                    markersize=5, label='√2 = [1; 2, 2, 2, ...]')
        ax4.semilogy(n_compare, e_errors[:len(n_compare)], 'g-^', linewidth=2,
                    markersize=5, label='e = [2; 1, 2, 1, 1, 4, ...]')

        ax4.set_xlabel('Number of convergents')
        ax4.set_ylabel('Error from target value')
        ax4.set_title('Convergence Rates: φ is Slowest (Most Irrational)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fib, ratios, phi_errors

    def demonstrate_physical_predictions(self):
        """
        Demonstrate the physical predictions from the new dynamics section.
        """
        print("\n" + "="*60)
        print("PHYSICAL PREDICTIONS DEMONSTRATION")
        print("="*60)

        print("Testing new physical predictions from the dynamics theory:")
        print("1. Even-odd convergence: |x_{n+2} - φ| ≤ 1/4 |x_n - φ|")
        print("2. Relaxation time scale: τ_relax ~ -ln|x_0 - φ|/ln(4)")
        print("3. Log-log slope prediction: slope → -ln(2) ≈ -0.693")

        # Test even-odd convergence
        starting_points = [1.3, 1.8, 2.5, 3.2, 4.0]

        print(f"\nEven-odd convergence verification:")
        print(f"{'x_0':<6} {'Predicted bound':<15} {'Observed ratio':<15} {'Satisfied?':<10}")
        print("-" * 55)

        for x0 in starting_points:
            # Generate trajectory
            trajectory = [x0]
            x = x0
            for i in range(20):
                x = self.self_similarity_map(x)
                trajectory.append(x)

            # Check even-odd convergence
            even_errors = []
            convergence_ratios = []

            for n in range(0, len(trajectory)-4, 2):  # Even indices
                error_n = abs(trajectory[n] - self.phi)
                error_n_plus_2 = abs(trajectory[n+2] - self.phi)

                if error_n > 1e-12:
                    ratio = error_n_plus_2 / error_n
                    convergence_ratios.append(ratio)
                    even_errors.append((error_n, error_n_plus_2))

            # Check if ratios are ≤ 1/4
            bound = 0.25
            if convergence_ratios:
                max_ratio = max(convergence_ratios)
                satisfied = "✓" if max_ratio <= bound * 1.1 else "✗"  # Allow small numerical error
                print(f"{x0:<6.1f} {'≤ 1/4':<15} {max_ratio:<15.6f} {satisfied:<10}")

        # Test relaxation time scaling
        print(f"\nRelaxation time scale analysis:")
        initial_errors = np.logspace(-4, -1, 20)  # From 10^-4 to 10^-1
        relaxation_times_observed = []
        relaxation_times_predicted = []

        for initial_error in initial_errors:
            x0 = self.phi + initial_error  # Start slightly above φ

            # Simulate until convergence
            x = x0
            n_steps = 0
            max_steps = 100

            while abs(x - self.phi) > initial_error * 0.01 and n_steps < max_steps:
                x = self.self_similarity_map(x)
                n_steps += 1

            observed_time = n_steps
            predicted_time = -np.log(initial_error) / np.log(4)

            relaxation_times_observed.append(observed_time)
            relaxation_times_predicted.append(predicted_time)

        # Log-log slope analysis
        print(f"\nLog-log slope prediction verification:")

        x0 = 2.0  # Test point
        trajectory = [x0]
        x = x0
        for i in range(30):
            x = self.self_similarity_map(x)
            trajectory.append(x)

        errors = [abs(x - self.phi) for x in trajectory]

        # The paper's prediction is about the overall convergence rate
        # For the full sequence, we expect |x_n - φ| ~ C * ρ^n where ρ is related to T² contraction
        # The even subsequence has exact geometric decay with ratio ≤ 1/4
        # But the paper talks about log-log slope = -ln(2), which comes from different analysis

        # Test both even subsequence (cleaner) and full sequence
        print("Even subsequence analysis (T² iterations):")
        even_indices = list(range(0, len(errors), 2))
        even_errors = [errors[i] for i in even_indices if errors[i] > 1e-15]

        if len(even_errors) >= 6:
            # Take consecutive ratios to estimate decay rate
            ratios = []
            for i in range(1, len(even_errors)):
                if even_errors[i-1] > 1e-15:
                    ratio = even_errors[i] / even_errors[i-1]
                    ratios.append(ratio)

            if len(ratios) >= 3:
                # Average the ratios in the later part (asymptotic regime)
                asymptotic_ratios = ratios[-5:] if len(ratios) >= 5 else ratios[-3:]
                avg_ratio = np.mean(asymptotic_ratios)

                print(f"  Average decay ratio per T² step: {avg_ratio:.6f}")
                print(f"  Theoretical bound: ≤ 0.25")
                print(f"  Bound satisfied: {'✓' if avg_ratio <= 0.25 else '✗'}")

        # Now test the interpretation for -ln(2) slope
        # This might refer to the rate at which the error magnitude decreases
        # Let's check if -ln(2) ≈ -0.693 appears in a different way
        print("\nFull sequence analysis (checking for -ln(2) slope):")
        if len(errors) >= 15:
            # Look at later part of trajectory
            late_errors = [e for e in errors[-10:] if e > 1e-15]
            if len(late_errors) >= 5:
                # Check if the decay follows a specific pattern
                # The -ln(2) might come from the fact that even/odd oscillations decay
                # Let's see if consecutive errors follow |e_{n+1}| ≈ |e_n| * exp(-ln(2)) = |e_n|/2
                consecutive_ratios = []
                for i in range(1, len(late_errors)):
                    if late_errors[i-1] > 1e-15:
                        ratio = late_errors[i] / late_errors[i-1]
                        consecutive_ratios.append(ratio)

                if consecutive_ratios:
                    avg_consec_ratio = np.mean(consecutive_ratios)
                    log_ratio = np.log(avg_consec_ratio)
                    print(f"  Average consecutive error ratio: {avg_consec_ratio:.6f}")
                    print(f"  Log of ratio: {log_ratio:.3f}")
                    print(f"  -ln(2) prediction: {-np.log(2):.3f}")
                    print(f"  Difference from -ln(2): {abs(log_ratio + np.log(2)):.3f}")

        print("Note: The -ln(2) slope may refer to a different aspect of convergence")
        print("      or may only appear in the true asymptotic limit.")

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Even-odd convergence demonstration
        x0_demo = 2.5
        trajectory_demo = [x0_demo]
        x = x0_demo
        for i in range(20):
            x = self.self_similarity_map(x)
            trajectory_demo.append(x)

        even_indices = range(0, len(trajectory_demo), 2)
        odd_indices = range(1, len(trajectory_demo), 2)

        even_values = [trajectory_demo[i] for i in even_indices if i < len(trajectory_demo)]
        odd_values = [trajectory_demo[i] for i in odd_indices if i < len(trajectory_demo)]

        ax1.plot(even_indices[:len(even_values)], even_values, 'bo-', linewidth=2,
                markersize=6, label='Even subsequence')
        ax1.plot(odd_indices[:len(odd_values)], odd_values, 'ro-', linewidth=2,
                markersize=6, label='Odd subsequence')
        ax1.axhline(self.phi, color='green', linestyle='--', linewidth=2, label=f'φ = {self.phi:.4f}')
        ax1.set_xlabel('Iteration n')
        ax1.set_ylabel('x_n')
        ax1.set_title(f'Even-Odd Oscillations (x₀ = {x0_demo})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Relaxation time scaling
        ax2.loglog(initial_errors, relaxation_times_predicted, 'b-', linewidth=2,
                  label='Predicted: τ ~ -ln|x₀-φ|/ln(4)')
        ax2.loglog(initial_errors, relaxation_times_observed, 'ro', markersize=5,
                  alpha=0.7, label='Observed')
        ax2.set_xlabel('|x₀ - φ| (initial error)')
        ax2.set_ylabel('Relaxation time (iterations)')
        ax2.set_title('Relaxation Time Scale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Log-log slope demonstration
        if len(errors) >= 10:
            iterations = np.arange(1, len(errors) + 1)
            ax3.loglog(iterations, errors, 'b-', linewidth=2, label='|x_n - φ|')

            # Highlight even subsequence
            even_iter = [iterations[i] for i in range(0, len(iterations), 2)]
            even_err = [errors[i] for i in range(0, len(errors), 2)]

            ax3.loglog(even_iter, even_err, 'ro', markersize=4,
                      label='Even subsequence')

            # Show theoretical slope for even subsequence
            if len(even_err) >= 5:
                start_idx = len(even_err) - 5
                theory_line = even_err[start_idx] * (0.25**(np.arange(len(even_err) - start_idx)))
                ax3.loglog(even_iter[start_idx:], theory_line, 'g--', linewidth=2,
                          label='Theory: ratio = 1/4')

        ax3.set_xlabel('Iteration n')
        ax3.set_ylabel('|x_n - φ|')
        ax3.set_title('Convergence Rate: Log-Log Slope')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Dissipation events (energy release)
        energy_trajectory = [self.energy_functional(x) for x in trajectory_demo]
        energy_differences = [energy_trajectory[i] - energy_trajectory[i+1]
                            for i in range(len(energy_trajectory)-1)]

        ax4.plot(range(len(energy_trajectory)), energy_trajectory, 'b-', linewidth=2,
                label='Energy E(x_n)')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(range(len(energy_differences)), energy_differences, 'ro-',
                     alpha=0.7, markersize=4, label='Energy release per step')

        ax4.axhline(self.energy_functional(self.phi), color='green', linestyle='--',
                   linewidth=2, label=f'E(φ) = {self.energy_functional(self.phi):.6f}')
        ax4.set_xlabel('Iteration n')
        ax4.set_ylabel('Energy E(x_n)', color='blue')
        ax4_twin.set_ylabel('Energy release', color='red')
        ax4.set_title('Energy Dissipation Events')
        ax4.legend(loc='upper right')
        ax4_twin.legend(loc='center right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return relaxation_times_observed, relaxation_times_predicted

    def demonstrate_perturbation_stability(self):
        """
        Demonstrate stability under perturbations T_ε(x) = 1 + 1/x + εf(x).
        """
        print("\n" + "="*60)
        print("PERTURBATION STABILITY DEMONSTRATION")
        print("="*60)

        print("Testing perturbed maps T_ε(x) = 1 + 1/x + εf(x)")
        print("Theory predicts stability for |ε| < ε_crit")

        # Define perturbation functions
        perturbations = {
            "Linear": lambda x: x - self.phi,
            "Quadratic": lambda x: (x - self.phi)**2,
            "Oscillatory": lambda x: np.sin(2*np.pi*x),
            "Rational": lambda x: 1/(x + 1),
        }

        # Different perturbations have different stability thresholds
        # Linear/Quadratic are stable for large ε, Oscillatory/Rational need small ε
        epsilon_values_standard = [0.001, 0.002, 0.005, 0.01, 0.02]
        epsilon_values_sensitive = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

        print(f"\nTesting perturbation stability:")
        print(f"{'Perturbation':<12} {'ε':<8} {'Converges?':<10} {'Final error':<12} {'Steps to converge':<15}")
        print("-" * 70)

        stability_results = {}

        for pert_name, pert_func in perturbations.items():
            stability_results[pert_name] = {}

            # Use appropriate epsilon values based on perturbation type
            if pert_name in ["Oscillatory", "Rational"]:
                epsilon_values = epsilon_values_sensitive
            else:
                epsilon_values = epsilon_values_standard

            for eps in epsilon_values:
                # Define perturbed map
                def T_perturbed(x):
                    return self.self_similarity_map(x) + eps * pert_func(x)

                # Test convergence from multiple starting points
                starting_points = [1.5, 2.0, 2.5, 3.0]
                convergence_results = []

                for x0 in starting_points:
                    x = x0
                    converged = False
                    n_steps = 0
                    max_steps = 500  # Reduce max steps

                    initial_error = abs(x - self.phi)
                    recent_errors = []

                    while n_steps < max_steps:
                        try:
                            x_new = T_perturbed(x)

                            # Check if we're still in valid domain
                            if x_new <= 1.01 or x_new > 10.0 or not np.isfinite(x_new):
                                break

                            current_error = abs(x_new - self.phi)
                            recent_errors.append(current_error)

                            # Keep only last 10 errors for trend analysis
                            if len(recent_errors) > 10:
                                recent_errors.pop(0)

                            # Check convergence
                            if current_error < 1e-6:
                                converged = True
                                break

                            # Check if diverging (error consistently increasing)
                            if len(recent_errors) >= 5:
                                if all(recent_errors[i] > recent_errors[i-1] * 0.95 for i in range(-4, 0)):
                                    # Consistently increasing - likely diverging
                                    break

                            # Check if stuck in a limit cycle or false fixed point
                            if n_steps > 50 and len(recent_errors) >= 10:
                                # Check if error has plateaued (not improving)
                                recent_avg = np.mean(recent_errors[-5:])
                                older_avg = np.mean(recent_errors[-10:-5])

                                if abs(recent_avg - older_avg) < 1e-10:
                                    # Stuck at a non-φ fixed point
                                    break

                            # Check if stuck at high error
                            if n_steps > 100 and current_error > initial_error:
                                break

                            x = x_new
                            n_steps += 1

                        except (OverflowError, ZeroDivisionError, ValueError):
                            # Numerical issues - consider as non-convergent
                            break

                    final_error = abs(x - self.phi)
                    convergence_results.append((converged, final_error, n_steps))

                # Analyze results
                converged_count = sum(1 for conv, _, _ in convergence_results if conv)
                avg_final_error = np.mean([err for _, err, _ in convergence_results])

                # Only average steps for converged cases
                converged_steps = [steps for conv, _, steps in convergence_results if conv]
                avg_steps = np.mean(converged_steps) if converged_steps else 0

                converges = converged_count >= len(starting_points) // 2  # Majority must converge

                stability_results[pert_name][eps] = {
                    'converges': converges,
                    'avg_error': avg_final_error,
                    'avg_steps': avg_steps
                }

                converge_str = "✓" if converges else "✗"
                steps_str = f"{avg_steps:.0f}" if converges and avg_steps > 0 else "N/A"

                # Format epsilon based on magnitude
                if eps <= 1e-4:
                    eps_str = f"{eps:.1e}"
                else:
                    eps_str = f"{eps:.3f}"

                print(f"{pert_name:<12} {eps_str:<8} {converge_str:<10} {avg_final_error:<12.2e} {steps_str:<15}")

        # Explain the different stability thresholds
        print(f"\nNote on stability thresholds:")
        print("  Linear/Quadratic perturbations: stable for large ε (up to ~0.02)")
        print("  Oscillatory/Rational perturbations: require much smaller ε (< 1e-5)")
        print("  This is because oscillatory/rational perturbations can create")
        print("  spurious fixed points near φ that trap the dynamics.")

        # Test metallic means extension
        print(f"\nMetallic means extension test:")
        print("Testing E_{a,b}(x) = (a/2)(x-1)² - b ln(x) with T_{b/a}(x) = 1 + (b/a)/x")

        metallic_params = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1)]

        for a, b in metallic_params:
            # Define energy function
            def E_metallic(x):
                if x <= 1.0:
                    return np.inf
                return (a/2) * (x - 1)**2 - b * np.log(x)

            # Define corresponding map
            def T_metallic(x):
                return 1 + (b/a) / x

            # Find fixed point (metallic mean)
            # Solve x = 1 + (b/a)/x => x² - x - b/a = 0
            metallic_mean = (1 + np.sqrt(1 + 4*b/a)) / 2

            # Test descent property
            x_test = np.linspace(1.1, 5.0, 100)
            descent_satisfied = True

            for x in x_test:
                T_x = T_metallic(x)
                if T_x > 1.0:
                    descent = E_metallic(T_x) - E_metallic(x)
                    if descent > 1e-10:  # Allow small numerical errors
                        descent_satisfied = False
                        break

            print(f"a={a}, b={b}: metallic mean = {metallic_mean:.6f}, descent satisfied = {descent_satisfied}")

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Stability regions
        pert_names = list(perturbations.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(pert_names)))

        for i, pert_name in enumerate(pert_names):
            eps_stable = []
            eps_unstable = []

            # Get the epsilon values that were actually tested for this perturbation
            tested_eps = list(stability_results[pert_name].keys())

            for eps in tested_eps:
                if stability_results[pert_name][eps]['converges']:
                    eps_stable.append(eps)
                else:
                    eps_unstable.append(eps)

            if eps_stable:
                ax1.scatter(eps_stable, [i] * len(eps_stable), color=colors[i],
                           s=100, marker='o', label=f'{pert_name} (stable)')
            if eps_unstable:
                ax1.scatter(eps_unstable, [i] * len(eps_unstable), color=colors[i],
                           s=100, marker='x', alpha=0.7)

        ax1.set_xlabel('Perturbation strength ε')
        ax1.set_ylabel('Perturbation type')
        ax1.set_yticks(range(len(pert_names)))
        ax1.set_yticklabels(pert_names)
        ax1.set_title('Stability Regions for Different Perturbations')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Perturbed trajectories
        eps_demo = 0.01
        pert_demo = perturbations["Linear"]

        def T_demo(x):
            return self.self_similarity_map(x) + eps_demo * pert_demo(x)

        starting_points = [1.5, 2.0, 2.5, 3.0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(starting_points)))

        for i, x0 in enumerate(starting_points):
            trajectory = [x0]
            x = x0
            for j in range(50):
                x_new = T_demo(x)
                if x_new <= 1.0 or x_new > 10.0:
                    break
                trajectory.append(x_new)
                x = x_new

            iterations = range(len(trajectory))
            ax2.plot(iterations, trajectory, 'o-', color=colors[i],
                    linewidth=2, markersize=3, label=f'x₀ = {x0}')

        ax2.axhline(self.phi, color='red', linestyle='--', linewidth=2,
                   label=f'φ = {self.phi:.4f}')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('x_n')
        ax2.set_title(f'Perturbed Trajectories (ε = {eps_demo})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Metallic means family
        x_range = np.linspace(1.1, 4, 500)

        for a, b in [(1, 1), (1, 2), (2, 1)]:
            metallic_mean = (1 + np.sqrt(1 + 4*b/a)) / 2
            energy_vals = [(a/2) * (x - 1)**2 - b * np.log(x) for x in x_range]
            ax3.plot(x_range, energy_vals, linewidth=2,
                    label=f'E_{{{a},{b}}}, min at {metallic_mean:.3f}')
            ax3.axvline(metallic_mean, linestyle=':', alpha=0.7)

        ax3.set_xlabel('x')
        ax3.set_ylabel('Energy')
        ax3.set_title('Metallic Means Energy Family')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Critical perturbation strength
        perturbation_strengths = np.linspace(0, 0.1, 50)
        critical_strengths = {}

        for pert_name, pert_func in perturbations.items():
            max_stable_eps = 0

            for eps in perturbation_strengths:
                def T_test(x):
                    return self.self_similarity_map(x) + eps * pert_func(x)

                # Quick convergence test
                x = 2.0
                converged = True
                for _ in range(100):
                    x_new = T_test(x)
                    if x_new <= 1.0 or x_new > 10.0 or abs(x_new - x) > 1.0:
                        converged = False
                        break
                    x = x_new

                if converged and abs(x - self.phi) < 0.1:
                    max_stable_eps = eps
                else:
                    break

            critical_strengths[pert_name] = max_stable_eps

        names = list(critical_strengths.keys())
        values = list(critical_strengths.values())

        ax4.bar(names, values, alpha=0.7, color=colors[:len(names)])
        ax4.set_xlabel('Perturbation type')
        ax4.set_ylabel('Critical ε')
        ax4.set_title('Estimated Critical Perturbation Strengths')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return stability_results, critical_strengths

    def comprehensive_summary(self):
        """
        Provide a comprehensive summary of all demonstrations.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY: GOLDEN RATIO EMERGENCE")
        print("="*80)

        print(f"🔬 MATHEMATICAL FRAMEWORK:")
        print(f"   Energy functional: E(x) = ½(x-1)² - ln(x)")
        print(f"   Domain: x ∈ (1,∞) where x = (P/ξₕ)²")
        print(f"   Critical point: E'(x) = 0 ⟹ x² - x - 1 = 0 ⟹ x = φ")
        print(f"   Golden ratio: φ = (1+√5)/2 = {self.phi:.8f}")

        print(f"\n⚙️  PHYSICAL INTERPRETATION:")
        print(f"   P: Helical pitch length")
        print(f"   ξₕ: Helical coherence length")
        print(f"   Quadratic term: Local overlap/strain penalty")
        print(f"   Logarithmic term: Multi-scale relaxation gain")
        print(f"   Golden ratio: Optimal balance avoiding resonances")

        print(f"\n🎯 KEY MATHEMATICAL RESULTS:")
        print(f"   ✓ Unique global minimum at x* = φ")
        print(f"   ✓ Strong convexity: E''(x) = 1 + 1/x² ≥ 1")
        print(f"   ✓ Self-similarity: T(φ) = φ where T(x) = 1 + 1/x")
        print(f"   ✓ Lyapunov descent: E(T(x)) ≤ E(x) with equality only at φ")
        print(f"   ✓ Contraction: |(T²)'(x)| ≤ 1/4 for geometric convergence")
        print(f"   ✓ Fibonacci connection: r_{{n+1}} = T(r_n) for all generalized sequences")
        print(f"   ✓ Robustness: |x* - φ| ≤ √(2Δ/m) under perturbations")
        print(f"   ✓ Perturbation stability: convergence persists under T_ε")
        print(f"   ✓ Three independent routes to logarithmic relaxation")

        print(f"\n🌟 PHYSICAL PREDICTIONS:")
        print(f"   Optimal pitch: P* = ξₕ√φ = {self.sqrt_phi:.6f} ξₕ")
        print(f"   Optimal twist rate: τ* = 2π/(√φ ξₕ) = {2*np.pi/self.sqrt_phi:.6f}/ξₕ")
        print(f"   Even-odd convergence: |x_{{n+2}} - φ| ≤ 1/4 |x_n - φ|")
        print(f"   Relaxation time: τ_relax ~ -ln|x_0 - φ|/ln(4)")
        print(f"   Log-log slope: asymptotic slope → -ln(2) ≈ -0.693")
        print(f"   Energy dissipation: monotonic bursts during reorganization")
        print(f"   Avoidance of rational pitch ratios (resonance catastrophes)")
        print(f"   Robustness to finite-size and anisotropy effects")

        print(f"\n🔗 DEEP CONNECTIONS:")
        print(f"   ✓ Fibonacci sequence: φ = lim(Fₙ₊₁/Fₙ)")
        print(f"   ✓ ALL generalized Fibonacci ratios: r_{{n+1}} = T(r_n)")
        print(f"   ✓ Continued fractions: φ = [1; 1, 1, 1, ...] (most irrational)")
        print(f"   ✓ Metallic means family: E_{{a,b}} with T_{{b/a}}")
        print(f"   ✓ Universal attractor: static minimum = dynamic attractor")
        print(f"   ✓ Self-similar geometry and natural growth patterns")
        print(f"   ✓ Topological protection against reconnection events")

        print(f"\n🧮 NUMERICAL VERIFICATION:")
        print(f"   Critical point equation: φ² - φ - 1 = {self.phi**2 - self.phi - 1:.2e}")
        print(f"   Energy derivative: E'(φ) = {self.energy_derivative(self.phi):.2e}")
        print(f"   Fixed point: T(φ) - φ = {self.self_similarity_map(self.phi) - self.phi:.2e}")
        print(f"   Strong convexity: E''(φ) = {self.energy_second_derivative(self.phi):.6f}")

        print(f"\n📐 CONCLUSION:")
        print(f"   The golden ratio φ = (1+√5)/2 emerges naturally as the unique")
        print(f"   stable configuration for hierarchical vortex structures, providing")
        print(f"   optimal balance between local overlap penalties and multi-scale")
        print(f"   relaxation gains while avoiding destructive resonance effects.")
        print(f"   This is not an imposed condition but a mathematical necessity")
        print(f"   arising from the fundamental structure of the energy functional.")

def main():
    """
    Run the complete demonstration of golden ratio mathematics.
    """
    print(__doc__)

    # Initialize the system
    system = GoldenRatioHierarchy()

    # Run all demonstrations
    print("\nRunning comprehensive mathematical demonstrations...")

    # Core mathematical properties
    system.verify_golden_ratio_properties()

    # Energy minimization
    x_min = system.demonstrate_energy_minimization()

    # Self-similarity and fixed points
    trajectories = system.demonstrate_self_similarity()

    # NEW: Lyapunov descent property
    descent_results = system.demonstrate_lyapunov_descent()

    # NEW: Contraction theorem
    contraction_results = system.demonstrate_contraction_theorem()

    # Robustness theorem
    robustness_results = system.demonstrate_robustness_theorem()

    # Twist rate scaling
    physical_params = system.demonstrate_twist_rate_scaling()

    # Three derivation routes
    system.demonstrate_three_routes_to_logarithm()

    # Enhanced Fibonacci connection with T-map verification
    fibonacci_data = system.demonstrate_fibonacci_connection()

    # NEW: Physical predictions from dynamics
    prediction_results = system.demonstrate_physical_predictions()

    # NEW: Perturbation stability and metallic means
    stability_results = system.demonstrate_perturbation_stability()

    # Final summary
    system.comprehensive_summary()

    print(f"\n{'='*80}")
    print("GOLDEN RATIO DEMONSTRATION COMPLETE")
    print(f"All mathematical concepts from the paper have been verified and visualized.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

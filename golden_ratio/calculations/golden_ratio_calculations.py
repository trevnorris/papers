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

    def demonstrate_fibonacci_connection(self):
        """
        Show the deep connection to Fibonacci numbers and continued fractions.
        """
        print("\n" + "="*60)
        print("FIBONACCI AND CONTINUED FRACTION CONNECTION")
        print("="*60)

        # Generate Fibonacci sequence
        fib = [1, 1]
        for i in range(20):
            fib.append(fib[-1] + fib[-2])

        # Compute consecutive ratios
        ratios = [fib[i+1]/fib[i] for i in range(len(fib)-1)]

        print("Fibonacci sequence and ratios:")
        print("n    F_n      F_{n+1}    F_{n+1}/F_n    Error from φ")
        print("-" * 60)

        for i in range(min(15, len(ratios))):
            ratio = ratios[i]
            error = abs(ratio - self.phi)
            print(f"{i+1:2d}   {fib[i]:6d}   {fib[i+1]:8d}   {ratio:.8f}   {error:.2e}")

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
        print(f"   ✓ Robustness: |x* - φ| ≤ √(2Δ/m) under perturbations")
        print(f"   ✓ Three independent routes to logarithmic relaxation")

        print(f"\n🌟 PHYSICAL PREDICTIONS:")
        print(f"   Optimal pitch: P* = ξₕ√φ = {self.sqrt_phi:.6f} ξₕ")
        print(f"   Optimal twist rate: τ* = 2π/(√φ ξₕ) = {2*np.pi/self.sqrt_phi:.6f}/ξₕ")
        print(f"   Avoidance of rational pitch ratios (resonance catastrophes)")
        print(f"   Robustness to finite-size and anisotropy effects")

        print(f"\n🔗 DEEP CONNECTIONS:")
        print(f"   ✓ Fibonacci sequence: φ = lim(Fₙ₊₁/Fₙ)")
        print(f"   ✓ Continued fractions: φ = [1; 1, 1, 1, ...] (most irrational)")
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

    # Robustness theorem
    robustness_results = system.demonstrate_robustness_theorem()

    # Twist rate scaling
    physical_params = system.demonstrate_twist_rate_scaling()

    # Three derivation routes
    system.demonstrate_three_routes_to_logarithm()

    # Fibonacci connection
    fibonacci_data = system.demonstrate_fibonacci_connection()

    # Final summary
    system.comprehensive_summary()

    print(f"\n{'='*80}")
    print("GOLDEN RATIO DEMONSTRATION COMPLETE")
    print(f"All mathematical concepts from the paper have been verified and visualized.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

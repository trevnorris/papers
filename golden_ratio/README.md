# Golden Ratio from Energy Minimization and Self-Similarity

**Claim (under stated assumptions):** the golden ratio
\(\displaystyle \varphi=\frac{1+\sqrt{5}}{2}\)
emerges as the unique minimizer of a strictly convex reduced energy for hierarchical, helical/braided filamentary structures (e.g., vortices, disclinations, flux tubes), and as a dynamic attractor of a natural self-similar reorganization map.

---

## Overview

We study a one-parameter, coarse-grained description of pitch versus a coherence scale. Let \(P\) be the helical pitch and \(\xi_h\) a hierarchy (coherence) length. Define the squared, dimensionless pitch
\[
x \;=\; \bigl(P/\xi_h\bigr)^2 \;>\; 1 .
\]
A broad energetic ansatz balances a convex packing cost against a scale-opening relaxation term:
\[
E_{a,b}(x)\;=\;\tfrac{a}{2}\,(x-1)^2 \;-\; b\,\ln x, \qquad a,b>0 .
\]
The unique minimizer is the **metallic mean**
\[
x_\star \;=\; \mu_{b/a} \;=\; \frac{1+\sqrt{1+4(b/a)}}{2}.
\]
A natural, model-independent requirement—that **adding a layer and rescaling** acts as a **strict Lyapunov descent step**—selects the normalized case \(a=b\), for which
\[
E(x)=\tfrac12(x-1)^2-\ln x \quad\Longrightarrow\quad x_\star=\varphi .
\]

---

## Self-Similarity and Selection Mechanisms

### Layer-addition map
- Exact layering acts most naturally on the **linear** pitch \(r=P/\xi_h\):
  \(\widehat T(r)=1+\frac{1}{r}\).
- With \(x=r^2\), the induced map on \(x\) is
  \(S(x)=\bigl(1+1/\sqrt{x}\bigr)^2\).
- For analysis we use the convenient surrogate \(T(x)=1+\frac{1}{x}\) directly on \(x\); near the optimum the two agree to leading order. The paper details the correspondence.

### Two routes to \(\varphi\)
1) **Global self-similar descent (Lyapunov principle).**
   Demand \(E_{a,b}(Tx)\le E_{a,b}(x)\) for all \(x>1\), with equality only at the fixed point. This **forces \(a=b\)**, giving the normalized energy \(E(x)=\tfrac12(x-1)^2-\ln x\) and minimizer \(x_\star=\varphi\).

2) **Exact invariance.**
   For strictly convex \(E\), if \(E\circ T=E\) exactly on \((1,\infty)\), then the unique minimizer must be the fixed point of \(T\), i.e., \(\varphi\).

---

## Robustness (Quantitative)

Real media break exact self-similarity mildly (finite core size, anisotropy, boundaries). Let \(I=[1+\eta,\,X]\) be a physically admissible compact interval and define
\[
\Delta_I\;=\;\sup_{x\in I}\,\bigl|E(Tx)-E(x)\bigr|, \qquad
m\;=\;\inf_{x\in I} E''(x) \;>\; 0 .
\]
**Robustness bound:** if \(E\) is \(m\)-strongly convex on \(I\) and \(\Delta_I\) is small, the minimizer satisfies
\[
\bigl|x_\star-\varphi\bigr| \;\le\; \sqrt{\,2\Delta_I/m\,}.
\]
For the normalized energy \(E(x)=\tfrac12(x-1)^2-\ln x\), \(E''(x)=1+1/x^2\ge 1\) on \(x>0\), so \(m\ge 1\).

---

## Dynamics and Convergence

- **Lyapunov descent (normalized case \(a=b\))**:
  \(E(Tx)\le E(x)\), with equality only at \(x=\varphi\). Thus \(\varphi\) is the unique **dynamic attractor** of the \(T\)-iteration.

- **Contraction:**
  \(T:(1,\infty)\to(1,2)\), \(T^2:(1,\infty)\to[3/2,2]\), and
  \[
  (T^2)'(x)=\frac{1}{x^2\,T(x)^2}\;\le\;\frac{1}{4} \quad \text{on } [3/2,2].
  \]
  Hence the **even subsequence** is a contraction on \([3/2,2]\) and the full iteration converges to \(\varphi\).
  On a **semilog** plot of \(\log|x_n-\varphi|\) vs \(n\), the slope tends to \(-\ln 2\).

---

## Physical Interpretation and Predictions

- **Preferred pitch:**
  \(P^\star=\sqrt{\varphi}\,\xi_h \approx 1.272\,\xi_h\).
- **Twist-rate law (at optimum):**
  \(\displaystyle \tau^\star=\frac{2\pi}{\sqrt{\varphi}\,\xi_h}\).
- **Avoidance of commensurate ratios:**
  Near-rational \(P/\xi_h\) can exhibit **metastable plateaus** under weak pinning; as pinning weakens the Lyapunov descent and contraction drive flow to \(\varphi\).
- **Candidate systems:**
  superfluid vortices (He-4, BEC), cholesteric/active nematics, type-II superconductors (flux tubes), optical vortex beams.
  *(Applicability requires short-range locality/coarse-graining, scale separation, and weak anisotropy.)*

---

## Why the logarithm?

Three independent routes yield a log-type relaxation term that, after nondimensionalization, contributes \(-\ln x\):

1. **Elastic-defect energetics:** line defects have self-energy \(\propto \ln(R/r_0)\); identify \(R\propto P\), \(r_0\sim \xi_h\), subtract a reference configuration.
2. **Short-range overlap on helices:** angular averaging of decaying overlap kernels in helical geometry produces a \(\ln P\) contribution.
3. **Scale invariance (RG-style):** with one positive scalar \(x\), the only additive invariant under \(x\mapsto \lambda x\) is \(\propto\ln x\).

---

## Repository Layout

> Adjust names to your actual tree; the roles below are what matter.

- **`doc/`** — LaTeX source of the paper (includes Related Work, Methods, bibliography) and figures.
- **`derivations/`** — SymPy scripts verifying each numbered identity and bound (derivatives, convexity, fixed points, Lyapunov step, contraction, robustness).
- **`calculations/`** — Lightweight numerical demos (energy curves, \(T\)-iterations, robustness plots).

Typical files:
- `doc/golden_ratio.tex` — paper source
- `derivations/golden_ratio_verification.py` — symbolic checks
- `calculations/golden_ratio_calculations.py` — figure generation

---

## Getting Started

### Requirements
```bash
python -m pip install sympy numpy matplotlib
````

### Build the paper

Compile `doc/golden_ratio.tex`. If you use `hyperref` + `cleveref`, ensure:

* `\usepackage{hyperref}` **before** `\usepackage[nameinlink]{cleveref}` (cleveref last)
* Clean builds when switching options (`latexmk -C` or delete `.aux/.toc/.out`), then compile twice
* If math appears in section titles, prefer `\texorpdfstring{$\cdot$}{ascii}`

### Verify derivations

```bash
python derivations/golden_ratio_verification.py
```

Checks include:

* $E'_{a,b}(x)=a(x-1)-b/x$, $E''_{a,b}(x)=a+b/x^2>0$
* Minimizer $x_\star=\tfrac{1+\sqrt{1+4(b/a)}}{2}$
* For $a=b$: $E(Tx)\le E(x)$ with equality only at $x=\varphi$
* $(T^2)'(x)=1/(x^2T(x)^2)\le 1/4$ on $[3/2,2]$
* Robustness bound on a user-chosen $I=[1+\eta,X]$: $|x_\star-\varphi|\le \sqrt{2\Delta_I/m}$

### Reproduce figures

```bash
python calculations/golden_ratio_calculations.py
```

* Energy landscape with golden-ratio minimum
* $T$-iteration convergence (even–odd oscillations, geometric decay)
* Robustness: $|x^\ast-\varphi|$ vs perturbation strength (√-law)
* Twist-rate scaling

---

## Notes on variables & generalizations

* **Linear vs squared variables:** exact layering acts on $r=P/\xi_h$ via $\widehat T(r)=1+1/r$. Using $x=r^2$ induces $S(x)=(1+1/\sqrt{x})^2$. Results stated with $T$ on $x$ translate to $r$ (paper appendix gives details).
* **Metallic-means family:** for the energy $E_{a,b}$ with ratio $k=b/a$, pairing with the **matched** map $T_k(x)=k+1/x$ selects the corresponding metallic mean. In the normalized case $k=1$ we recover $\varphi$.

---

## Scope & Assumptions

* Short-range/local coarse-graining valid; clear scale separation (core $\xi_c\ll\xi_h\ll$ system size); weak anisotropy.
* Strong long-range interactions or hard boundaries can shift the optimum (captured by $\Delta_I$ in the robustness bound).
* No claims are made about topological protection or reconnection dynamics beyond what is proved.

---

## Citation

```bibtex
@article{golden_ratio_vortices,
  title   = {Golden Ratio from Energy Minimization and Self-Similarity in Hierarchical Vortices},
  author  = {Trevor Norris},
  year    = {2025},
  note    = {Preprint with verification scripts and figures at https://github.com/trevnorris/papers}
}
```

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).

## Contact

[trev.norris@gmail.com](mailto:trev.norris@gmail.com)

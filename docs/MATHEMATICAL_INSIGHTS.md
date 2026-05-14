# Key Mathematical Insights

## 1. SLERP = Grade-1 Projection of Rotor Sandwich

The most important identity in this work. For unit vectors $z_1, z_0 \in S^{d-1}$ with $z_1 \cdot z_0 = \cos\omega$:

**SLERP (trigonometric):**
$$\text{SLERP}(z_1, z_0, \alpha) = \frac{\sin((1-\alpha)\omega)}{\sin\omega} z_1 + \frac{\sin(\alpha\omega)}{\sin\omega} z_0$$

**Rotor sandwich (algebraic):**
$$z_t = R_\alpha \, z_1 \, \tilde{R}_\alpha \quad \text{where} \quad R_\alpha = \exp\!\left(\frac{\alpha}{2}(z_0 \wedge z_1)\right)$$

These are **algebraically identical**:
$$\langle R_\alpha \, z_1 \, \tilde{R}_\alpha \rangle_1 = \text{SLERP}(z_1, z_0, \alpha)$$

The grade-1 (vector) projection of the rotor sandwich recovers SLERP exactly. The rotor formulation additionally:
- Contains the bivector grade (rotation plane information) — discarded by SLERP
- Composes algebraically: $R_{\alpha_1} R_{\alpha_2} = R_{\alpha_1 + \alpha_2}$ (same plane)
- Avoids $\arccos$ and $\sin(\omega)$ divisions

## 2. Log-Map = Bivector Extraction

The Riemannian logarithmic map:
$$\log_{z_t}(z_1) = \frac{\omega}{\sin\omega}(z_1 - z_t \cos\omega)$$

is the grade-1 projection of the bivector field:
$$B_{z_t,z_1} = z_t \wedge z_1$$

$$\langle z_t \wedge z_1 \rangle_2 \xrightarrow{\text{grade projection}} \frac{\omega}{\sin\omega}(z_1 - z_t \cos\omega)$$

The bivector $B_{z_t,z_1}$ encodes the full rotation plane. The log-map extracts only the tangent vector component, losing the orientation of the plane (which direction to rotate in — this is recovered by the sign convention, but not structurally preserved).

## 3. Exp-Map = Rotor Application

$$\exp\_map(x, \delta) = x\cos\|\delta\| + \frac{\delta}{\|\delta\|}\sin\|\delta\|$$

is exactly:
$$\langle \exp(B_\delta) \, x \, \widetilde{\exp(B_\delta)} \rangle_1$$

where $B_\delta$ is the bivector such that $\langle B_\delta \times x \rangle_1 = \delta$ (bivector cross product producing the tangent vector).

Again: the trigonometric form is a projection of the algebraic rotor operation.

## 4. S-arch Residual = First-Order Rotor Approximation

S-FLM's residual connection:
$$h_{out} = \text{justnorm}(h_{in} + \gamma(B_{att} - h_{in}))$$

This approximates SLERP for small $\gamma$:
$$\text{SLERP}(h_{in}, B_{att}, \gamma) \approx h_{in} + \gamma(B_{att} - h_{in}) \quad \text{for } \gamma \ll 1$$

The exact form is the rotor:
$$R_\gamma = \exp\!\left(\frac{\gamma}{2}(B_{att} \wedge h_{in})\right)$$
$$h_{out} = R_\gamma \, h_{in} \, \tilde{R}_\gamma$$

## 5. Norm Preservation by Construction

For a rotor $R$ with $|R| = 1$ (i.e., $\langle R\tilde{R}\rangle_0 = 1$):
$$\|Rx\tilde{R}\|^2 = \langle Rx\tilde{R}Rx\tilde{R}\rangle_0 = \langle R x^2 \tilde{R}\rangle_0 = \langle R\tilde{R}\rangle_0 \cdot \langle x^2\rangle_0 = \|x\|^2$$

So the sandwich product preserves norm by construction. S-FLM needs `justnorm()` after every operation; rotor application doesn't need it.

## 6. Multivector Sphere Constraint

For a multivector $M \in \text{Cl}(k,0,0)$, the unit-norm constraint generalizes to:
$$\langle M\tilde{M}\rangle_0 = 1$$

This is the **spinor norm**. For a pure vector $v \in \mathbb{R}^k$, this reduces to $\|v\|^2 = 1$. The rotor sandwich $R M \tilde{R}$ preserves this constraint:
$$\langle R M \tilde{R} R \tilde{M} \tilde{R}\rangle_0 = \langle R M \tilde{M} \tilde{R}\rangle_0 = \langle R\tilde{R}\rangle_0 \cdot \langle M\tilde{M}\rangle_0 = 1$$

## 7. Clifford NVP Log-Determinant

From Alesiani & Maruyama (2024), scaling the transformed half of a multivector coupling layer by a scalar:
$$y_i^{l+1} = y_i^l \cdot \exp(s_{i,\theta}(x)) + t_{i,\phi}(x)$$

The log-determinant collapses to:
$$\ln|\det J| = \sum_i \langle s_{i,\theta}(x)\rangle_0$$

because $s$ is scalar-valued (embedded as grade-0). This is exactly the Real-NVP formula, but operating over multivector data. No special Jacobian computation needed.

## 8. Autograd Compatibility

From Clifford Flows Corollary 3.1: For any polynomial function $F$ over $\text{Cl}(k,0,0)$, the gradient computed by PyTorch Autograd (over the $\mathbb{R}^{2^k}$ coordinate representation) coincides with the analytical Clifford gradient.

**Implication:** We can use standard PyTorch backpropagation through our geometric product layers without custom autograd functions. The `einsum('...j,...k,jki->...i', A, B, cayley)` pattern is fully differentiable.

## 9. Cayley Tensor Precomputation

The geometric product in $\text{Cl}(k,0,0)$ is defined by the multiplication rule:
$$e_i e_j = \begin{cases} e_i \wedge e_j = e_{i \cup j} & \text{if } i \neq j \\ +1 & \text{if } i = j \text{ and } e_i^2 = +1 \end{cases}$$

For $\text{Cl}(k,0,0)$: all basis vectors square to $+1$. The full product table can be encoded as a rank-3 tensor $C_{ijk}$ where $C_{ijk} = 1$ if $e_i e_j$ has a component along $e_k$ with sign $+1$, etc.

For $k=8$: $C$ is $256 \times 256 \times 256 = 16.7M$ entries, but **extremely sparse** (each product has exactly one nonzero output grade). Storage: ~16.7M × 1 byte = ~17 MB. Precomputed once and registered as a buffer.

For $k=4$: $C$ is $16 \times 16 \times 16 = 4K$ entries. Negligible.
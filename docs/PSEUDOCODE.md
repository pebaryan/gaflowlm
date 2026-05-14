# Pseudocode: Rotor-Based Velocity Field and Sampling

## 1. Core Clifford Algebra Primitives

```python
# All operations in Cl(k, 0, 0) — Euclidean signature
# Multivector M stored as dict of grade -> tensor, or flat vector of 2^k coefficients
# For k=8, we use flat representation: 256 coefficients

import torch

def geometric_product(A: torch.Tensor, B: torch.Tensor, 
                      cayley: torch.Tensor) -> torch.Tensor:
    """
    Clifford geometric product AB in Cl(k,0,0).
    
    Args:
        A, B: [..., 2^k] multivector coefficients
        cayley: [2^k, 2^k, 2^k] Cayley tensor (precomputed)
                cayley[i,j,:] gives coefficients of e_i * e_j
    
    Returns:
        C: [..., 2^k] coefficients of AB
    """
    # AB_i = sum_j sum_k A_j * B_k * cayley[j, k, i]
    # Batched einsum: [...,j] x [...,k] x [j,k,i] -> [...,i]
    C = torch.einsum('...j,...k,jki->...i', A, B, cayley)
    return C


def reverse_mv(M: torch.Tensor, grade_signs: torch.Tensor) -> torch.Tensor:
    """
    Reverse of multivector: M̃.
    Grade-r component gets sign (-1)^(r(r-1)/2).
    
    Args:
        M: [..., 2^k] multivector
        grade_signs: [2^k] sign for each basis element
    Returns:
        M̃: [..., 2^k]
    """
    return M * grade_signs


def scalar_part(M: torch.Tensor, idx_0: int = 0) -> torch.Tensor:
    """Extract scalar (grade-0) component."""
    return M[..., idx_0:idx_0+1]


def grade_project(M: torch.Tensor, grade_mask: torch.Tensor) -> torch.Tensor:
    """Project to specific grade. grade_mask: [2^k] binary mask."""
    return M * grade_mask


def outer_product(A, B, cayley):
    """A ∧ B = (AB + BA) / 2 (anti-symmetric part for vectors;
    more generally: grade projection of AB."""
    AB = geometric_product(A, B, cayley)
    # For vectors a,b: a∧b = (ab - ba)/2
    # General: grade-project AB to grade(a) + grade(b)
    # Simplified: use the grade-raising formula
    BA = geometric_product(B, A, cayley)
    return (AB - BA) / 2  # For vector arguments


def inner_product(A, B, cayley):
    """A · B = (AB + BA) / 2 for vectors (symmetric part).
    More generally: grade-project AB to |grade(a) - grade(b)|."""
    AB = geometric_product(A, B, cayley)
    BA = geometric_product(B, A, cayley)
    return (AB + BA) / 2


def bivector_from_vectors(a: torch.Tensor, b: torch.Tensor, 
                          cayley: torch.Tensor) -> torch.Tensor:
    """
    Compute bivector B = (1/2) a ∧ b from unit vectors a, b.
    This encodes the rotation plane from b to a.
    """
    return outer_product(a, b, cayley) / 2


def rotor_exp(B: torch.Tensor, cayley: torch.Tensor,
              grade2_mask: torch.Tensor) -> torch.Tensor:
    """
    Exponential of a bivector B: R = exp(B).
    For a simple bivector B = θ ẑ (where ẑ is a unit 2-blade):
        R = cos(θ) + ẑ sin(θ)
    
    General bivector (sum of commuting blades):
        Decompose into eigen-blades, exponentiate each.
    
    Simplified implementation:
        B² = B · B (scalar, via geometric product with itself)
        R = cos(√|B²|) + B/√|B²| · sin(√|B²|)   [if B² < 0]
        R = 1 + B                                    [if B² ≈ 0, first-order]
    """
    # B² = B*B (geometric product of B with itself)
    BB = geometric_product(B, B, cayley)
    # Scalar part of B² (for bivectors in Cl(k,0,0), B² is negative)
    B_sq = scalar_part(BB)  # [..., 1]
    
    # For Euclidean GA: bivectors square to negative scalars
    # B² = -θ² => θ = √(-B²)
    theta = torch.sqrt(-B_sq.clamp(max=0) + 1e-8)
    theta_safe = theta.clamp(min=1e-8)
    
    # R = cos(θ) + (B/θ) sin(θ)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    R = cos_theta * grade2_mask  # scalar part: cos(θ)
    R = R + (sin_theta / theta_safe) * B  # bivector part: (B/θ) sin(θ)
    
    return R


def rotor_apply(R: torch.Tensor, x: torch.Tensor,
                cayley: torch.Tensor, 
                grade_signs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotor R to multivector x via sandwich product:
    x' = R x R̃
    
    This preserves:
    - Norm: |x'| = |x| when |R| = 1
    - Grade: grade(x') = grade(x) for pure-grade inputs
    """
    R_rev = reverse_mv(R, grade_signs)
    Rx = geometric_product(R, x, cayley)
    RxRrev = geometric_product(Rx, R_rev, cayley)
    return RxRrev


def commutator(A, B, cayley):
    """[A, B] = AB - BA (Lie bracket / commutator product)"""
    AB = geometric_product(A, B, cayley)
    BA = geometric_product(B, A, cayley)
    return AB - BA
```

## 2. Rotor Forward Process (Replaces SLERP)

```python
def rotor_noise_process(z1: torch.Tensor,     # [B, L, d] clean embeddings
                        noise: torch.Tensor,    # [B, L, d] uniform noise on sphere
                        alpha_t: float,         # noise level (0=clean, 1=noise)
                        clifford: CliffordEngine) -> torch.Tensor:
    """
    Forward (noise) process using rotors.
    Replaces SLERP(z1, z0, alpha_t).
    
    Algebraic equivalence:
        SLERP: z_t = sin((1-α)ω)/sin(ω) · z1 + sin(αω)/sin(ω) · z0
        Rotor: z_t = R_{α} z1 R̃_{α}  where R_{α} = exp(α/2 · (z0 ∧ z1))
    
    The rotor formulation:
    - No acos() computation
    - No division by sin(ω) 
    - Norm preserved by construction
    - Composes: R_{α₂}R_{α₁} = R_{α₁+α₂} (for same plane)
    """
    # Project to Clifford space
    z1_cl = clifford.embed_to_cl(z1)    # [B, L, 2^k]
    z0_cl = clifford.embed_to_cl(noise)  # [B, L, 2^k]
    
    # Compute rotation bivector: B = (1/2) z0 ∧ z1
    B = bivector_from_vectors(z0_cl, z1_cl, clifford.cayley)
    
    # Rotor at noise level alpha_t: R = exp(α_t · B)
    R_alpha = rotor_exp(alpha_t * B, clifford.cayley, clifford.grade2_mask)
    
    # Apply sandwich product: z_t = R z1 R̃
    zt_cl = rotor_apply(R_alpha, z1_cl, clifford.cayley, clifford.grade_signs)
    
    # Project back to embedding space
    zt = clifford.cl_to_embed(zt_cl)    # [B, L, d]
    
    return zt
```

## 3. Rotor Velocity Field (Replaces Log-Map)

```python
def rotor_velocity_field(zt: torch.Tensor,        # [B, L, d] current state
                         E: torch.Tensor,           # [B, L, V, d] all embeddings
                         p_1_given_t: torch.Tensor,  # [B, L, V] posterior probs
                         clifford: CliffordEngine,
                         top_k: int = -1) -> torch.Tensor:
    """
    Compute the bivector velocity field on the multivector sphere.
    Replaces v_t = (α̇/(1-α)) Σ_v p(v|z_t) log_{z_t}(ê_v)
    
    With rotors:
        v_t (bivector) = (α̇/(1-α)) Σ_v p(v|z_t) · (z_t ∧ ê_v)
    
    The outer product z_t ∧ ê_v directly gives the rotation plane
    from z_t toward ê_v — no acos, no log-map.
    """
    B, L, d = zt.shape
    V = E.shape[2]
    
    # Optional: top-k filtering for efficiency
    if top_k > 0:
        topk_vals, topk_idx = p_1_given_t.topk(top_k, dim=-1)
        p_filtered = torch.zeros_like(p_1_given_t)
        p_filtered.scatter_(-1, topk_idx, topk_vals)
        p_filtered = p_filtered / p_filtered.sum(dim=-1, keepdim=True)
        p_1_given_t = p_filtered
    
    # Scale factor (same as S-FLM)
    alpha_t = ...  # from schedule
    alpha_dot = ... # derivative
    scale = alpha_dot / (1 - alpha_t).clamp(min=1e-8)
    
    # Compute bivector field: weighted sum of rotation planes
    # For each vocabulary token v: B_v = z_t ∧ ê_v
    # Marginal: B_θ = scale * Σ_v p(v|z_t) * B_v
    
    # Project to Clifford space
    zt_cl = clifford.embed_to_cl(zt)         # [B, L, 2^k]
    E_cl = clifford.embed_to_cl(E.reshape(B*L*V, d))  # [B*L*V, 2^k]
    E_cl = E_cl.reshape(B, L, V, -1)        # [B, L, V, 2^k]
    
    # Bivectors for each token: outer product z_t ∧ ê_v
    # z_t: [B, L, 1, 2^k], E: [B, L, V, 2^k]
    zt_expanded = zt_cl.unsqueeze(2)         # [B, L, 1, 2^k]
    bivectors = outer_product(
        zt_expanded.expand_as(E_cl), E_cl, clifford.cayley
    )  # [B, L, V, 2^k]
    
    # Weight by posterior: [B, L, V, 1] * [B, L, V, 2^k] -> [B, L, V, 2^k]
    p_expanded = p_1_given_t.unsqueeze(-1)    # [B, L, V, 1]
    B_theta = (p_expanded * bivectors).sum(dim=2)  # [B, L, 2^k]
    B_theta = scale * B_theta
    
    return B_theta  # Predicted bivector field
```

## 4. Rotor Sampling Step (Replaces Exp-Map)

```python
def rotor_euler_step(zt: torch.Tensor,      # [B, L, d] current state
                     B_theta: torch.Tensor,  # [B, L, 2^k] predicted bivector
                     dt: float,              # step size
                     clifford: CliffordEngine) -> torch.Tensor:
    """
    Euler step on the multivector sphere using rotor composition.
    Replaces z_{t+Δt} = exp_map(z_t, h · v_t).
    
    With rotors:
        R_Δt = exp(B_θ · Δt)
        z_{t+Δt} = R_Δt z_t R̃_Δt
    
    Advantages:
    - No explicit normalization (rotor preserves norm)
    - Rotor composition: R_3 = R_2 R_1 (algebraic, not approximate)
    - Geodesic by construction (rotor exponential is exact geodesic)
    """
    # Rotor from predicted bivector
    R_dt = rotor_exp(B_theta * dt, clifford.cayley, clifford.grade2_mask)
    
    # Apply sandwich product
    zt_cl = clifford.embed_to_cl(zt)
    zt_next_cl = rotor_apply(R_dt, zt_cl, clifford.cayley, clifford.grade_signs)
    
    # Project back
    zt_next = clifford.cl_to_embed(zt_next_cl)
    
    return zt_next
```

## 5. CFA Attention Block (Variant B)

```python
class CliffordFrameAttentionBlock(nn.Module):
    """
    Clifford Frame Attention for language modeling.
    Adapted from GAFL's CFA for protein design.
    """
    def __init__(self, k=8, n_heads=8, d_model=768):
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        self.dim_mv = 2**k  # multivector dimension
        
        # Multivector linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, n_heads * self.dim_mv)
        self.k_proj = nn.Linear(d_model, n_heads * self.dim_mv)
        self.v_proj = nn.Linear(d_model, n_heads * self.dim_mv)
        
        # Geometric bilinear: separate inner and outer product pathways
        self.inner_gate = nn.Linear(d_model, n_heads)  # gate for A·B pathway
        self.outer_gate = nn.Linear(d_model, n_heads)  # gate for A∧B pathway
        
        # Output projection
        self.out_proj = nn.Linear(n_heads * self.dim_mv, d_model)
        
        # Higher-order message composition
        self.use_higher_order = True
        
        # Cayley tensor (precomputed, not learned)
        self.register_buffer('cayley', build_cayley_tensor(k))
        self.register_buffer('grade_signs', build_grade_signs(k))
        
        # Rotor residual gate
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x_mv, t_emb):
        """
        Args:
            x_mv: [B, L, dim_mv] multivector representations
            t_emb: [B, d_model] time embedding
        Returns:
            out: [B, L, dim_mv] updated multivector representations
        """
        B, L, _ = x_mv.shape
        
        # Q, K, V projections (multivector-valued)
        Q = self.q_proj(x_mv).reshape(B, L, self.n_heads, self.dim_mv)
        K = self.k_proj(x_mv).reshape(B, L, self.n_heads, self.dim_mv)
        V = self.v_proj(x_mv).reshape(B, L, self.n_heads, self.dim_mv)
        
        # Attention weights: scalar part of geometric product Q*K
        # Q_i · K_j -> scalar = "inner product" = cosine similarity in GA
        QK = torch.einsum('blhd,bmhd->blmh', Q, K)  # placeholder
        # More precisely: scalar_part(geometric_product(Q_i, K_j))
        # For efficiency: use grade-0 projection
        attn_logits = self._scalar_product(Q, K)  # [B, L, L, H]
        attn_weights = F.softmax(attn_logits / math.sqrt(self.dim_mv), dim=2)
        
        # Geometric bilinear messages
        # m_{ij} = (Q_i · V_j) * gate_inner + (Q_i ∧ V_j) * gate_outer
        inner_msg = self._inner_product_path(Q, V)   # grade-lowering
        outer_msg = self._outer_product_path(Q, V)   # grade-raising
        
        gate_inner = torch.sigmoid(self.inner_gate(x_mv))
        gate_outer = torch.sigmoid(self.outer_gate(x_mv))
        
        messages = gate_inner.unsqueeze(-1) * inner_msg + \
                   gate_outer.unsqueeze(-1) * outer_msg
        
        # Weighted aggregation
        attn_exp = attn_weights.permute(0, 3, 1, 2)  # [B, H, L, L]
        msg_perm = messages.permute(0, 2, 1, 3, 4)    # [B, H, L_src, L_tgt, dim_mv]
        
        # Aggregate: o_i = Σ_j a_{ij} m_{ij}
        output = torch.einsum('bhls,bhltd->bhtd', attn_exp, msg_perm)
        output = output.reshape(B, L, -1)
        output = self.out_proj(output)
        
        # Higher-order message passing (3-body from 2-body)
        if self.use_higher_order:
            M2_agg = messages.sum(dim=2)  # aggregate over j: [B, L, H, dim_mv]
            # Geometric product of two 2-body aggregates
            M3 = geometric_product(M2_agg, M2_agg, self.cayley)
            # Add to output (projected)
            output = output + self.ho_proj(M3.reshape(B, L, -1))
        
        # Rotor residual connection
        # Instead of x + gamma * output, use:
        # R_gamma = exp(gamma * bivector_residual), then sandwich product
        B_residual = bivector_from_vectors(
            self.embed_to_cl(output), self.embed_to_cl(x_mv), self.cayley
        )
        R_gamma = rotor_exp(self.gamma * B_residual, self.cayley, self.grade2_mask)
        x_mv_out = rotor_apply(R_gamma, x_mv, self.cayley, self.grade_signs)
        
        # Multivector normalization
        x_mv_out = normalize_multivector(x_mv_out)
        
        return x_mv_out
```

## 6. Full Training Loop

```python
def train_step(model, batch, clifford, noise_schedule, optimizer):
    """Single training step for GAFlowLM."""
    x0 = batch['input_ids']  # [B, L]
    
    # 1. Get sphere-normalized embeddings
    E = model.backbone.get_sphere_embeddings(x0)  # [B, L, d], unit norm
    
    # 2. Sample noise level
    t = torch.rand(x0.shape[0], device=x0.device)
    alpha_t, alpha_dot_t = noise_schedule(t)
    
    # 3. Sample noise on sphere
    noise = sample_uniform_sphere(x0.shape, device=x0.device)
    
    # 4. Forward process (rotor-based, not SLERP)
    #    B = (1/2) noise ∧ E  (rotation bivector)
    #    R_{alpha_t} = exp(alpha_t * B)
    #    z_t = R_{alpha_t} E R̃_{alpha_t}
    E_cl = clifford.embed_to_cl(E)
    noise_cl = clifford.embed_to_cl(noise)
    B = bivector_from_vectors(noise_cl, E_cl, clifford.cayley)
    R_t = rotor_exp(alpha_t.unsqueeze(-1) * B, clifford.cayley, clifford.grade2_mask)
    zt_cl = rotor_apply(R_t, E_cl, clifford.cayley, clifford.grade_signs)
    zt = clifford.cl_to_embed(zt_cl)
    
    # 5. Forward pass through backbone
    logits = model(zt, sigma=t)  # [B, L, V]
    
    # 6. Cross-entropy loss (same as S-FLM)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x0.reshape(-1))
    
    # 7. Backward + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 8. [Optional] Renormalize embeddings
    with torch.no_grad():
        model.backbone.sphere_embed.weight.data = \
            F.normalize(model.backbone.sphere_embed.weight.data, dim=1)
    
    return loss.item()
```

## 7. Full Sampling Loop

```python
@torch.no_grad()
def sample(model, clifford, noise_schedule, 
           shape, n_steps=128, velocity='exact', top_k_velocity=1):
    """Generate tokens via rotor-based flow sampling."""
    B, L, d = shape
    
    # 1. Start from uniform noise on sphere
    z = sample_uniform_sphere((B, L, d))
    
    # 2. Precompute schedule
    t_schedule = torch.linspace(0, 1, n_steps + 1)
    
    # 3. Euler integration with rotors
    E_all = clifford.embed_to_cl(
        model.backbone.get_all_embeddings()
    )  # [V, 2^k]
    
    for step in range(n_steps):
        t_now = t_schedule[step]
        t_next = t_schedule[step + 1]
        alpha_t, alpha_dot_t = noise_schedule(t_now)
        alpha_s, _ = noise_schedule(t_next)
        
        # Compute step size
        h = (alpha_t - alpha_s) / alpha_t.clamp(min=1e-8)
        
        # Predict logits
        z_embed = clifford.cl_to_embed(z) if z.is_clifford else z
        logits = model(z_embed, sigma=t_now)  # [B, L, V]
        
        # Compute posterior
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        
        # Top-k velocity
        if top_k_velocity > 0:
            topk_vals, topk_idx = p.topk(top_k_velocity, dim=-1)
            p = torch.zeros_like(p).scatter_(-1, topk_idx, topk_vals)
            p = p / p.sum(dim=-1, keepdim=True)
        
        # Compute bivector velocity field
        z_cl = clifford.embed_to_cl(z_embed)  # [B, L, 2^k]
        E_expanded = E_all.unsqueeze(0).unsqueeze(0)  # [1, 1, V, 2^k]
        bivectors = outer_product(
            z_cl.unsqueeze(2).expand(-1, -1, len(E_all), -1),
            E_expanded.expand(B, L, -1, -1),
            clifford.cayley
        )  # [B, L, V, 2^k]
        
        B_theta = (p.unsqueeze(-1) * bivectors).sum(dim=2)  # [B, L, 2^k]
        
        # Rotor Euler step
        R_step = rotor_exp(h * B_theta, clifford.cayley, clifford.grade2_mask)
        z_cl = rotor_apply(R_step, z_cl, clifford.cayley, clifford.grade_signs)
        z = clifford.cl_to_embed(z_cl)
    
    # 4. Decode
    logits = model(z, sigma=torch.tensor(1.0))
    token_ids = logits.argmax(dim=-1)  # [B, L]
    
    return token_ids
```
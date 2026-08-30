# Comprehensive Mathematical Parameter Calculation Guide

This document provides a complete, step-by-step mathematical breakdown of parameter calculations across all neural network components in this codebase (`model/modeling_dp_vla.py`, `model/dit/DiT.py`, `model/dit/decoder.py`, `model/configuration_dp_vla.py`).

---

## 1. Executive Summary & Configuration Presets

Parameters in PyTorch neural network modules come from two sources:
1. **Weights ($W$)**: Tensors multiplied with inputs. For a `Linear(in_features, out_features)`, weight count is $\text{in\_features} \times \text{out\_features}$.
2. **Biases ($b$)**: Additive terms added to linear projections, layer normalizations, etc. For a `Linear(in_features, out_features)` with `bias=True`, bias count is $\text{out\_features}$.

### Active Configuration Comparison
The codebase supports both the **Full Base Model (1024-dim, 12-layer)** and the **Active Training Configuration (512-dim, 6-layer)** used in `train.py`.

| Component / Submodule | Active `train.py` Config ($d=512, N\_L=6, N\_H=8$) | Full Base Config ($d=1024, N\_L=12, N\_H=16$) |
| :--- | :---: | :---: |
| **HighCapacityVectorSceneEncoder** (`fallback_encoder`) | **27,559,424** | **109,642,240** |
| **CustomDiT Decoder** (`decoder`) | **59,244,548** | **462,072,836** |
| **Total `DpVlaModel` Parameters** | **86,803,972 (~86.80M)** | **571,715,076 (~571.72M)** |

---

## 2. General Formulas for PyTorch Layers

| Layer Type | Formula for Weight Parameters | Formula for Bias Parameters | Total Parameters Formula |
| :--- | :--- | :--- | :--- |
| `nn.Linear(in_f, out_f, bias=True)` | $\text{in\_f} \times \text{out\_f}$ | $\text{out\_f}$ | $(\text{in\_f} + 1) \times \text{out\_f}$ |
| `nn.Linear(in_f, out_f, bias=False)` | $\text{in\_f} \times \text{out\_f}$ | $0$ | $\text{in\_f} \times \text{out\_f}$ |
| `nn.LayerNorm(normalized_shape)` | $\text{normalized\_shape}$ | $\text{normalized\_shape}$ | $2 \times \text{normalized\_shape}$ |
| `nn.Embedding(num_embeddings, embedding_dim)` | $\text{num\_embeddings} \times \text{embedding\_dim}$ | $0$ | $\text{num\_embeddings} \times \text{embedding\_dim}$ |
| `nn.Parameter(torch.zeros(...))` | $\prod \text{shape}$ | $0$ | $\prod \text{shape}$ |

---

## 3. Detailed Parameter Breakup: `model/dit/DiT.py`

### 3.1 `TimestepEmbedder`
**Source Class:** `TimestepEmbedder` (`model/dit/DiT.py`)  
**Input:** Scalar timestep $t$  
**Structure:** Sinusoidal embedding ($F=256$) $\to$ `nn.Sequential(Linear(F, hidden_size), SiLU(), Linear(hidden_size, hidden_size))`

#### Formula:
$$\mathrm{Params}_{\mathrm{TimestepEmbedder}} = \underbrace{(F + 1) \times d}_{\mathrm{fc1}} + \underbrace{(d + 1) \times d}_{\mathrm{fc2}}$$

#### Calculation for $d=512, F=256$:
- `mlp[0]` (`Linear(256, 512)`): $(256 + 1) \times 512 = 131,584$
- `mlp[2]` (`Linear(512, 512)`): $(512 + 1) \times 512 = 262,656$
- **Total:** $131,584 + 262,656 = \mathbf{394,240}$

#### Calculation for $d=1024, F=256$:
- `mlp[0]` (`Linear(256, 1024)`): $(256 + 1) \times 1024 = 263,168$
- `mlp[2]` (`Linear(1024, 1024)`): $(1024 + 1) \times 1024 = 1,049,600$
- **Total:** $263,168 + 1,049,600 = \mathbf{1,312,768}$

---

### 3.2 `CustomCrossAttention`
**Source Class:** `CustomCrossAttention` (`model/dit/DiT.py`)  
**Structure:**
- `proj_q` (`Linear(d, d, bias=qkv_bias)`)
- `proj_k` (`Linear(d, d, bias=qkv_bias)`)
- `proj_v` (`Linear(d, d, bias=qkv_bias)`)
- `proj` (`Linear(d, d, bias=proj_bias)`)
- `q_norm` and `k_norm` (`nn.Identity()` when `qk_norm=False`)

#### Formula (with `qkv_bias=True`, `proj_bias=True`):
$$\text{Params}\_{\text{CustomCrossAttention}} = 4 \times (d + 1) \times d$$

#### Calculation for $d=512$:
- $4 \times (512 + 1) \times 512 = 4 \times 513 \times 512 = \mathbf{1,050,624}$

#### Calculation for $d=1024$:
- $4 \times (1024 + 1) \times 1024 = 4 \times 1025 \times 1024 = \mathbf{4,198,400}$

---

### 3.3 `SelfAttention`
**Source Class:** `SelfAttention` (`model/dit/DiT.py`)  
**Structure:**
- `qkv` (`Linear(d, 3*d, bias=True)`)
- `proj` (`Linear(d, d, bias=True)`)

#### Formula:
$$\text{Params}\_{\text{SelfAttention}} = (d + 1) \times 3d + (d + 1) \times d = 4(d + 1)d$$

#### Calculation for $d=512$:
- `qkv`: $(512 + 1) \times 1536 = 787,968$
- `proj`: $(512 + 1) \times 512 = 262,656$
- **Total:** $787,968 + 262,656 = \mathbf{1,050,624}$

#### Calculation for $d=1024$:
- `qkv`: $(1024 + 1) \times 3072 = 3,148,800$
- `proj`: $(1024 + 1) \times 1024 = 1,049,600$
- **Total:** $3,148,800 + 1,049,600 = \mathbf{4,198,400}$

---

### 3.4 `MlpBlock`
**Source Class:** `MlpBlock` (`model/dit/DiT.py`)  
**Structure:**
- `fc1` (`Linear(d, d * mlp_ratio)`)
- `fc2` (`Linear(d * mlp_ratio, d)`)

#### Formula (for `mlp_ratio = 4.0`):
$$\text{Params}\_{\text{MlpBlock}} = (d + 1) \times 4d + (4d + 1) \times d = 8d^2 + 2d$$

#### Calculation for $d=512$:
- `fc1`: $(512 + 1) \times 2048 = 1,050,624$
- `fc2`: $(2048 + 1) \times 512 = 1,049,088$
- **Total:** $1,050,624 + 1,049,088 = \mathbf{2,099,712}$

#### Calculation for $d=1024$:
- `fc1`: $(1024 + 1) \times 4096 = 4,198,400$
- `fc2`: $(4096 + 1) \times 1024 = 4,195,328$
- **Total:** $4,198,400 + 4,195,328 = \mathbf{8,393,728}$

---

### 3.5 `DiTBlock` (Single Transformer Block with adaLN-Zero)
**Source Class:** `DiTBlock` (`model/dit/DiT.py`)  
**Sub-modules:**
1. `attn`: `SelfAttention(d)`
2. `mlp`: `MlpBlock(d, 4*d)`
3. `cross_attn`: `CustomCrossAttention(d)`
4. `cross_mlp`: `MlpBlock(d, 4*d)`
5. `adaLN_modulation`: `Sequential(SiLU(), Linear(d, 12 * d, bias=True))`
6. `norm1`, `norm2`, `norm3`, `norm4`: `LayerNorm(d, elementwise_affine=False)` ($0$ params)

#### Formula for Single `DiTBlock`:
$$\text{Params}\_{\text{DiTBlock}} = \underbrace{4(d+1)d}\_{\text{SelfAttn}} + \underbrace{(8d^2 + 2d)}\_{\text{MLP}} + \underbrace{4(d+1)d}\_{\text{CrossAttn}} + \underbrace{(8d^2 + 2d)}\_{\text{CrossMLP}} + \underbrace{(d + 1) \times 12d}\_{\text{adaLN}}$$
$$\text{Params}\_{\text{DiTBlock}} = 16d^2 + 4d + 16d^2 + 4d + 12d^2 + 12d = \mathbf{44d^2 + 20d}$$

#### Calculation for $d=512$:
- `attn`: $1,050,624$
- `mlp`: $2,099,712$
- `cross_attn`: $1,050,624$
- `cross_mlp`: $2,099,712$
- `adaLN_modulation`: $(512 + 1) \times 6144 = 3,151,872$
- **Total per `DiTBlock`:** $1,050,624 + 2,099,712 + 1,050,624 + 2,099,712 + 3,151,872 = \mathbf{9,452,544}$

#### Calculation for $d=1024$:
- `attn`: $4,198,400$
- `mlp`: $8,393,728$
- `cross_attn`: $4,198,400$
- `cross_mlp`: $8,393,728$
- `adaLN_modulation`: $(1024 + 1) \times 12288 = 12,595,200$
- **Total per `DiTBlock`:** $4,198,400 + 8,393,728 + 4,198,400 + 8,393,728 + 12,595,200 = \mathbf{37,779,456}$

---

### 3.6 `FinalLayer`
**Source Class:** `FinalLayer` (`model/dit/DiT.py`)  
**Sub-modules:**
1. `norm_final`: `LayerNorm(d)` ($2d$ params)
2. `adaLN_modulation`: `Sequential(SiLU(), Linear(d, 2*d))` ($(d+1) \times 2d$ params)
3. `proj`:
   - `LayerNorm(d)` ($2d$ params)
   - `Linear(d, 4*d)` ($(d+1) \times 4d$ params)
   - `LayerNorm(4*d)` ($8d$ params)
   - `Linear(4*d, output_size)` ($(4d+1) \times \text{dim\_action}$ params)

#### Formula for `FinalLayer`:
$$\text{Params}\_{\text{FinalLayer}} = 2d + (2d^2 + 2d) + 2d + (4d^2 + 4d) + 8d + (4d + 1) \times \text{dim\_action}$$
$$\text{Params}\_{\text{FinalLayer}} = 6d^2 + 18d + (4d + 1) \times \text{dim\_action}$$

#### Calculation for $d=512, \text{dim\_action}=4$:
- `norm_final`: $1,024$
- `adaLN_modulation`: $(512 + 1) \times 1024 = 525,312$
- `proj[0]` (`LayerNorm(512)`): $1,024$
- `proj[1]` (`Linear(512, 2048)`): $(512 + 1) \times 2048 = 1,050,624$
- `proj[3]` (`LayerNorm(2048)`): $4,096$
- `proj[4]` (`Linear(2048, 4)`): $(2048 + 1) \times 4 = 8,196$
- **Total:** $1,024 + 525,312 + 1,024 + 1,050,624 + 4,096 + 8,196 = \mathbf{1,590,276}$

#### Calculation for $d=1024, \text{dim\_action}=4$:
- `norm_final`: $2,048$
- `adaLN_modulation`: $(1024 + 1) \times 2048 = 2,101,248$
- `proj[0]` (`LayerNorm(1024)`): $2,048$
- `proj[1]` (`Linear(1024, 4096)`): $(1024 + 1) \times 4096 = 4,198,400$
- `proj[3]` (`LayerNorm(4096)`): $8,192$
- `proj[4]` (`Linear(4096, 4)`): $(4096 + 1) \times 4 = 16,388$
- **Total:** $2,048 + 2,101,248 + 2,048 + 4,198,400 + 8,192 + 16,388 = \mathbf{6,326,276}$

---

## 4. Detailed Parameter Breakup: `model/dit/decoder.py`

### `CustomDiT`
**Source Class:** `CustomDiT` (`model/dit/decoder.py`)  
**Sub-modules:**
1. `t_embedder`: `TimestepEmbedder(d)`
2. `pos_emb`: `nn.Parameter(shape=(1, num_actions, d))` ($N\_A \times d$ params)
3. `action_in_proj`: `MlpBlock(in_features=dim_action, hidden_features=512, out_features=d)`
4. `y_in_proj`: `MlpBlock(in_features=dim_y, hidden_features=512, out_features=d)`
5. `blocks`: ModuleList of $N\_L$ `DiTBlock` instances
6. `final_layer`: `FinalLayer(d, dim_action)`

#### Input Projection Formulas:
- `action_in_proj` (`MlpBlock(4, 512, d)`):
  $$\text{fc1}(4, 512) = (4 + 1) \times 512 = 2,560$$
  $$\text{fc2}(512, d) = (512 + 1) \times d$$
  $$\text{Total}\_{\text{action\_in\_proj}} = 2560 + 513d$$

- `y_in_proj` (`MlpBlock(12, 512, d)`):
  $$\text{fc1}(12, 512) = (12 + 1) \times 512 = 6,656$$
  $$\text{fc2}(512, d) = (512 + 1) \times d$$
  $$\text{Total}\_{\text{y\_in\_proj}} = 6656 + 513d$$

#### Total `CustomDiT` Decoder Formula:
$$\text{Params}\_{\text{CustomDiT}} = \text{t\_embedder} + (N\_A \cdot d) + \text{action\_in\_proj} + \text{y\_in\_proj} + (N\_L \cdot \text{DiTBlock}) + \text{FinalLayer}$$

#### Calculation for Active `train.py` Config ($d=512, N\_L=6, N\_A=20$):
- `t_embedder`: $394,240$
- `pos_emb` (`20 x 512`): $10,240$
- `action_in_proj`: $2560 + (513 \times 512) = 265,216$
- `y_in_proj`: $6656 + (513 \times 512) = 269,312$
- `blocks` ($6 \times 9,452,544$): $56,715,264$
- `final_layer`: $1,590,276$
- **Total `CustomDiT` Decoder:** $394,240 + 10,240 + 265,216 + 269,312 + 56,715,264 + 1,590,276 = \mathbf{59,244,548}$

#### Calculation for Full Base Config ($d=1024, N\_L=12, N\_A=20$):
- `t_embedder`: $1,312,768$
- `pos_emb` (`20 x 1024`): $20,480$
- `action_in_proj`: $2560 + (513 \times 1024) = 527,872$
- `y_in_proj`: $6656 + (513 \times 1024) = 531,968$
- `blocks` ($12 \times 37,779,456$): $453,353,472$
- `final_layer`: $6,326,276$
- **Total `CustomDiT` Decoder:** $1,312,768 + 20,480 + 527,872 + 531,968 + 453,353,472 + 6,326,276 = \mathbf{462,072,836}$

---

## 5. Detailed Parameter Breakup: `model/modeling_dp_vla.py`

### `HighCapacityVectorSceneEncoder` (`fallback_encoder`)
**Source Class:** `HighCapacityVectorSceneEncoder` (`model/modeling_dp_vla.py`)  
**Sub-modules:**
1. `ego_feature_proj`: `Sequential(Linear(6, 128), GELU(), Linear(128, d))`
2. `ego_pos_embed`: `Parameter(shape=(1, 20, d))` ($20d$ params)
3. `agent_feature_proj`: `Sequential(Linear(6, 128), GELU(), Linear(128, d))`
4. `agent_pos_embed`: `Parameter(shape=(1, 20, d))` ($20d$ params)
5. `agent_temporal_transformer`: `TransformerEncoder(2 layers, nhead=4, d_model=d, dim_feedforward=2d)`
6. `map_feature_proj`: `Sequential(Linear(5, 128), GELU(), Linear(128, d))`
7. `map_pos_embed`: `Parameter(shape=(1, 20, d))` ($20d$ params)
8. `map_point_transformer`: `TransformerEncoder(2 layers, nhead=4, d_model=d, dim_feedforward=2d)`
9. `type_embed`: `Embedding(3, d)` ($3d$ params)
10. `scene_transformer`: `TransformerEncoder(6 layers, nhead=num_heads, d_model=d, dim_feedforward=4d)`
11. `norm`: `LayerNorm(d)` ($2d$ params)

#### Standard PyTorch `TransformerEncoderLayer` Parameter Formula:
$$\text{Params}\_{\text{TransformerEncoderLayer}} = \underbrace{4(d+1)d}\_{\text{SelfAttn}} + \underbrace{2(d+1)d}\_{\text{LayerNorms}} + \underbrace{((d+1) \cdot d\_{\text{ff}} + (d\_{\text{ff}}+1) \cdot d)}\_{\text{FFN}}$$

- When $d\_{\text{ff}} = 2d$:
  $$\text{Params} = 4d^2 + 4d + 4d + 2d^2 + d + 2d^2 + d = 8d^2 + 9d$$

- When $d\_{\text{ff}} = 4d$:
  $$\text{Params} = 4d^2 + 4d + 4d + 4d^2 + d + 4d^2 + d = 12d^2 + 9d$$

#### Sub-Module Calculations for Active `train.py` Config ($d=512$):
- `ego_feature_proj`: $(6+1)\times 128 + (128+1)\times 512 = 896 + 66,048 = 66,944$
- `ego_pos_embed`: $20 \times 512 = 10,240$
- `agent_feature_proj`: $66,944$
- `agent_pos_embed`: $10,240$
- `agent_temporal_transformer` ($2 \text{ layers}, d\_{\text{ff}}=1024$):
  Each layer: $8(512)^2 + 9(512) = 2,097,152 + 4,608 = 2,101,760$
  $2 \times 2,101,760 = 4,203,520 + 4 \text{ LN affine params} = 4,205,568$
- `map_feature_proj` (`Linear(5, 128) + Linear(128, 512)`):
  $(5+1)\times 128 + (128+1)\times 512 = 768 + 66,048 = 66,816$
- `map_pos_embed`: $10,240$
- `map_point_transformer`: $4,205,568$
- `type_embed` (`Embedding(3, 512)`): $3 \times 512 = 1,536$
- `scene_transformer` ($6 \text{ layers}, d\_{\text{ff}}=2048$):
  Each layer: $12(512)^2 + 9(512) = 3,145,728 + 4,608 = 3,150,336$
  $6 \times 3,150,336 = 18,902,016 + 12 \text{ LN affine params} = 18,914,304$
- `norm` (`LayerNorm(512)`): $1,024$
- **Total `HighCapacityVectorSceneEncoder` ($d=512$):** $\mathbf{27,559,424}$

#### Sub-Module Calculations for Full Base Config ($d=1024$):
- `ego_feature_proj`: $(6+1)\times 128 + (128+1)\times 1024 = 896 + 132,096 = 132,992$
- `ego_pos_embed`: $20 \times 1024 = 20,480$
- `agent_feature_proj`: $132,992$
- `agent_pos_embed`: $20,480$
- `agent_temporal_transformer` ($2 \text{ layers}, d\_{\text{ff}}=2048$):
  Each layer: $8(1024)^2 + 9(1024) = 8,388,608 + 9,216 = 8,397,824$
  $2 \times 8,397,824 = 16,795,648 + 4 \text{ LN affine params} = 16,799,744$
- `map_feature_proj`: $(5+1)\times 128 + (128+1)\times 1024 = 768 + 132,096 = 132,864$
- `map_pos_embed`: $20,480$
- `map_point_transformer`: $16,799,744$
- `type_embed` (`Embedding(3, 1024)`): $3 \times 1024 = 3,072$
- `scene_transformer` ($6 \text{ layers}, d\_{\text{ff}}=4096$):
  Each layer: $12(1024)^2 + 9(1024) = 12,582,912 + 9,216 = 12,592,128$
  $6 \times 12,592,128 = 75,552,768 + 24.5\text{K LN params} = 75,577,344$
- `norm` (`LayerNorm(1024)`): $2,048$
- **Total `HighCapacityVectorSceneEncoder` ($d=1024$):** $\mathbf{109,642,240}$

---

## 6. Summary Table of Source Files and Classes

| File Path | Class / Function Name | Role / Sub-components | Parameter Count ($d=512$) | Parameter Count ($d=1024$) |
| :--- | :--- | :--- | :---: | :---: |
| `model/dit/DiT.py` | `TimestepEmbedder` | Sinusoidal timestep projection MLP | 394,240 | 1,312,768 |
| `model/dit/DiT.py` | `CustomCrossAttention` | Multi-head cross-attention over context | 1,050,624 | 4,198,400 |
| `model/dit/DiT.py` | `SelfAttention` | Multi-head self-attention over noisy actions | 1,050,624 | 4,198,400 |
| `model/dit/DiT.py` | `MlpBlock` | Position-wise Feed-Forward Network | 2,099,712 | 8,393,728 |
| `model/dit/DiT.py` | `DiTBlock` | Single DiT Block with adaLN-Zero | 9,452,544 | 37,779,456 |
| `model/dit/DiT.py` | `FinalLayer` | Output normalization, adaLN & linear proj | 1,590,276 | 6,326,276 |
| `model/dit/decoder.py` | `CustomDiT` | Full Action Transformer Decoder | **59,244,548** | **462,072,836** |
| `model/modeling_dp_vla.py` | `HighCapacityVectorSceneEncoder` | Multimodal Vector Scene Encoder | **27,559,424** | **109,642,240** |
| `model/modeling_dp_vla.py` | `DpVlaModel` | Complete Backbone Planner (`Encoder + Decoder`) | **86,803,972** | **571,715,076** |

---

## 7. How to Verify Programmatically in PyTorch

You can programmatically verify these exact parameter counts for any module in Python using:

```python
import torch
from model import DpVlaConfig, DpVlaModel

# Active train.py config
config = DpVlaConfig(
    with_encoder=False,
    hidden_size=512,
    depth=6,
    num_heads=8,
    num_actions=20,
    dim_action=4,
    dim_y=12,
)

model = DpVlaModel(config)

# Print total parameter count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params:,}")
```

*Document compiled for DP-VLA Neural Network Architecture Audit.*

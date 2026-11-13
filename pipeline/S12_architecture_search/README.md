# S12: Architecture Search & Loss-Function Study

This stage explores architectural alternatives (S12a) and candidate loss formulations (S12b) to enhance pedestrian-level wind field prediction and peak-speed fidelity over the S11 U-Net baseline.

---

## S12b: Loss-Function Study

Standard masked $L_1$ optimizes the median error across valid open-ground cells, but can produce overly smooth velocity gradients and underestimate peak channeling speeds. S12b implements four candidate loss terms designed to sharpen spatial patterns, preserve high-speed extremes, match spectral energy, and enforce domain-wide mass/mean consistency.

Every loss follows the project interface:
```python
loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
```
All losses are computed in `float32`, normalized by the total count of valid open-ground cells, and return `0.0` (not `NaN`) when a patch contains no valid cells.

---

### Candidate Losses and Formulations

#### 1. Sobel-Gradient Loss (`sobel` / `gradient`)
- **Formula:**
  $$\mathcal{L}_{\text{sobel}} = \mathcal{L}_1 + \alpha \, \frac{\sum |G_{y, \text{pred}} - G_{y, \text{true}}| \cdot M_y + \sum |G_{x, \text{pred}} - G_{x, \text{true}}| \cdot M_x}{\sum M_y + \sum M_x}$$
  where $G_y, G_x$ are computed using $3 \times 3$ Sobel filters with circular periodic padding (`padding_mode="circular"`).
- **Valid Masking:** $M_y = M \odot \text{roll}(M, 1, y) \odot \text{roll}(M, -1, y)$ ensures gradients are only penalized where both neighbouring cells across the difference axis are valid open ground.
- **Reason to try:** Sharpens velocity shear layers along building corners and street canyon boundaries where simple $L_1$ tends to blur transition zones.

#### 2. Tail-Weighted $L_1$ Loss (`tail_w1`, `tail_w2`)
- **Formula:**
  $$\mathcal{L}_{\text{tail}} = \frac{\sum |\text{pred} - y| \cdot w \cdot M}{\sum M}$$
  $$w = \min\left(\frac{y}{U_{\text{mean}}}, c\right)^\gamma \quad (\gamma \in \{1, 2\}, \, c = 5.0)$$
  where $U_{\text{mean}}$ is the open-ground mean speed of the patch/case.
- **Reason to try:** Pedestrian wind comfort and safety hazards depend strongly on peak gusts ($P_{95}$, $U_{\max}$). Tail weighting shifts model attention towards high-speed channeling regions without discarding low-speed wake data.

#### 3. FFT-Magnitude Spectral Loss (`fft` / `spectral`)
- **Formula:**
  $$\mathcal{L}_{\text{fft}} = \mathcal{L}_1 + \alpha \, \frac{\sum |\mathcal{F}(\text{pred} \odot M) - \mathcal{F}(y \odot M)|}{\sum M}$$
  where $\mathcal{F}$ is the 2D orthonormal Discrete Fourier Transform (`torch.fft.rfft2(..., norm="ortho")`).
- **Reason to try:** Enforces spectral fidelity across both low-frequency circulation patterns and high-frequency turbulent structures, mitigating spectral bias.

#### 4. Per-Case Mean-Consistency Loss (`mean_consistency`)
- **Formula:**
  $$\mathcal{L}_{\text{mean}} = \mathcal{L}_1 + \alpha \, |\mu_{\text{pred}} - \mu_y|$$
  $$\mu = \frac{\sum (\cdot) \odot M}{\sum M}$$
- **Reason to try:** Penalizes systemic level offsets between the predicted and actual case-wide open-ground mean, guaranteeing that the model maintains overall energy balance.

---

### Registered Loss Keys for Configs

In `src/urbanwind/dl/losses.py`, all losses are exposed through the `LOSSES` registry:
- `"l1"`: Masked $L_1$ baseline
- `"mse"`: Masked MSE
- `"sobel"` / `"gradient"`: Sobel gradient + $L_1$
- `"tail_w1"`: Tail-weighted $L_1$ ($\gamma = 1$)
- `"tail_w2"`: Tail-weighted $L_1$ ($\gamma = 2$)
- `"fft"` / `"spectral"`: FFT-magnitude + $L_1$
- `"mean_consistency"`: Mean consistency + $L_1$

---

### Verification and Tests
Unit tests in `src/urbanwind/dl/tests/test_losses.py` verify:
1. Exact hand-checked numerical values on a $4 \times 4$ grid with building cells.
2. Zero error on identical inputs ($\text{pred} == y$).
3. Zero return value and finite safety when $\text{mask} == 0$.
4. Optimization convergence: every loss decreases and remains finite over backpropagation steps.

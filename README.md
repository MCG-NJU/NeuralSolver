# NeuralSolver: Differentiable Solver Search for Fast Diffusion Sampling

![](./figs/coeffs_timestep2.png)
This repository contains the code for the paper:
**Differentiable Solver Search for Fast Diffusion Sampling**
#### [NEWS] [11.25] 🍺 Our searched NeuralSolvers and corresponding code are now available in the official repo!

## Methods
Our motivation is that Adams-like linear multi-step methods employ lagrange interpolations, ignoring the *x* related terms.
So we defines a universal interpolation function `P(x, t)` and directly estimate the pre-intergal **coefficients and timesteps** used in the sampling. 

![](./figs/method.png)


## Expertiments

Compared to Linear-Multi-Step Methods, our NeuralSolvers(searched on FlowDCN-B-R256) consistently improve the FID metrics by a large margin respectively. 

![](./figs/fid_performance.png)

We provide a adams-like linear-multi-step solver for the recitified flow sampling. The related configs are named with `adam2` or `adam4`. The solver code are placed in `./src/diffusion/flow_matching/adam_sampling.py`.

Compared to Henu/RK4, the linear-multi-step solver is more stable and faster.

| SiT-XL-R256 | Steps | NFE-CFG  | Extra-Paramters | FID  | IS    | PR   | Recall |
|--|-------|----------|-----------------|------|-------|------|--------|
| Heun | 8     | 16x2     | 0               | 3.68 | /     | /    | /      |
| Heun | 11    | 22x2     | 0               | 2.79 | /     | /    | /      |
| Heun | 15    | 30x2     | 0               | 2.42 | /     | /    | /      |
| Adam2 | 6     | 6x2      | 0 | 6.35 | 190   | 0.75 | 0.55   |
| Adam2 | 8     | 8x2      | 0 | 4.16 | 212   | 0.78 | 0.56   |
| Adam2 | 16    | 16x2     | 0 | 2.42 | 237   | 0.80 | 0.60   |
| Adam4 | 16    | 16x2     | 0 | 2.27 | 243   | 0.80 | 0.60   |
| FlowTurbo | 6     | (7+3)x2  | 30408704(29M)   | 3.93 | 223.6 | 0.79 | 0.56   |
| FlowTurbo | 8     | (8+2)x2  | 30408704(29M)   | 3.63 | /     | /    | /      |
| FlowTurbo | 10    | (12+2)x2 | 30408704(29M)   | 2.69 | /     | /    | /      |
| FlowTurbo | 15    | (17+3)x2 | 30408704(29M)   | 2.22 | 248   | 0.81 | 0.60   |
| NeuralSolver | 6     | 6x2      | 21              | 3.57 | 214   | 0.77 | 0.58   |
| NeuralSolver | 7     | 7x2      | 28              | 2.78 | 229   | 0.79 | 0.60   |
| NeuralSolver | 8     | 8x2      | 36              | 2.65 | 234   | 0.79 | 0.60   |
| NeuralSolver | 10    | 10x2     | 55              | 2.40 | 238   | 0.79 | 0.60   |
| NeuralSolver | 15    | 15x2     | 110              | 2.24 | 244   | 0.80 | 0.60   |

## Visualizations of zero-shot T2I

#### Flux Models with Euler-shift3 and our NeuralSolver(searched on SiT-XL-R256) under CFG=2.0
![](./figs/flux_cfg2.png)

#### PixArt Models with UniPC/DPMSolver++ and our NeuralSolver(searched on DiT-XL-R256) under CFG=2.0
![](./figs/pixart1024_cfg2.png)


## Citation
```bibtex
@article{wangdifferentiable,
  title={Differentiable Solver Search for fast diffusion sampling},
  author={Wang, Shuai and Li, Zexian and Song, Tianhui and Li, Xubin and Ge, Tiezheng and Zheng, Bo and Wang, Limin and others}
}
}
```

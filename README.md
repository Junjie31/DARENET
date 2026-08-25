# DareNet

**DareNet: Domain-Aware Remote Physiological Measurement**

DareNet 是一个面向跨数据集（cross-domain）远程生理信号（rPPG）估计的双专家骨干网络。
它建立在 [RhythmMamba](https://arxiv.org/abs/2404.06483) 之上，引入两个关键模块：

1. **Stability-Aware Fusion Stem (SAFS)** —— 通过 RMS 归一化稳定特征能量，并用轻量门控
   `SafuGate` 自适应融合「原始帧」与「时序差分」两个支路，缓解跨域时的特征尺度漂移。
2. **Domain-Aware Robust Expert (DARE)** —— 用两个互补的专家替代单一 FFN：
   - **Complex（spectral）专家**：在复数频域做全局谱分析，保留相位一致性，适合干净、周期性的信号。
   - **Gated（gated_tf）专家**：时域卷积 + 频域特征的**门控时频融合**，适合带运动伪影的信号。

两个专家共享同一个 stem 与 Mamba 骨干，仅在 `Frequencydomain_FFN` 中按 `ffn_branch ∈ {spectral, gated_tf}`
切换，因此**同一份权重结构可加载两类 checkpoint**。

> 代码基于 [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) 与
> [RhythmMamba](https://github.com/zizheng-guo/RhythmMamba) 构建，训练/评估协议与二者保持一致。

---

## 目录结构

```
DareNet/
├── main.py                  # 训练/推理入口（train_and_test / only_test）
├── config.py                # 配置系统（yacs）
├── configs/
│   ├── cross/               # 跨数据集「训练」配置（Tab.3）
│   ├── intra/               # 数据集内「训练」配置（Tab.4）
│   └── infer/               # 预训练权重「推理」配置（only_test，含各 seed）
├── neural_methods/
│   ├── model/DareNet.py            # ★ 单专家骨干（SAFS + 可切换 FFN 专家）
│   ├── model/DualExpertDareNet.py  # ★ 双专家路由视图（显式 MoE，见下文）
│   ├── model/RhythmMamba.py        # RhythmMamba 基线（仅代码，不发布权重）
│   └── trainer/                    # 各模型 trainer
├── dataset/                 # PURE/UBFC/MMPD 等数据加载器
├── evaluation/              # 指标（MAE/RMSE/MAPE/Pearson/SNR）+ HR 估计
├── unsupervised_methods/    # 传统无监督方法（GREEN/ICA/CHROM/LGI/PBV/POS）
├── pretrained/PreTrainedModels/  # 已发布的 checkpoint（.pth，见 CHECKSUMS.txt）
├── scripts/
│   ├── gen_configs.py          # 按本地数据路径重新生成配置（单一数据来源）
│   ├── aggregate.py            # 汇总多 seed 结果 → mean±std 表
│   └── demo_dual_expert.py     # 双专家路由 demo（加载两专家 + 路由算子）
└── reproduce.sh             # 一键复现 Tab.3/Tab.4 的推理脚本
```

---

## 环境安装

```bash
bash setup.sh conda        # 或 bash setup.sh uv
conda activate DareNet
```

依赖见 `requirements.txt`，关键项：`torch==2.1.2+cu121`、`mamba-ssm==2.2.2`、`causal-conv1d==1.4.0`。

---

## 数据准备

使用 rPPG-Toolbox 的预处理管线。三个数据集均可按下面步骤得到预处理的 `.npy`（首次运行设
`DO_PREPROCESS: True`，之后为 `False` 直接读缓存）：

| 数据集 | 原始数据来源 | 预处理要点 |
|--------|--------------|------------|
| PURE | 向[作者](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)申请 | 30 Hz、160 帧 chunk、128×128、face crop（large box 1.5）、Standardized |
| UBFC-rPPG | [链接](https://sites.google.com/view/ybenezeth/ubfcrppg) | 同上 |
| MMPD | 向作者申请 | 同上；测试时开启全部 metadata 过滤（LIGHT/MOTION/EXERCISE/… 全选） |

把数据路径填进 `scripts/gen_configs.py` 顶部的 `PATHS` / `INTRA_SPLIT` / `CROSS_SPLIT`，
再运行 `python scripts/gen_configs.py` 即可生成与仓库一致的配置。

---

## 复现论文结果

### 1) 直接用预训练权重推理（最快）

```bash
bash reproduce.sh
```

该脚本对 Tab.3 / Tab.4 每个方向用对应的 `configs/infer/*.yaml`（`only_test` 模式）跑一遍，
结果写入 `./saved_metrics/<task>/`，最后用 `scripts/aggregate.py` 汇总成 mean±std 表。

也可以单独跑某一条，例如 PURE→MMPD（gated，seed42）：

```bash
python main.py --config_file configs/infer/PURE_MMPD_gatedtf_seed42.yaml
```

### 2) 从头训练

跨数据集（训练集 → 测试集）用 `configs/cross/`，数据集内用 `configs/intra/`，例如：

```bash
# 跨数据集：PURE 训练 → MMPD 测试（gated 专家）
python main.py --config_file configs/cross/PURE_MMPD_gatedtf.yaml
```

训练采用 `train_and_test` 模式，`USE_LAST_EPOCH: True`，即训练 30 epoch 后直接用最后一轮
（Epoch29）权重测试。

### 3) 多 seed

论文 Tab.3 的「Ours」行是 3-seed（UBFC→MMPD 为 6-seed）均值，但为控制体积，本仓库只发布
每个方向的 **seed42** checkpoint（论文 UBFC→MMPD 各 seed MAE 差异 < 0.005，见补充材料）。
如需复现多 seed 均值，用 `--seed <N>` 从头训练（会追加 `_s<N>` 到 `MODEL_FILE_NAME`）。

---

## 论文数字 ↔ 本仓库 checkpoint 对照表

以下数值均可由 `pretrained/PreTrainedModels/` 中的 checkpoint 复现（`predict_hr_fft_all` vs
`gt_hr_fft_all`，subject-level，Welch FFT）。发布的是每个方向的 seed42 checkpoint，
对应论文 3-seed/6-seed 均值里的该 seed。

### Tab.3 跨数据集（Ours = 互补优势分支）

| 方向 | 分支 | checkpoint（seed42） | 论文均值 MAE | 本仓库 MAE | RMSE | ρ |
|------|------|-----------|---------|---------|------|----|
| PURE→UBFC | gated_tf | `PURE_UBFC_DareNet_gated_tf_seed42_Epoch29.pth` | 0.89 | 0.87 | 1.76 | 1.00 |
| UBFC→PURE | spectral | `UBFC_PURE_DareNet_spectral_seed42_Epoch29.pth` | 0.94 | 0.60 | 1.02 | 1.00 |
| PURE→MMPD | gated_tf | `PURE_MMPD_DareNet_gated_tf_seed42_Epoch29.pth` | 8.53 | 8.51 | 14.14 | 0.47 |
| UBFC→MMPD | gated_tf | `UBFC_MMPD_DareNet_gated_tf_seed42_Epoch29.pth` | 8.25 | 8.25 | 14.05 | 0.47 |

> UBFC→PURE 的 3-seed 均值 0.94 是 spectral 分支，seed42 单 seed MAE=0.60（其它 seed 更高）；
> 该方向谱专家波动较大，论文均值按 3 seed 报告。

### Tab.4 数据集内（Ours）

| 数据集 | 分支 | checkpoint（seed42） | 论文 MAE | 本仓库 MAE | RMSE | ρ |
|--------|------|-----------|---------|---------|------|----|
| PURE | spectral | `PURE_PURE_DareNet_spectral_seed42_Epoch29.pth` | 0.23 | 0.23 | 0.35 | 0.99 |
| UBFC | spectral | `UBFC_UBFC_DareNet_spectral_seed42_Epoch29.pth` | 0.54 | 0.54 | 0.79 | 0.99 |
| MMPD | gated_tf | `MMPD_MMPD_DareNet_gated_tf_seed42_Epoch29.pth` | 4.21 | 4.21 | 8.48 | 0.80 |

---

## 模型与代码讲解

### 核心文件：`neural_methods/model/DareNet.py`

- **`Fusion_Stem`（SAFS）**：7×7 卷积 + BN + MaxPool 的空间编码器（总下采样 /8），在时间维度上
  做帧间差分得到 12 通道差分特征，先经 RMS 归一化稳定能量，再用 `SafuGate` 输出
  `α, β ∈ [0,1]`，把原始支路 `X_raw` 与差分支路 `X_diff` 按
  `A·xr + B·xd`（`A = apha·(1+eps·(α−0.5)·2)`）自适应融合。
- **`Frequencydomain_FFN`（DARE）**：根据 `ffn_branch` 二选一实例化
  - `spectral`：复数 FFT → 复数线性变换（`r/i/rb/ib`）→ iFFT → `fc2`，保留相位。
  - `gated_tf`：`time_branch`（深度可分离 1D 卷积）+ `freq_branch`（幅度谱 MLP），由
    `fusion_gate` 门控融合，最后残差连接 `x + out_proj(fused)`。
- **`Block_mamba`**：Mamba 层 + 上述 FFN，24 层堆叠（`embed_dim=96`），用多路径时间移位
  （segment=4）捕捉时序信息。

### 双专家结构（DARE 的 MoE 视图）：`neural_methods/model/DualExpertDareNet.py`

论文（Sec.3 DARE + Sec.4 Policy）把 DARE 描述成一个「共享骨干 + 两个专家 + 路由」的 MoE。
代码层面，DARE-Net 的 MoE 是通过**「专门化训练」+「预测级路由」**两段实现的，对应论文
Eq. routing 的两个极端情形：

- **训练（Sec.3，Eq. routing 的 β∈{0,1}）**：`DareNet(dataset_name, ffn_branch=...)` 一次只
  专门化**一个**专家 —— `β=1` 对应 Complex（`ffn_branch=spectral`），`β=0` 对应 Gated
  （`ffn_branch=gated_tf`）。因此两个专家各自拥有一套 stem / Mamba 骨干权重，分开训练。
- **推理（Sec.4）**：`DualExpertDareNet` 把**两个**已训练专家同时加载，对同一输入各产出
  一个预测 `(y_c, y_g)`，再在其上应用路由算子。

`DualExpertDareNet` 显式暴露论文 Sec.4 的路由算子：

| 算子 | 公式 | 对应论文 |
|------|------|---------|
| `fixed_avg` | `(y_c + y_g) / 2` | FixedAvg |
| `soft_blend` | `α·y_c + (1−α)·y_g` | SoftBlend（α 由可靠性分数拟合）|
| `hard_select` | `1{π=c}·y_c + 1{π=g}·y_g` | HardSelector / FailureGate |

`predict()` 还返回 domain fingerprint（时序差分能量 + 谱熵，论文 Fig. fingerprint），可作为
domain-conditioned 路由的依据。快速验证：

```bash
python scripts/demo_dual_expert.py \
  --complex_ckpt pretrained/PreTrainedModels/MMPD_MMPD_DareNet_spectral_seed42_Epoch29.pth \
  --gated_ckpt   pretrained/PreTrainedModels/MMPD_MMPD_DareNet_gated_tf_full_seed42_Epoch29.pth \
  --test_dataset MMPD --train_dataset MMPD \
  --cached_path /home/junjie/Workspace/Data/mini_MMPD --max_windows 400
```

> 这种「训练单分支专门化 + 推理双专家路由」是论文自述的设计（"the implementation selects one
> FFN expert branch … best viewed as a training-time expert-specialisation formulation"）。
> `DualExpertDareNet` 把这条链路在代码里显式化，而不改动已发布的 checkpoint 与复现数字。

### 加载与切换

`DareNet(dataset_name=..., ffn_branch=...)` 构造模型后，两类 checkpoint 均以 `strict=True` 加载
（仅 `module.` 前缀需要剥离，因为原始权重由 `DataParallel` 保存）。训练/推理入口是
`neural_methods/trainer/DareNetTrainer.py`，数据增强（AUG=1）、`Hybrid_Loss`、`OneCycleLR`、
`Epoch29` 末轮测试与论文一致。

---

## Reliability-Aware Policy Layer 复现（论文 Sec.4）

论文 Sec.4 的可靠性策略分析是 **per-window 双专家**口径（每个 160 帧窗口算一个 HR），与
Tab.3/Tab.4 的 **subject-level 单专家**口径不同。三个层次不要混淆：

| 层次 | 模型 | 口径 | 关键结果 |
|------|------|------|---------|
| Tab.3 跨数据集 | 单专家 BestSingle | subject-level | PURE→MMPD 8.53、UBFC→MMPD 8.25、UBFC→PURE 0.94、PURE→UBFC 0.89 |
| Tab.4 数据集内 | 单专家 | subject-level | 0.23 / 0.54 / 4.21 |
| Sec.4 Policy | 双专家 + 路由 | per-window | BestSingle 13.95、FailureGate@0.3 → **7.53** |

复现命令：

```bash
bash reproduce_policy.sh
```

流程：`scripts/dump_policy_windows.py`（分别 dump Complex/Gated 两专家的 per-window 预测）
→ `tools/bmvc_analysis/run_all.py --use_official_errors`（对齐、特征、failure 预测 selector、
路由策略评估、abstention）。

关键结果（已用官方 `dumps/aligned/` label 核对）：

| 表 | 数字 | 论文 |
|----|------|------|
| policy_eval (MMPD FailureGate@0.3) | MAE 7.53 / FR@10 0.23 / coverage 0.38 | ✅ 完全一致 |
| policy_eval (MMPD BestSingle / Oracle) | 13.95 / 10.07 | ✅ |
| failureauc (MMPD oracle_failure) | AUC 0.70 (pos 2167/6732) | 0.71 ✅ |

> failure_c/failure_g 的 AUC 因论文未写死 margin δ 的数值，代码取 `DELTA=3`
> （见 `tools/bmvc_analysis/failure_selectors.py`，对应论文 Eq. `failure_c = I(e_c > e_g + δ)`）。
> 修复前的版本用「单边绝对阈值 `err>10`」作 label，与论文的「相对 margin」定义不符；
> 现已修正为论文定义，并接入官方 per-window err/HR。

---

## 说明与致谢

- **可复现范围**：Tab.3 / Tab.4 的 **DareNet（Ours）** 由本仓库的 checkpoint 与代码直接复现；
  Sec.4 policy 的 failureauc/policy_eval/failuregate 由 `tools/bmvc_analysis/` 复现。
  RhythmMamba 及其余基线（DeepPhys/PhysNet/TS-CAN/EfficientPhys/PhysFormer/RhythmFormer/
  BigSmall 及传统无监督方法）保留了模型代码与配置入口，其表格数值为引用已有工作，
  不在本次发布的 checkpoint 覆盖范围内。
- **数据路径**：`configs/` 内路径为占位符，请按 `scripts/gen_configs.py` 填写本地路径。
- **`.pth` 不入 git**：`pretrained/PreTrainedModels/` 已由 `.gitignore` 排除；完整性见
  `pretrained/PreTrainedModels/CHECKSUMS.txt`（发布时可用云盘/脚本下载）。

感谢 [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) 与
[RhythmMamba](https://github.com/zizheng-guo/RhythmMamba) 的作者。

## Citation

```
@inproceedings{darenet2026,
  title={DareNet: Domain-Aware Remote Physiological Measurement},
  author={Hou, Junjie and Chow, Chong Tong and He, Yudong and Odinaev, Ismoil and Wong, Kwan Long and Chin, Jing Wei and So, Richard H. Y.},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2026}
}
```

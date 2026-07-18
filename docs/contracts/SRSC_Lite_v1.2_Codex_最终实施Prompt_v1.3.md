# Codex 最终实施 Prompt：SRSC-Lite v1.2
## 工程母体：PromptIR 官方仓库；实际方法基线：Restormer-AiO
## 版本：v1.4（ResearchStudio 终审版：严格局部坐标、Oracle/Predicted 双梯级、等信息/等容量预算、单卡 4090 可执行）

你现在是一名严谨的高级 PyTorch 图像复原研究工程师。请在 PromptIR 官方仓库的训练与评测协议上，先构造一个**完全删除 PromptIR 提示创新的干净 Restormer-AiO 基线**，随后实现 **SRSC-Lite v1.2**。

本任务不是把几个模块随意拼在一起，也不是为了无条件跑满训练。目标是用可复核的代码和严格的等容量实验回答：

> 在第一次复原之后，带符号的修复进度与低维偏离方向，是否比输出特征、first-pass residual、误差幅值/uncertainty，以及同维度直接 GT residual code，提供更可学习、更可执行的第二阶段反馈？

只有信息价值闸门通过，才允许进入联合完整训练。

本文件中的 “GO” 分为两个层次，严禁混淆：

- `SCIENTIFIC_GO`：Stage-B 证明状态表示具有可重复的信息价值，允许进入联合训练；
- `PUBLICATION_GO`：Stage-C 在同协议、同 checkpoint、公平预算下达到论文级性能与稳健性门槛。

`SCIENTIFIC_GO` 不等于方法已经达到顶刊顶会发表强度；只有两级闸门均通过，才能将 SRSC-Lite 冻结为最终论文主方法。

---

# 0. 最终宿主选择与论文命名边界

## 0.1 工程母体

使用：

```text
https://github.com/va1shn9v/PromptIR
```

优先固定到已审计 commit：

```text
106159ab809101f2e25b6714195cd6fa9a938d36
```

先验证该 commit 属于官方 remote 并 checkout。若它不存在或无法获取，不得猜测；停止并在 `AUDIT.md` 报告，除非用户明确授权改用新的实际 HEAD。若获准使用新 HEAD，必须记录 commit、diff 摘要和日期，不能静默漂移。

PromptIR 仓库用于：

- AIO-3 数据加载与评价协议；
- Restormer 的 MDTA、GDFN、LayerNorm、上下采样基础块；
- 官方 PromptIR 外部基线复现；
- 训练、验证和测试基础设施。

## 0.2 实际内部基线名称

删除 PromptIR 的 prompts 后，模型不再称为 PromptIR。

论文和报告统一写：

```text
Restormer-AiO
```

或：

```text
a clean Restormer backbone trained under the PromptIR all-in-one protocol
```

PromptIR 原模型作为独立外部已发表基线进入比较表。

## 0.3 为什么不以 MoCE-IR 为代码母体

MoCE-IR 仅作为强外部比较方法，不拆解其代码作为母体。禁止将以下机制带入主方法：

- complexity experts；
- sparse/top-k routing；
- complexity bias；
- frequency-conditioned gate；
- expert-specific rank/depth/receptive field；
- load balancing；
- task-label routing。

原因必须写入 `AUDIT.md`：这些机制嵌入主计算块，删除后会同时改变 block、容量和路由，难以把增益归因给 SRSC。

## 0.4 必须彻底删除 PromptIR 的创新

不得只设置 `decoder=False` 后保留死参数。必须从新模型定义中完全删除：

- `PromptGenBlock`；
- `prompt1`, `prompt2`, `prompt3`；
- `prompt_param`, prompt bank, prompt weights；
- prompt concatenation；
- `noise_level1/2/3`；
- `reduce_noise_level1/2/3`；
- `reduce_noise_channel_*`；
- `chnl_reduce*`；
- 所有 prompt-conditioned decoder path；
- 任何真实 task label、degradation label、noise level 输入。

保留的只有通用 Restormer 基础块、U-Net encoder/decoder 和数据/评测协议。

---

# 1. 硬约束

## 1.1 严禁

- 修改原始官方仓库；必须复制到独立工作区；
- 读取旧 UARIR、MEB、RGM、R2R 或其他历史私有实现；
- 用 GT、task label、noise level 或 state target 进入推理图；
- 测试集调学习率、feedback 形式、方向维数、阈值或 checkpoint；
- TTA、checkpoint soup、per-task output calibration；
- 在不同反馈变体间改变 D2、训练步数、数据、参数量或 feedback 接口宽度；
- 用不同的 active/dummy assessor、feedback adapter 或 SRSC-Mod 计算路径制造参数/MACs差异；
- 在 Stage-B 信息价值未通过前运行 AIO-5 或长程 Stage-C；
- 因某个 oracle 很强而把它当作可部署结果；
- 把 full-e 或 direct GT residual oracle 放进方法主表；
- 自动派生无穷配置；
- 声称提升，除非真实命令、日志和 checkpoint 都存在。

## 1.2 主模型固定

```text
single model
single degraded-image input
fixed K=2
one shared encoder
one coarse decoder D1
one deterministic state assessor
one correction decoder D2
no recurrence
no dynamic stopping
no VLM/LLM
no MoE
no agent
no tool library
```

## 1.3 单卡预算

目标 GPU：单张 RTX 4090/3090 24GB。

```text
MAX_WALL_CLOCK_HOURS_STAGE_B_PILOT = 24
MAX_GPU_HOURS_STAGE_B_PILOT = 20
```

超时：

- 完成当前原子写入；
- 生成已有报告；
- 标记 `TIME_BUDGET_EXCEEDED`；
- 不得擅自缩减测试集后宣称通过。

---

# 2. 工作区与输出

建议：

```text
/root/autodl-tmp/srsc_lite_v12/
```

结构：

```text
srsc_lite_v12/
  upstream/PromptIR/                 # 只读 clone
  src/
    net/
      restormer_blocks.py
      clean_restormer_aio.py
      shared_encoder.py
      coarse_decoder.py
      correction_decoder.py
      feedback_interface.py
      srsc_coordinates.py
      state_assessor.py
      srsc_modulation.py
      srsc_lite.py
    data/
    losses/
    metrics/
  configs/
    protocol_aio3.yaml
    stage_a_coarse.yaml
    stage_b_oracle/*.yaml
    stage_b_predicted/*.yaml
    stage_c_joint.yaml
  scripts/
    audit_repo.py
    verify_promptir_baseline.py
    train_stage_a.py
    cache_stage_a.py
    train_stage_b_oracle.py
    train_stage_b_predicted.py
    train_stage_c.py
    eval_locked.py
    profile_models.py
    orchestrate.py
    status.sh
  tests/
    test_shapes.py
    test_prompt_removal.py
    test_no_gt_inference.py
    test_coordinate_math.py
    test_clean_region_stability.py
    test_projection.py
    test_feedback_width.py
    test_zero_init_identity.py
    test_gradient_routing.py
    test_padding.py
    test_checkpoint_roundtrip.py
  reports/
    AUDIT.md
    ARCHITECTURE.md
    BASELINE_PARITY.md
    STAGE_A_REPORT.md
    STAGE_B_ORACLE_REPORT.md
    STAGE_B_PREDICTED_REPORT.md
    FINAL_DECISION.md
    decision.json
  artifacts/
    manifests/
    stats/
    metrics/
    plots/
    checkpoints/
    logs/
  RUNNING_STATUS.md
  STOP_REASON.md
```

---

# 3. 启动审计

先执行：

```bash
nvidia-smi
tmux ls || true
ps -ef | grep -E "python|torchrun|train" | grep -v grep || true
df -h
free -h
```

随后：

1. clone PromptIR；
2. 记录 remote、HEAD、dirty status；
3. 计算 prompt、config、dataset manifest、checkpoint SHA256；
4. 记录 Python/PyTorch/CUDA/cuDNN；
5. 阅读并记录：
   - `net/model.py`
   - `train.py`
   - `test.py`
   - `options.py`
   - dataset code
   - scheduler
   - loss
   - metric
6. 实际列出 PromptIR prompt 路径及删除映射；
7. 记录 AIO-3 数据路径与缺失项；
8. 禁止未审计就写模型。

输出 `reports/AUDIT.md` 与 `reports/ARCHITECTURE.md`。

---

# 4. 数据与评价协议

## 4.1 第一阶段只做 AIO-3

训练任务：

```text
denoise sigma 15/25/50
derain
dehaze
```

测试：

```text
BSD68 sigma 15/25/50
Rain100L
SOTS
```

AIO-5 只有在 Stage-B GO 后才允许。

## 4.2 划分

创建：

```text
train
locked_val
official_test
```

约束：

- 同一 clean content 的不同 sigma 不得跨 split；
- official test 不能用于 early stopping 或配置选择；
- 所有 feedback 选择只看 locked_val；
- 最终配置锁定后 official test 只运行一次；
- manifest 写入 hash。

## 4.3 指标

至少：

```text
PSNR RGB full image
SSIM
per-image PSNR delta
five-setting mean
three-task macro
```

评价代码必须与 PromptIR 官方协议逐项核对：

- RGB/BGR；
- [0,1]/[0,255]；
- clamp；
- border crop；
- padding crop-back；
- Gaussian noise seed；
- full RGB vs Y；
- image averaging方式。

先复现官方 PromptIR checkpoint；差异超过 0.10 dB 时停止，不得继续方法实验。

---

# 5. 干净 Restormer-AiO 基线

## 5.1 基础块

无语义变化地迁移：

- BiasFree/WithBias LayerNorm；
- MDTA；
- GDFN；
- TransformerBlock；
- OverlapPatchEmbed；
- PixelUnshuffle Downsample；
- PixelShuffle Upsample。

## 5.2 Shared Encoder

默认：

```yaml
dim: 48
encoder_blocks: [4, 6, 6, 8]
heads: [1, 2, 4, 8]
ffn_expansion_factor: 2.66
layernorm_type: WithBias
bias: false
```

返回：

```text
F1: B x 48  x H   x W
F2: B x 96  x H/2 x W/2
F3: B x 192 x H/4 x W/4
F4: B x 384 x H/8 x W/8
```

Encoder 只执行一次。

## 5.3 单阶段 Restormer-AiO

实现一个干净的单 decoder 基线：

```text
level3 blocks = 6
level2 blocks = 6
level1 blocks = 4
refinement blocks = 4
```

输出：

```text
y = x + delta
```

这只是干净基线，不是 SRSC 主模型。

必须支持 reflect padding 到 8 的倍数并准确 crop 回原尺寸。

---

# 6. SRSC-Lite 主模型拓扑

## 6.1 计算预算设计

为避免“只是多一个 decoder”：

- Shared encoder 与单阶段基线相同；
- D1 与 D2 的 Transformer block 总数固定为 20，等于干净单 decoder 的总 block 数；
- 推荐：
  - D1：level3/2/1 = [2,2,2]，refinement=0；
  - D2：level3/2/1 = [4,4,4]，refinement=2；
- 所有 feedback 变体使用完全相同的 E/D1/D2；
- 另外实现相同参数量/MACs 的 two-stage-no-state baseline；
- 方法与 two-stage-no-state 的参数差异需 <0.5%；
- 方法与单阶段基线的参数/MACs差异必须如实报告，不要求 <0.5%。

若审计发现此分配显存不可行，只允许统一改动一次，并在看指标前锁定。

为排除“SRSC 只在被人为削弱的 D1 上有效”，Stage-B 主配置通过后必须补一个预注册容量分配复核：

```text
main split:        D1/D2 = 6/14 blocks
robustness split:  D1/D2 = 10/10 blocks
```

两者总 decoder block 数均为 20，encoder、接口和训练协议不变。10/10 只验证核心排序 `p+m+d > p+m` 和 `SRSC > matched residual code`，不重新搜索超参数。若结论只在 6/14 成立，必须报告为对 coarse-stage capacity 敏感，不能宣称一般性的 restoration-state 优势。

## 6.2 D1：Coarse Decoder

流程：

```text
F4
→ upsample 384→192
→ concat F3[192]
→ 1x1 fusion 384→192
→ 2 Restormer blocks
→ upsample 192→96
→ concat F2[96]
→ 1x1 fusion 192→96
→ 2 Restormer blocks
→ upsample 96→48
→ concat F1[48]
→ 1x1 fusion 96→48
→ 2 Restormer blocks
→ 3x3 head delta0
```

D1 level1/2/3分别使用1/2/4 heads。干净单阶段 Restormer-AiO、D1与D2使用同一 decoder-width convention `[48,96,192]`；不能一处保留 PromptIR level1 concat 后的96通道、另一处降到48通道。若审计决定严格保留官方 decoder width，必须在看任何方法指标前对 clean baseline、D1、D2及所有容量对照一起修改并锁定。

```python
y1 = x + delta0
```

训练 forward 中不 hard clamp。

## 6.3 y1 shallow pyramid

轻量 stem，非完整 encoder：

```text
G1: 3x3 conv,             B x 24  x H   x W
G2: 3x3 stride-2 conv,    B x 48  x H/2 x W/2
G3: 3x3 stride-2 conv,    B x 96  x H/4 x W/4
G4: 3x3 stride-2 conv,    B x 192 x H/8 x W/8
```

每层使用 `Conv -> GELU`，除上述层外不增加完整 encoder blocks。主配置固定为 `[24,48,96,192]`，所有 feedback/no-state/容量对照完全相同；不得在看指标后改单独尺度宽度。

## 6.4 D2：Correction Decoder

D2 的 decoder feature width 固定为：

```text
Z4: 384 channels at H/8
Z3: 192 channels at H/4
Z2: 96 channels at H/2
Z1: 48 channels at H
```

H/8 latent 起点必须显式消费 S4：

```text
concat(F4[384], G4[192])
→ 1x1 fusion 576→384
→ SRSC-Mod4(S4)
→ Z4
```

随后三个尺度顺序严格固定：

```text
Z4 upsample 384→192
→ concat F3[192], G3[96]
→ 1x1 fusion 480→192
→ SRSC-Mod3(S3)
→ 4 Restormer blocks → Z3

Z3 upsample 192→96
→ concat F2[96], G2[48]
→ 1x1 fusion 240→96
→ SRSC-Mod2(S2)
→ 4 Restormer blocks → Z2

Z2 upsample 96→48
→ concat F1[48], G1[24]
→ 1x1 fusion 120→48
→ SRSC-Mod1(S1)
→ 4 Restormer blocks
→ 2 refinement Restormer blocks → Z1
```

Restormer heads对应 `[1,2,4,8]`：Z1/Z2/Z3/Z4分别使用1/2/4/8 heads。S1/S2/S3/S4必须由 assessor 直接输出到对应分辨率，不允许用一个 full-resolution state 任意插值得到全部尺度作为主实现。

最后：

```python
delta1 = correction_head(feature)
y2 = y1 + delta1
```

D2 不直接重建整张 y2。

D1/D2 不共享权重。

---

# 7. 固定 8 通道 Feedback Interface

这是公平实验的核心。

每个尺度所有变体必须输出：

```text
B x 8 x Hi x Wi
```

D2、SRSC-Mod、state adapter 完全相同。

## 7.1 Oracle ladder 与 Predicted ladder 分开

### Oracle ladder

GT-derived feedback 直接送入相同的 8-channel interface，不使用 assessor。

用途：

> 只比较反馈表示的信息价值。

### Predicted ladder

所有表示使用完全相同的 assessor 架构、输入、参数量和 8-channel output head。

用途：

> 比较反馈表示的可学习性。

禁止把 oracle 与 predicted 数字混在同一主表。

### Zero/no-state 的等容量定义

`O0/P0/two-stage-no-state` 不能通过删除 assessor、feedback adapter 或 SRSC-Mod 实现。它必须实例化并执行与主方法完全相同的：

```text
E + D1 + y1 pyramid + Assessor + feedback adapter + SRSC-Mod + D2
```

唯一差异是在统一接口处执行：

```python
state_for_d2 = torch.zeros_like(predicted_or_dummy_state)
```

assessor 输出不参与 D2，但 assessor 前向、参数、MACs均保留；可停止其梯度以避免无意义更新。这样 no-state 与 SRSC 的参数量和推理计算路径匹配。另行报告一个删除 assessor 的部署型 no-state 仅作效率参考，不能用于核心因果比较。

---

# 8. 固定局部描述子与 SRSC target

## 8.1 多尺度计算顺序

每个尺度：

1. 用 `area` 将 x/y1/gt 下采样到当前尺度；
2. 再计算固定描述子；
3. 不允许先在 full resolution 计算 signed state 后任意插值。

## 8.2 描述子

```math
psi(I) = concat(I, 0.5 Sobel_x(I), 0.5 Sobel_y(I))
```

9 通道。

Sobel kernel 注册为 buffer，不训练。

## 8.3 严格一致的局部 patch-vector 坐标空间

固定：

```text
k = 3
descriptor channels = 9
q_dim = 9 * k * k = 81
padding = reflect
```

对描述子差分执行局部展开：

```math
v^\ast=\operatorname{unfold}_k\left(\psi(gt)-\psi(x)\right)
```

```math
v_1=\operatorname{unfold}_k\left(\psi(y_1)-\psi(x)\right)
```

每个位置的 `vstar` 与 `v1` 均为81维。内积、范数、投影残差和方向投影必须全部在同一个81维空间计算：

```math
dot=\sum_{j=1}^{81}v_{1,j}v^\ast_j,
\qquad
n_\ast=\sum_{j=1}^{81}(v^\ast_j)^2
```

```math
\alpha=\frac{dot}{n_\ast+\epsilon},
\qquad \epsilon=10^{-6}.
```

允许用 `torch.nn.functional.unfold`、固定 shift-and-stack 或等价 grouped-convolution 实现；必须用单元测试证明数值等价。为节省24GB显存，target builder允许按尺度/空间块计算，且不保留不必要的 autograd graph。严禁退回“邻域 AvgPool 计算 alpha、只对中心9维 e 投影”的不一致近似。

## 8.4 Signed progress

保留 raw 值用于分析：

```math
p_raw = 1 - alpha
```

训练 target 使用平滑有界映射：

```math
p = 2 tanh(p_raw / 2)
```

它保留符号，避免硬 clip 的梯度断点。

## 8.5 Deviation

```math
e = v1 - alpha * vstar
```

由于 `alpha`、`e` 与 `vstar` 均定义在同一个81维局部空间，按构造有：

```math
\langle e,v^\ast\rangle\approx0.
```

误差只应来自 `eps` 与浮点数值。必须测试有效区域中的相对正交误差。论文可以称其为 **target-relative local transverse deviation**，但仍不得声称它是真实物理退化的唯一分解。

## 8.6 Deviation magnitude

```math
m_raw = sqrt(sum_81(e^2) + eps) /
        (sqrt(nstar + eps) + eps)
```

训练：

```math
m = 2 tanh(m_raw / 2)
```

## 8.7 有效编辑权重

避免 `||vstar||≈0` 时 p/d 不稳定。

在 train split 上、正式训练前统计有效 `sqrt(nstar)` 的第10百分位，记为 `tau_v`，锁定到 config。只纳入：

```text
sqrt(nstar) > max(1e-4, 1e-3 * train_median_positive_norm)
```

并规定 `tau_v=max(percentile10,1e-4)`。保存统计样本数、排除比例、最终阈值和统计文件 SHA256；不能把浮点微小正数当作有效编辑。

```math
q = nstar / (nstar + tau_v^2)
```

p 与 direction loss 使用 q 加权，不做硬阈值删除。仅降低 loss 权重还不够：实际送入 feedback interface 的状态也必须门控，见8.9。

clean/near-clean 区域同时使用 no-unnecessary-edit loss：

```math
L_clean = (1-q) * |y2-y1|
```

## 8.8 低维方向

先在同一个81维局部空间归一化：

```math
u = e / (sqrt(sum_81(e^2)) + eps)
```

固定随机正交投影：

```text
P: 6 x 81
seed: 20260713
P P^T ≈ I
```

```math
d = P u
```

P 注册为 buffer，不训练。

方向有效权重：

```math
w_dir = q * m_raw / (m_raw + tau_e)
```

`tau_e` 使用 train split 有效 positive `m_raw` 第10百分位并冻结，同时设置 `tau_e>=1e-4`。保存统计规则、样本数、阈值和 SHA256。

PCA 投影只做消融；PCA 仅使用 train labels 拟合并冻结。

## 8.9 SRSC-Lite target

损失权重不能防止 Oracle 或预测器在无效区域向 D2 输入不稳定状态，因此定义实际进入 feedback interface 的有效坐标：

```math
\tilde p=q\,p,
\qquad
\tilde m=q\,m,
\qquad
\tilde d=w_{dir}\,d.
```

最终 target：

```text
S_eff = [p_tilde, m_tilde, d_tilde1..d_tilde6] = 8 channels
```

Oracle ladder直接使用 `S_eff`；Predicted ladder预测 `S_eff`。未经门控的 raw `p/m/d` 只用于标签分析和可视化，不得直接送入 D2。必须记录 q 与 w_dir 的分布，确认其不是几乎处处为0或1。

---

# 9. Equal-information residual-code baseline

这是不可省略的核心公平对照。

定义：

```math
rstar = unfold_k(psi(gt) - psi(y1))
```

用固定投影：

```text
Pr: 8 x 81
seed: 20260714
Pr Pr^T ≈ I
```

```math
zr = Pr rstar
```

要求：

- 与 SRSC target 相同空间尺度；
- 同为 8 通道；
- 与 SRSC 使用相同的 `k=3` 和81维局部描述子空间；
- 只用 train split 统计做逐通道 robust normalization；
- 相同 assessor；
- 相同 D2；
- 相同 state loss 权重尺度；
- 相同训练步数；
- 相同 feedback adapter。

报告：

- oracle residual code；
- predicted residual code；
- code prediction MAE/cosine；
- code entropy/variance；
- downstream PSNR。

不能只比较 SRSC 与 uncertainty。

---

# 10. Oracle ladder 定义

所有版本都通过相同 8-channel feedback interface。

```text
O0  zero feedback
O1  y1 feature code
O2  first-pass edit code from y1-x
O3  magnitude-only error/uncertainty proxy
O4  v1.0 unsigned U/D
O5  signed p only
O6  p + m
O7  p + m + d      # SRSC-Lite
O8  |p| + m + d    # sign control
O9  shuffled d     # train-time sample shuffle, retrain
O10 zero d
O11 equal-dimensional random noise
O12 equal-dimensional GT residual code
O13 full e diagnostic
O14 direct gt-y1 ceiling
```

## 10.1 所有 code 型 Oracle 的精确定义

所有固定投影均使用独立、预注册 seed 生成行正交矩阵，注册为不可训练 buffer；所有 code 使用 train-only robust normalization、相同8通道接口和相同多尺度计算顺序。

### O1：y1 feature code

O1 不直接复用 D1 隐藏特征，以免其通道容量远大于8。定义一个仅从 y1 构造的可部署低级描述子码：

```math
z_{y_1}=P_{y_1}\operatorname{unfold}_3(\psi(y_1)),
\qquad P_{y_1}\in\mathbb R^{8\times81}.
```

它用于回答“第二阶段仅看到压缩后的 y1 外观是否已经足够”。

### O2：first-pass edit code

```math
z_{edit}=P_{edit}\operatorname{unfold}_3(\psi(y_1)-\psi(x)),
\qquad P_{edit}\in\mathbb R^{8\times81}.
```

它不使用 GT，是可部署的 first-pass edit feedback。Predicted ladder无需再预测 O1/O2，因为它们可从 x/y1 直接确定性计算；主表中必须明确标为 deterministic feasible feedback。

### O13：full-e leakage diagnostic

`e` 是81维 GT-derived transverse residual。为通过相同接口，只允许：

```math
z_e=P_e e,
\qquad P_e\in\mathbb R^{8\times81}.
```

它必须命名为 `oracle_full_e_projected_diagnostic`，不得称为可部署 SRSC，也不得进入主方法结果。

### O14：direct GT correction ceiling

为区分“8维等信息对照”和“不压缩的绝对上限”，定义：

- `O12`：8维 `z_r=P_r unfold_3(psi(gt)-psi(y1))`，用于严格公平比较；
- `O14`：完整81维 direct correction，经一个为所有 ceiling 统一使用的 `81→8` **可训练** adapter 后进入D2，仅作为宽信息/可学习投影上限。

O14 的 adapter 参数必须用 dummy 参数补齐到其他 oracle 变体，或单独列在“non-capacity-matched ceiling”表中；不得与 O7 做公平增益主张。若希望核心比较完全无歧义，优先以 O12 为决定性 residual-code 对照。

## 10.2 O3 的准确含义

O3 不冒充 OPIR 官方实现。

定义为：

```math
u = robust_norm(sqrt(sum_81(unfold_k(psi(gt)-psi(y1))^2) + eps))
```

只放在第一个通道，其余补零，再通过相同 feedback adapter。

报告名：

```text
oracle error-magnitude / uncertainty proxy
```

Predicted 版本由相同 assessor 预测该 target。

OPIR 若可复现，仅作为独立外部方法比较。

## 10.3 O4 v1.0 U/D

使用同一投影几何构造：

```math
U = relu(p_raw)
D = m_raw + relu(-p_raw)
```

robust normalize 后放入前两通道，其余补零。

## 10.4 缺失通道

p-only、p+m 等版本：

- 不用不同维度 head；
- 始终 8 通道；
- 缺失部分固定为零；
- feedback adapter、D2 参数完全相同。

---

# 11. Predicted ladder

Oracle ladder 通过后，只训练以下 predicted 版本：

```text
P0 zero/no state
P3 predicted magnitude proxy
P4 predicted U/D
P5 predicted signed p
P6 predicted p+m
P7 predicted SRSC-Lite
P12 predicted residual code
```

所有 predictor：

- 相同 evidence 输入；
- 相同 backbone；
- 相同输出 8 通道；
- 相同参数数；
- 相同训练步数；
- 相同 state loss总尺度；
- 相同 seed。

输入：

```text
x.detach()
y1.detach()
(y1-x).detach()
stopgrad(F1..F4)
```

---

# 12. State Assessor

轻量、多尺度、确定性。

## 12.1 输入 stem

full resolution：

```text
concat(x, y1, y1-x) = 9 channels
→ 3x3 conv 32
→ GELU
→ 3x3 conv 32
```

逐尺度 stride-2 建立 evidence pyramid。

每尺度：

```text
evidence feature
+ 1x1 compressed stopgrad(Fi)
→ concat
→ 1x1 fuse
→ two depthwise-separable residual conv blocks
→ 8-channel state head
```

## 12.2 输出约束

针对不同 target 采用统一线性 8-channel head。  
不要在模型内部写死不同激活；target normalization 统一后使用线性输出。

需要符号的通道不做 ReLU。

## 12.3 梯度

- state loss 不得更新 E/D1；
- assessor evidence全部 detach；
- Stage B 中 E/D1 冻结；
- Stage C 中 final restoration loss可通过 y1 residual path 和 D2 evidence path按设计更新 E/D1；
- target builder始终使用 `y1.detach()`。

---

# 13. SRSC-Mod

每个 D2 尺度：

```python
state_feat = Conv3x3(8, c_state)
state_feat = GELU(state_feat)
state_feat = Conv3x3(c_state, c_state)

gamma = gamma_head(state_feat)
beta  = beta_head(state_feat)

gamma = gamma_scale * tanh(gamma)
Fmod = (1 + gamma) * Ffuse + beta
```

默认：

```text
gamma_scale = 0.1
```

gamma/beta 最后一层权重与 bias 全零初始化。

SRSC-Mod 只在：

```text
skip/y1 fusion之后
Restormer blocks之前
```

不放进 encoder，不替换 MDTA/GDFN，不直接乘到最终 RGB residual。

---

# 14. Loss

## 14.1 Restoration

遵循官方主 loss。若官方主 loss 是 L1：

```math
Lcoarse = L1(y1, gt)
Lfinal = L1(y2, gt)
Lrest = 0.5 Lcoarse + 1.0 Lfinal
```

不能仅给 SRSC 使用额外 perceptual/GAN/SSIM loss。

## 14.2 State

所有 target 先用 train-only robust statistics 做逐通道归一化。

为避免不同表示使用任意不同尺度的监督，所有 predicted 变体共享基础逐通道损失：

```math
L_{base}=\frac{1}{8}\sum_{c=1}^{8} SmoothL1(\hat z_c,z_c).
```

- magnitude proxy、U/D、residual code和其他一般8通道 code：使用 `Lbase`；
- p-only、p+m等缺失通道的 target 固定为0，8个输出通道仍全部参与 `Lbase`；
- SRSC 的有效 target 为 `S_eff`，其所有8通道同样参与 `Lbase`；
- 下述 cosine 项只作为方向几何的预注册附加项，不能替代基础8通道损失。

```math
Lp = SmoothL1(pred_p_tilde, p_tilde)
Lm = SmoothL1(pred_m_tilde, m_tilde)
Ld = w_dir * [1-cos(pred_d_tilde,d_tilde)]
```

```math
Lstate_srsc = Lbase + lambda_dir_cos * Ld
```

默认 `lambda_dir_cos=0.1`。Residual-code target与其他 code使用相同 `Lbase`。pilot 中记录每个分量与 assessor head 的梯度范数；只允许对所有表示统一缩放 `lambda_state`，不允许根据 validation PSNR 为某一表示单独调 loss。

cosine 必须用 `eps` 安全实现；当 `w_dir<1e-3` 或 `||d_tilde||<1e-6` 时该位置的 cosine 项置0，避免零向量产生未定义方向。基础 `Lbase` 仍保留。

默认：

```text
lambda_state = 0.1
```

在 pilot 中只允许检查梯度量级并做一次全表示统一缩放；不得按表示或任务调。

## 14.3 Clean preservation

```math
Lclean = mean((1-q) * |y2-y1|)
```

对所有反馈变体统一开启或统一关闭，不能只给 SRSC。

---

# 15. 三阶段训练与闸门

## 15.1 Stage 0：官方 PromptIR parity

- 官方 checkpoint；
- official test；
- 差异 <=0.10 dB；
- 否则停止。

## 15.2 Stage A：Coarse E+D1

Primary 结果必须从头训练 clean E+D1。  
官方 PromptIR/Restormer权重只允许用于 smoke/warm-start pilot，不能与 from-scratch结果混为主结论。

训练到 locked_val 收敛，保存：

```text
best_val
last
```

冻结 best_val，生成 train/val/test y1 cache，带 SHA256。

## 15.3 Stage B1：Oracle ladder（kill experiment）

- 冻结 E/D1；
- 不训练 assessor；
- 每个 oracle feedback 使用相同 D2 template与初始化；
- 先 1 seed short pilot；
- 只在 pilot数据管线正确后跑正式 locked_val；
- official test不参与筛选。

### 分层执行，禁止一次跑完 15 个大模型

Tier 1：

```text
O0, O3, O4, O5, O6, O7, O12
```

若 O5 不优于 O4，则停止 sign 主张。  
若 O7 不优于 O6，则停止 direction 主张。

Tier 2 仅在对应主张通过后：

```text
O8, O9, O10, O11, O13, O14
```

### Oracle GO 参考线（工程闸门，不是理论定理）

以下仅是 `SCIENTIFIC_GO` 的早期信息价值参考线，用来决定是否值得进入 predicted/joint 阶段；它们不是论文成功标准，不得把 `+0.02/+0.03 dB` 写成最终有效涨点。

在 locked_val 上：

1. `O5 - O4`
   - five-setting mean >= +0.03 dB；
   - 三个 denoise 均不低于 -0.01 dB；
   - 至少两个 denoise >= +0.02 dB。

2. `O7 - O6`
   - five-setting mean >= +0.02 dB；
   - 方向打乱/zero后增益明显回落；
   - 不能仅由 dehaze贡献。

3. O7 相对 O3/O2：
   - 五设置平均为正；
   - 每任务 median delta非负。

若不满足，写 NO-GO，不进入 predicted ladder/full train。

## 15.4 Stage B2：Predicted ladder

只在 Oracle GO 后执行。

先训练：

```text
P3, P4, P5, P6, P7, P12
```

### Predicted GO

在 locked_val 上：

- P7 高于 P6，且方向控制成立；
- P7 高于 P4/P3；
- P7 相对 P12：
  - 平均至少 +0.02 dB，或
  - 在 leave-one-severity / cross-task / OOD验证中稳定更优；
- predicted P7 捕获 oracle information gain 的 >=40%；
- 三个 denoise setting不系统性下降；
- 3 seeds至少方向一致。

若 P12 与 P7相同或更好且无泛化差异，优先 residual code，SRSC 主张 NO-GO。

## 15.5 Stage C：Joint fine-tune

只有 `decision.json` 标记：

```text
ORACLE_GO=true
PREDICTED_GO=true
```

才允许。

- E/D1 lr = 主 lr 的 0.1；
- assessor/D2 主 lr；
- state evidence/target detach；
- 只用 locked_val early stop；
- official test配置冻结后运行一次。

## 15.6 Publication GO：论文级最终闸门

Stage-C 完成后，使用一个 checkpoint、一个推理协议和预先锁定配置，与最强公平内部 baseline 比较。只有同时满足以下条件，`PUBLICATION_GO=true`：

1. AIO-3 five-setting mean PSNR 提升 `>= +0.30 dB`；
2. 每个设置（BSD68 σ15/25/50、Rain100L、SOTS）PSNR 提升均 `>= +0.10 dB`；
3. SSIM 不出现系统性下降；任何设置下降超过预注册数值容差都必须标记；
4. 三个 seed 的核心方向一致，并报告 per-image paired bootstrap 95% CI；
5. 提升不能主要由 dehaze 单项驱动，三个 denoise setting不能被平均数掩盖；
6. `P7 > P6`、direction 负控制、`P7 vs P12` 的 Stage-B 机制结论在 joint model 上仍成立；
7. 至少在一个未用于选择配置的 paired-real、跨任务、leave-one-severity、OOD或局部复合退化评估中表现出明确优势；
8. 参数、MACs、显存和延迟代价得到完整报告，并与 two-stage-no-state、参数匹配 baseline公平比较。

若平均达到 `+0.30 dB` 但任一设置低于 `+0.10 dB`，只能标记 `PROMISING_NOT_PUBLICATION_GO`；不得通过手工合并不同任务 checkpoint、task-specific calibration、TTA或挑 seed 补齐。

若未达到上述性能门槛，但 Stage-B 机制证据非常稳定，只能结论为：

```text
Scientific hypothesis supported; publication-strength restoration gain not established.
```

只有 `SCIENTIFIC_GO=true` 且 `PUBLICATION_GO=true`，才允许称 SRSC-Lite 为最终论文主方法。该门槛用于项目决策，不构成对任何会议录用的保证。

---

# 16. 负控制必须从训练开始

必须独立完整训练：

- signed p vs |p|；
- p+m vs p+m+d；
- d 在 train batch 内跨样本打乱；
- zero d；
- equal-dimensional random noise；
- fixed random P vs train-only PCA P；
- cross-image d；
- matched residual code predictor。

禁止仅在测试时旋转方向或翻转符号后宣称机制成立。

---

# 17. 等容量、等计算对照

必须至少包含：

1. 单阶段 Restormer-AiO；
2. 两阶段 no-state，结构与主方法完全相同；
3. 两阶段 y1 feature/residual；
4. 参数匹配的 widened/deepened two-stage baseline；
5. SRSC-Lite。

对于 feedback 变体：

```text
params difference < 0.5%
MACs difference < 0.5%
same D2
same interface
same schedule
same seed
```

报告：

```text
params
MACs@256
peak VRAM
latency@256
training GPU-hours
```

---

# 18. 单元测试

必须真实运行：

1. `[1,3,128,128]` 与 `[2,3,127,131]` 前向；
2. 输出尺寸等于输入；
3. `model.eval(); model(x)` 只接受 x；
4. inference graph不调用 target builder；
5. prompt/MoE关键词与参数完全不存在；
6. F/G/S所有尺度 shape正确；
7. feedback始终8通道；
8. zero-init时 Fmod与Ffuse误差 <1e-6；
9. state loss backward时 E/D1 grad为None，assessor有grad；
10. Stage-C final loss时D2有grad，E/D1按设计有grad；
11. `P.shape==(6,81)`、`Pr.shape==(8,81)` 且 `P @ P.T≈I`、`Pr @ Pr.T≈I`；
12. y1=gt时 p≈0,m≈0；
13. y1=x+t(gt-x), 0<t<1 时 p>0；
14. t>1 时 p<0；
15. 有效区域中 `<e,vstar>` 的相对误差低于预注册容差；加入与 vstar正交扰动时 m增大；
16. clean区域无NaN/Inf；
17. AMP forward/backward稳定；
18. checkpoint roundtrip一致；
19. oracle与predicted checkpoint命名明确；
20. official-test lock生效。

输出真实：

```bash
pytest -q
```

日志。

---

# 19. 单卡运行策略

自动探测最大安全 micro-batch：

- BF16优先；
- gradient checkpoint可配置；
- 使用 gradient accumulation保持所有变体相同 effective batch；
- OOM时只允许减 micro-batch并增加accumulation；
- 不改变 crop、模型、loss或数据比例。

每 500/1000 step：

- 保存 last；
- 定期 locked_val；
- top-3 best；
- 写 `RUNNING_STATUS.md`。

使用 tmux：

```bash
tmux new-session -d -s srsc_v12 \
"cd /root/autodl-tmp/srsc_lite_v12 && \
 python scripts/orchestrate.py 2>&1 | tee artifacts/logs/orchestrate.log"
```

支持 resume，保存 optimizer/scheduler/scaler/RNG。

---

# 20. 统计

所有关键比较使用 per-image paired difference：

- mean；
- median；
- win rate；
- worst 10%；
- 10,000 次 bootstrap CI；
- denoise sigma15/25/50分别；
- 不只报平均。

定义：

```math
G_oracle = metric(O7) - max(metric(O2), metric(O3), metric(O4), metric(O12))
```

但解释时必须拆分：
- SRSC相对 magnitude/residual/U-D；
- SRSC相对 equal-dimensional residual code。

可学习恢复比例：

```math
rho = [metric(P7)-metric(best predicted baseline)] /
      [metric(O7)-metric(best oracle baseline)]
```

当分母 <=0 时，rho标记为 undefined，不得报告夸张比例。

---

# 21. 报告与决策

`reports/decision.json`：

```json
{
  "promptir_parity": "PASS|FAIL",
  "stage_a": "PASS|FAIL|INCOMPLETE",
  "oracle_sign": "GO|NO_GO|INCOMPLETE",
  "oracle_direction": "GO|NO_GO|INCOMPLETE",
  "predicted_srsc": "GO|NO_GO|INCOMPLETE",
  "scientific_go": "GO|NO_GO|INCOMPLETE",
  "publication_go": "GO|PROMISING_NOT_PUBLICATION_GO|NO_GO|INCOMPLETE",
  "residual_code_control": "SRSC_BETTER|RESIDUAL_BETTER|TIE|INCOMPLETE",
  "selected_model": "SRSC_LITE|SIGNED_PROGRESS_ONLY|UD_V1|RESIDUAL_CODE|NO_GO",
  "per_task_deltas": {},
  "params": {},
  "macs": {},
  "gpu_hours": 0.0,
  "blocking_issues": [],
  "next_command": ""
}
```

`FINAL_DECISION.md` 必须回答：

1. PromptIR与评价协议是否复现？
2. prompt是否被彻底删除？
3. 两阶段增益是否只是容量？
4. signed p是否优于unsigned U/D？
5. direction是否在重训控制下有独立贡献？
6. SRSC oracle是否只是GT带宽？
7. predicted SRSC是否优于matched predicted residual code？
8. 三个denoise是否均成立？
9. 结论是否被dehaze主导？
10. 选择SRSC、signed-only、v1.0、residual code还是NO-GO？
11. 是否允许Stage-C？
12. 下一条真实命令是什么？
13. 是否达到 average `+0.30 dB`、每设置 `+0.10 dB` 和 SSIM guardrail？
14. 当前结论是 scientific-only，还是 publication-strength？

---

# 22. 最终交付

必须有：

```text
reports/AUDIT.md
reports/ARCHITECTURE.md
reports/BASELINE_PARITY.md
reports/STAGE_A_REPORT.md
reports/STAGE_B_ORACLE_REPORT.md
reports/STAGE_B_PREDICTED_REPORT.md
reports/FINAL_DECISION.md
reports/decision.json

src/net/...
configs/...
scripts/...
tests/...
artifacts/metrics/metrics_long.csv
artifacts/logs/
```

打包：

```bash
tar -czf srsc_lite_v12_results.tar.gz \
  src configs scripts tests reports artifacts/metrics artifacts/plots \
  RUNNING_STATUS.md STOP_REASON.md
```

最后只允许声称实际完成的内容。

如果只完成实现和 smoke test：

```text
Implementation complete; method efficacy not yet established.
```

如果 Oracle NO-GO：

```text
SRSC information hypothesis rejected under the preregistered protocol.
```

如果 Oracle 与 Predicted均GO：

```text
SRSC-Lite passed the preregistered information-value and learnability gates.
```

现在开始执行：

1. 审计；
2. 官方 PromptIR parity；
3. 重构 Restormer-AiO；
4. 完成单元测试；
5. Stage-A；
6. Tier-1 Oracle ladder；
7. 只在闸门通过后继续。

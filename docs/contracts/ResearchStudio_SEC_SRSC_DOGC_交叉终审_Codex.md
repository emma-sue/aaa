# Codex × Microsoft ResearchStudio 交叉终审
## SEC-IR、SRSC-Lite、DOGC 与二次独立 Idea Spark 重跑的最终结论

> 文件性质：**Codex 的 ResearchStudio 交叉审核与最终综合结果**  
> 日期：2026-07-18  
> 范围：标准已见退化设定下的 all-in-one image restoration；常规、固定深度、端到端判别式模型；不以 unseen degradation 为角度；不使用 agent、VLM/LLM、MoE、工具库、动态 stopping 或多轮调度。

---

## 0. 板上钉钉的结论

### 0.1 不建议把 Claude 的 SEC-IR 直接变成主方法

SEC-IR 的表层故事非常吸引人：让复原器在测试时通过自探针检查自己的状态评估是否可信。但它目前不是可直接实施的最终方法，因为其主探针保证在随机 mask 下不成立，`under/over` 也并非真正的有符号坐标。这两点会同时动摇它的状态语义、信任分数、信任门控和因果解释。

### 0.2 不建议用这次独立重跑的两个新候选替换 SRSC

Codex 重新独立执行了一遍 ResearchStudio Idea Spark，并启用了 ArXiv、OpenAlex、Semantic Scholar 和 OpenReview 四个连接器。ResearchStudio 允许的两次尝试均被 Phase 3 硬性否决，最终没有任何候选进入 Phase 4：

1. **Counterfactual Utility**：用 `7×7` 共同动作生成标签，却用该标签控制单像素动作，存在致命的计算粒度错配。
2. **Secant-State Carrier**：`midpoint–secant` 本身只是 `[f(x),f(y1)]` 的可逆正交旋转；候选方案又没有定义清楚 `2C→C` 的载体接口和多尺度到 Haar 输出头的数据流，同时触发 ResearchStudio 明示的 `audit_decomp_operator` 高风险组合。

### 0.3 当前最佳路线

> **保留 SRSC-Lite 作为唯一代码主线和数学核心；不中断当前 Stage-A。在 Stage-B 中以严格增量对照引入经修正的 DOGC “等距欠修/过修反事实对”和“符号—动作一致性”。只有两者分别稳定增益，才把它们升级为论文创新点。**

我建议的工作名是：

> **CG-SRSC: Counterfactually Grounded Signed Restoration-State Correction**  
> **反事实锚定的有符号复原状态校正**

这是一个**待实验证伪的最佳候选**，不是已经被证明能中顶会的方法。

---

## 1. ResearchStudio 实际执行与证据边界

### 1.1 Claude SEC-IR 这一跑是真实的 ResearchStudio 跑法

- `lit_grounding_mode=real`；
- Phase 0 表格共 22 条文献；
- 使用 ArXiv、OpenAlex、Semantic Scholar；
- OpenReview 当时被记录为 `skipped`；
- 完成了 Phase 1–4 文件链路。

因此，SEC-IR 不是随口生成的点子。但“Phase 4 文件已生成”只能说明它通过了当次工作流的结构性闸门，不等于其数学推导已经被独立证明。

### 1.2 Codex 交叉重跑

新跑路径：

```text
/root/autodl-tmp/ideaspark_run/allinone-restoration-state-crossaudit-codex
```

这次使用了全部四个可用检索连接器，并把 RAR、DFC-IR 和 AgenticIR 作为用户给定锚点尝试深挖。其中 RAR 和 DFC-IR 取得全文缓存；AgenticIR 全文抓取失败，因此新跑对 AgenticIR 不做超出现有证据的实现级声称。

新跑的 Phase 1 独立支持了原始问题：

> 第一遍复原后，相同的中间响应或误差幅度可能分别意味着“继续更新有益”或“继续更新会损伤内容”；因此，静态输入侧条件不一定是第二阶段的充分条件。

但 ResearchStudio 没有因为问题成立就强行让新方法通过。两次候选都在 Phase 3 被 `hard_floor abandon`，重跑最终状态为：

```text
TERMINAL — Phase 3 audit abandoned; retry budget exhausted.
No candidate advances to Phase 4.
```

这一结果比勉强生成一张“看起来很新”的 idea card 更有价值。

---

## 2. 对 SEC-IR 的数学与架构审核

### 2.1 真正有价值的地方

SEC-IR 提出了一个很好的研究问题：

> 测试时无 GT 的情况下，一个复原状态评估器能否检查自己的预测是否可信？

这个问题比单纯“再加一个 assessment head”更吸引审稿人。它的 `zero / permute / swap / oracle` 干预表也值得保留为论文的补充机制审计。

### 2.2 `under/over` 不是“有符号双通道”

SEC 的定义是：

```math
\alpha=\operatorname{clamp}\!\left(
\frac{\langle e,d\rangle_W}{\langle d,d\rangle_W+\epsilon},0,1
\right),
```

```math
S_{under}=\alpha\,\operatorname{RMS}_W(d),\qquad
S_{over}=\operatorname{RMS}_W(e-\alpha d).
```

`alpha` 被截断到 `[0,1]`，两个 RMS 又都非负。所以这是“两类非负幅度”，不是能表示欠修和过修反号的 signed coordinate。

更重要的是，当原始投影系数落在 `[0,1]` 之外时，clamp 后的 `e-alpha d` 不再与 `d` 正交。因此 `over` 不能严格被解释为“正交的复原器自伤”。即使不 clamp，正交余量也只是 **target-relative transverse residual**，不是真实物理意义上可唯一归因给复原器的内容损伤。

### 2.3 自探针的主保证在随机 mask 下不成立

SEC 声称将 `y1` 在 mask `M` 中向退化输入回混后：

```math
\langle tM(d-e),d\rangle_W
=t(1-\alpha)\langle d,d\rangle_W\ge0.
```

但实际左边是：

```math
t\langle Md,d\rangle_W-t\langle Me,d\rangle_W.
```

用整个窗口定义的 `alpha` 不能代换两个带 mask 的内积。再加上 clamp，等式更不成立。因此，“回混必然使 under 升高”不是建立在现有定义上的构造性保证。

而且，如果当前区域已经过修，向退化输入回混可能恰好是一个有益的 rollback，正确的状态响应本就不应被预设为“under 必然上升”。

这个问题会传递到：

- `phi` 是否真的表示状态准确性；
- `L_probe` 是否在训练真实状态可信性；
- `g=max(0,phi)` 是否应该控制 D2；
- 信任门控是否真的“风险可控”。

### 2.4 其他必须修正的问题

1. Gaussian blur 和 channel shift 不能保证只落到 `over` 通道；Claude 跑的最终可实现性文件也把它标记为 `severity=open`。
2. 具体 backbone、尺度、宽度、D2 如何消费 y1 在最终卡片中仍是 `severity=open`。
3. 四个 probe 需要基准 assessor 前向加四个扰动 assessor 前向，不是普通的单前向推理。
4. 若 `phi` 是信任，而 `state_error` 是误差，应预注册负相关，例如 `rho(phi,error)<=-0.6`；`|rho|>=0.6` 会错误地接受“越自信越错”。
5. `swap < ablate` 可以是有用的反事实现象，但两套自由 FiLM heads 产生的不对称，并不自动证明它们学到了 `continue/restore` 的语义。

### 2.5 对 SEC 的客观处置

- **不把自探针信任门控放入当前主方法。**
- `zero / shuffle / swap / oracle` 可保留为补充审计，但不算架构创新点。
- 若以后重做 probe，只能先称其为“合成扰动下的局部响应一致性”，不能在没有新的可识别性证明前声称它验证了未扰动状态的真实准确性。

---

## 3. 二次独立重跑为什么没有产生更好方案

### 3.1 尝试 1：State Before Update / Counterfactual Utility

它希望从一个预计校正 `d` 构造 `{-1,0,+1}` 的局部效用，学习 rollback / keep / apply。问题是：

```text
标签：对 7×7 全部 49 个像素施加同一动作后的收益
部署：该位置的 gate 只改变中心像素动作
```

一个极端但合法的例子是：中心像素的 `+d` 有害，另外 48 个像素的 `+d` 有益。补丁标签会给 `+1`，但实际部署的中心像素变差。这不是调一个 loss 就能修好的小 bug，而是需要改变动作单位的 gap-level redesign，因此 ResearchStudio 依规则硬性放弃。

### 3.2 尝试 2：Secant-State Carriers

原始变换：

```math
m=(f(x)+f(y_1))/\sqrt2,\qquad
s=(f(y_1)-f(x))/\sqrt2.
```

这是 `[f(x),f(y1)]` 的可逆 Hadamard 旋转。它能保能量，但不凭空增加信息。要让它成为创新，必须证明这个坐标系对后续压缩、可识别状态或优化动力学产生了不可由直接 concat 取代的效果。

但候选方案同时存在：

- 原始 carrier 是 `2C` 通道，固定预算又声称为 `C`；
- 它既可解释为先压缩再重建回 `2C`，又可解释为不重建、直接消费 `C` 维 code；
- 多尺度 carrier 没有被明确映射到 `12×H/2×W/2` 的 Haar 校正头；
- `|z·∂L/∂z|` 影响度只是常见敏感度 proxy，没有异质性分布证据、实际延迟/显存指标或压缩误差界；
- C12 + C04 组合正好命中 ResearchStudio 的 `audit_decomp_operator` 反模式，但候选没有给出规定的独立识别实验。

因此内部 retry 也被 `hard_floor abandon`。

---

## 4. 五个方案的客观比较

| 方案 | 故事吸引力 | 数学闭合性 | 架构可执行性 | 主要风险 | 终审处置 |
|---|---|---|---|---|---|
| SEC-IR | 原始 hook 最亮：无参考自证可信性 | 不通过；主 probe 保证错误，双通道非 signed | 有两个 open 实现空洞，且约 5 次 assessor 前向 | 信任门控整条主张依赖错误保证 | 不做主方法；干预审计可作补充 |
| Counterfactual Utility | 动作故事直接 | 动作粒度不一致 | 无合法 per-pixel oracle | Review Residuals 等更广泛的 update gate 碰撞 | ResearchStudio 已放弃 |
| Secant-State Carrier | “实际转移而非输入 prompt”有趣 | 变换可逆，但主方法数据流未定义 | 不可直接写成唯一模型 | 可逆重参数化、压缩 proxy、Transport Keys 别名碰撞 | ResearchStudio retry 已放弃 |
| DOGC | **可防守故事最强**：等距反事实 + continue/rollback/redirect | 状态坐标与 SRSC 高度等价；原 `q` 无法保证所有重叠局部窗口正交 | 反事实与三头路由需修正 | 二维状态没有 redirect 方向；三头容量混杂 | 不单独换轨；只吸收可严格定义的增量 |
| SRSC-Lite | 故事需加强，但“符号进展 + 横向偏离”清楚 | **当前最强**；坐标、投影、空间粒度一致 | **当前最强**；仓库、宽度、接口、loss、tests、gate 全部固定 | 6D 只是 81D 横向方向的低维 sketch；可能不胜直接 residual code | 作为唯一实施主线 |

一句话概括：

> **SEC 赢在未校验的 hook；DOGC 赢在可防守的故事结构；SRSC 赢在数学严谨性和实施完整性。最合理的综合不是三套全堆上，而是以 SRSC 为核心，只吸收 DOGC 中能被严格修正的两项。**

---

## 5. 最终候选 CG-SRSC 的数学核心

记：

- `X`：退化输入；
- `Y`：干净 GT，仅训练时可见；
- `Y1`：第一遍粗复原结果；
- `psi`：RGB + 固定 Sobel-x/y 线性描述子；
- `U`：`3×3` 局部 unfold，得到 81 维 patch vector。

定义：

```math
v^*=U[\psi(Y)-\psi(X)],\qquad
v_1=U[\psi(Y_1)-\psi(X)].
```

```math
\alpha=\frac{\langle v_1,v^*\rangle}{\|v^*\|^2+\epsilon},
\qquad p=1-\alpha,
```

```math
e_\perp=v_1-\alpha v^*,
\qquad
m=\frac{\|e_\perp\|}{\|v^*\|+\epsilon},
\qquad
d=P\frac{e_\perp}{\|e_\perp\|+\epsilon}.
```

其中 `P` 是固定的 `6×81` 行正交投影。主状态仍为：

```text
S = [signed progress p, transverse magnitude m, projected direction d1..d6]
```

这组坐标有一个关键而可防守的动作恒等式：

```math
U[\psi(Y)-\psi(Y_1)]
=v^*-v_1
=p\,v^*-e_\perp.
```

它严格说明：

- `p>0`：沿目标方向还没走完，需要 continue；
- `p<0`：沿目标方向已经走过头，需要 rollback；
- `e_perp!=0`：需要用 `-e_perp` 做 transverse redirect。

这是比 SEC 的非负 `under/over RMS` 更强、也比 DOGC 只保留偏离幅度更完整的理论核心。它只应被称为 **target-relative local edit coordinates**，不应夸大成真实物理退化的唯一分解或最优策略的充分统计量。

---

## 6. 最终模型架构

### 6.1 总体拓扑

```text
                              ├───> D1 coarse decoder ───> Y1
X ───> shared encoder E ───> {F1,F2,F3,F4}
                              │
Y1 ───> shallow pyramid ───> {G1,G2,G3,G4}

(X, Y1, Y1-X, stopgrad(F)) ───> assessor A ───> {S1,S2,S3,S4}

{F,G} fusion ───> signed-state modulation ───> D2 correction decoder
             ───> Delta2 ───> Y2 = Y1 + Delta2
```

固定为：

```text
one shared Restormer-style encoder, executed once
one light coarse decoder D1
one deterministic multi-scale assessor A
one correction decoder D2
fixed K=2
no recurrence
no dynamic stopping
no tool library
no extra full encoder for Y1
```

`Y1` 只经过 shallow pyramid，D2 复用 E 的 skips，这比 SEC 卡片中未决定的拓扑更清楚，也比对 X/Y1 各跑一次完整 encoder 更省。

### 6.2 为什么“一个 encoder + 两个 decoder”不是问题

它不应被画成两个无关 U-Net，而应被画成一个共享表征的 **predictor–corrector**：

```text
D1 = cheap predictor that creates an explicit image-level state Y1
D2 = state-conditioned residual corrector
```

若只用一个常规 decoder，网络中没有一个完整 `Y1` 供 assessor 判断“第一次编辑的后果”。若使用循环共享块，又会把状态价值与迭代次数、权重共享和 stopping 混在一起。因此，固定两阶段是针对当前科学问题的最干净实验载体，不需要因为它相对少见就换掉。

### 6.3 状态在 D2 中的唯一插入位置

每个 D2 尺度统一采用：

```text
upsampled decoder feature
→ concat encoder skip Fi
→ concat Y1 shallow feature Gi
→ 1x1 fusion
→ state modulation
→ Restormer blocks
```

状态不插入 encoder，不改 MDTA/GDFN，不直接乘最终 RGB residual。D2 只预测 correction residual，不从头重建整张 Y2。

---

## 7. 建议冻结的三个创新点候选

### 创新点 1：Strict Signed Restoration-State Coordinates

将第一遍复原的后果表示为同一 81 维局部空间中的：

```text
signed progress + transverse magnitude + transverse direction
```

它不是一个退化类别 prompt，也不是单标量 IQA，而是输出经历一次真实编辑后的局部动作坐标。所有内积、范数、投影和方向都在同一粒度下计算，避免 SEC 和第一个新候选的粒度/符号问题。

### 创新点 2：Matched-Distance Under/Over Counterfactual Pair

DOGC 原始三元组里的 `wrong-direction q` 不能保证每个重叠局部窗口都正交且等范数，因此不直接纳入主创新。先使用可严格成立的 under/over 等距对：

```math
Z^- = X+(1-r)(Y-X),\qquad
Z^+ = X+(1+r)(Y-X),\qquad 0<r<1.
```

它们到 GT 的距离完全相同：

```math
\|Z^- -Y\|=\|Z^+ -Y\|=r\|Y-X\|,
```

但其符号进展分别为：

```math
p(Z^-)=+r,\qquad p(Z^+)=-r.
```

因为 `psi` 是线性描述子，上述对称性在局部描述子空间中同样成立。主反事实分支不对 `Z+` hard clamp，选用小 `r`（例如 0.05–0.10）并报告超出归一化输入范围的像素比例；如果必须限制输入范围，则只在未受 clipping 影响的有效窗口上计算辅助损失，且 clipping 版本不得用于声称全图精确等距。

它的作用不是增加 GT 带宽，而是构造一个强对照：

> **误差幅度完全相同时，assessor 是否真能分清需要 continue 还是 rollback？**

它只作为 assessor 的训练期辅助和独立 kill experiment；不改测试输入，不引入工具，不破坏标准 AIO 数据协议。

### 创新点 3：Sign-Equivariant State-to-Action Correction

当前 SRSC-Mod 可以用一个自由 CNN 把 8 通道状态转成 affine modulation，但自由 CNN 不保证符号反转会对应动作反转。建议把状态分成：

```text
odd coordinates:  p, d1..d6
even coordinate:  m
```

在每个 D2 尺度使用奇偶分支：

```math
\gamma_o=0.1\tanh(W_{\gamma,o}S_o),\qquad
\beta_o=0.1\tanh(W_{\beta,o}S_o),
```

```math
\gamma_e=0.1\tanh(h_{\gamma,e}(m)),\qquad
\beta_e=0.1\tanh(h_{\beta,e}(m)),
```

```math
F'=[1+\gamma_o+\gamma_e]\odot F+\beta_o+\beta_e.
```

`W_o` 使用无 bias 线性卷积，因此对 `p/d` 的全部反号，odd branch 严格反号；`m` 分支保持不变。最后一层全部零初始化，初始时精确退化为无状态 D2。

同时在训练期用动作恒等式约束 D2 的实际 correction：

```math
c_2=U[\psi(Y_2)-\psi(Y_1)],
```

```math
a_2=\frac{\langle c_2,v^*\rangle}{\|v^*\|^2+\epsilon},
\qquad c_{2,\perp}=c_2-a_2v^*.
```

```math
L_{action}=\operatorname{SmoothL1}
\left(a_2,p\right)
+w_{dir}\left[1-\cos(c_{2,\perp},-e_\perp)\right].
```

该损失只在训练时使用 GT，推理图不变。论文主张应限定为：

> 符号状态不只被预测，其奇偶结构还在校正接口中被保留，并与实际输出动作对齐。

它比 DOGC 的三个大 residual heads 更容易做等容量对照，也比 SEC 的两套自由 FiLM heads 更能支撑符号语义。但这是新的增量候选，必须与原 SRSC-Mod 做同容量重训；如果不增益，就不能作为论文创新点。

---

## 8. 应该如何在当前实验上落地

### 8.1 Stage-A 完全不变

当前正在训练的 `E+D1` 是 SRSC、DOGC 和 CG-SRSC 的共同地基。不应因为本次思路审核停掉、重跑或改它。

### 8.2 Stage-B 必须按以下顺序，不能一次堆满

#### Gate 1：先证明 SRSC 核心信息价值

```text
signed p > unsigned U/D
p+m+d > p+m
shuffled/zero d destroys the direction gain
predicted SRSC > matched predicted 8-channel residual code
```

若这一层不过，不应加 DOGC 或 SEC 来掩盖核心失败。

#### Gate 2：单独测反事实锚定

```text
B0: original predicted SRSC
B1: B0 + matched-distance under/over pair supervision
B2: B1 + distance-matched but sign-shuffled labels
```

要求 B1 稳定优于 B0，且 B2 的收益回落。否则反事实只是普通数据增强或额外训练量。

#### Gate 3：单独测符号—动作接口

```text
C0: B1 + current unrestricted SRSC-Mod
C1: B1 + sign-equivariant modulator
C2: C1 without L_action
C3: C1 with sign/direction channels shuffled from training start
```

要求 C1 优于 C0，且 C2/C3 按预注册方向回落。只有这时，创新点 3 才成立。

### 8.3 不能缺的公平对照

- 单阶段 Restormer-AiO；
- 等容量 two-stage-no-state；
- `Y1-X` deterministic edit code；
- unsigned magnitude/uncertainty；
- SRSC `p`、`p+m`、`p+m+d`；
- 同为 8 通道的 direct GT residual code，oracle 和 predicted 分开；
- 参数、MACs、训练步数、assessor、D2、feedback width 全部匹配；
- 反事实和动作分支必须分开加入，不允许只报全堆满结果。

### 8.4 决策规则

```text
SRSC core fails                    -> residual-code fallback or NO-GO
SRSC passes, counterfactual fails  -> keep original SRSC
counterfactual passes, action fails-> SRSC + counterfactual only
all three pass                     -> freeze CG-SRSC as paper method
```

SEC probe 不参与这个主决策树。

---

## 9. 顶会范式的动机段落

> 现有 all-in-one 图像复原方法通常把退化建模为输入的静态属性：在复原开始前从退化图像中生成 prompt、路由或不确定性，然后在后续计算中重复使用。然而，第一次复原不仅减少了退化，也可能越过正确解或引入与目标方向正交的新偏离。因而，相同的局部误差幅度可以对应三种相反的动作：继续复原、撤销过度编辑，或校正偏离方向。标量质量分数、无符号不确定性和仅由原输入决定的退化表征无法识别这一动作歧义。我们因此将固定两阶段复原重新表述为有符号状态下的 predictor–corrector：第一阶段产生一个显式中间估计，第二阶段不再重复使用静态退化提示，而是根据目标相对的有符号进展与横向偏离执行校正。为防止评估器只读取误差大小，我们构造到干净目标等距但所需动作相反的欠修/过修反事实对，并在校正接口中显式保留符号坐标的奇偶结构。这使“评估”不再是一个可被旁路绕过的辅助头，而成为可由等距反事实、符号翻转和方向打乱实验直接证伪的复原状态。

---

## 10. 论文级宣称边界

### 目前可以说

- 独立 ResearchStudio 重跑支持“第一遍后状态未知”是一个真正的前沿残留问题；
- SRSC 是目前唯一个数学闭合且实现规格完整的候选；
- DOGC 的等距反事实思路是最值得吸收的故事和监督升级；
- SEC 的自验证问题很有价值，但它的当前 probe 不能支撑对未扰动状态可信性的声称。

### 目前不能说

- CG-SRSC 已经比 SRSC、PromptIR、R2R、DFC-IR、RAR 或其他 SOTA 更强；
- 反事实监督或 sign-equivariant 接口已经带来涨点；
- 6D 随机投影是横向 81D 方向的充分统计量；
- 当前方法已达到顶刊顶会强度；
- ResearchStudio 的文献检索已经等价于绝对穷尽所有相关工作。

最终只有下列证据同时成立时，才适合将 CG-SRSC 冻结为投稿主方法：

1. SRSC 相对等维 residual code 有稳定机制或泛化优势；
2. 等距反事实监督对真实 Y1 的 predicted-state 有稳定增益；
3. sign-equivariant interface 与 `L_action` 各自有独立贡献；
4. 符号和方向负控制成立；
5. 参数、MACs、latency、VRAM、训练步数和数据完全公平；
6. Stage-C 恢复指标达到预注册的 publication gate，而不只是 oracle 或短程 pilot 更好。

---

## 11. 最终决策

### 现在不做的事

- 不停 Stage-A；
- 不重写当前 E+D1；
- 不将 SEC 的四探针和信任门控直接堆进模型；
- 不转向 Secant-State Carrier；
- 不一次同时加反事实、新 modulation、新 loss 后只报一个总涨点。

### 现在冻结的路线

1. **代码主线：SRSC-Lite；**
2. **最佳论文候选：CG-SRSC，但仅通过分级 kill experiments 逐项升级；**
3. **最先测试的新增量：等距 under/over 反事实对；**
4. **其次测试：sign-equivariant state-to-action interface；**
5. **SEC 自探针：暂不进入主方法。**

> **客观终审：现在没有证据说 SEC 或新重跑候选比 SRSC 更好。有证据说，SRSC 的严格坐标是当前最可防守的核心，DOGC 经修正的等距反事实和动作闭环是最值得逐项实验吸收的升级。这是当前最接近顶会审稿范式、同时科学风险最可控的方案，但它的发表强度必须由 Stage-B/Stage-C 的真实对照结果决定。**

---

## 12. 证据文件

### Claude SEC-IR

```text
/root/autodl-tmp/ideaspark_run/allinone-restoration-state/phase4/idea.std.zh.md
/root/autodl-tmp/ideaspark_run/allinone-restoration-state/phase4/idea.detail.en.md
/root/autodl-tmp/ideaspark_run/allinone-restoration-state/phase4/phase4_implementability.json
/root/autodl-tmp/ideaspark_run/allinone-restoration-state/phase3_critique/phase3_critique_output.json
```

### Codex 独立 ResearchStudio 重跑

```text
/root/autodl-tmp/ideaspark_run/allinone-restoration-state-crossaudit-codex/phase1/phase1_output.json
/root/autodl-tmp/ideaspark_run/allinone-restoration-state-crossaudit-codex/attempt_1/phase3_critique/phase3_critique_output.json
/root/autodl-tmp/ideaspark_run/allinone-restoration-state-crossaudit-codex/phase2_coherence/phase2_coherence_output.json
/root/autodl-tmp/ideaspark_run/allinone-restoration-state-crossaudit-codex/phase3_critique/phase3_critique_output.json
/root/autodl-tmp/ideaspark_run/allinone-restoration-state-crossaudit-codex/phase_3_failed.md
```

### 现有主线与 DOGC

```text
/root/aaa/SRSC_Lite_v1.2_Codex_最终实施Prompt_v1.3.md
/root/aaa/ResearchStudio_DOGC_Codex_最终方案.md
```

---

## 13. 声明

> 本文件为 **Codex 在 Microsoft ResearchStudio Idea Spark 工作流、Claude SEC-IR 全部产物、当前 SRSC-Lite 实施规格和既有 DOGC 终审之上的交叉审核结果**。新的 ResearchStudio 重跑并未生成通过 Phase 4 的方法；CG-SRSC 是对所有审核后仍存活机制的 Codex 综合候选，必须按本文件的分级闸门实验，不得预先宣称已经有效或已达到顶刊顶会水准。

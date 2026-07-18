# Codex × Microsoft ResearchStudio 最终方案
## Directed Outcome Geometry Corrector（DOGC）及其与 SRSC-Lite v1.2 的终审结论

> **结果归属：这是 OpenAI Codex 生成、执行并终审的结果。**  
> **工作流：本次只使用 Microsoft ResearchStudio / ResearchStudio-Idea，不调用其他本地科研 skill。**  
> **日期：2026-07-18 UTC**  
> **状态：研究候选方案，不是已经由实验验证有效的最终方法。**

---

## 0. 一句话结论

ResearchStudio 给出的单一候选是：

> **DOGC：把第一遍复原的后果表示成目标相对的有向几何，并让第二遍显式选择 continue、rollback 或 redirect。**

ResearchStudio 的 idea-quality 盲评中：

- DOGC：83/100；
- 当前 SRSC-Lite v1.2：75/100；
- pairwise winner：DOGC。

但是，Codex 的最终客观判断是：

> **现在不应停止或推翻正在训练的 SRSC-Lite。DOGC 的二维坐标本质上与 SRSC 的核心坐标高度等价：`tau = alpha = 1-p`，`kappa` 近似 SRSC 的横向幅值 `m`。DOGC 真正新增且值得吸收的，不是另起炉灶的两通道状态，而是“等目标距离的欠修/过修/错向反事实监督”和“状态到动作的显式语义闭环”。**

因此本文件给出两层结论：

1. **ResearchStudio 原始候选：DOGC**；
2. **Codex 推荐实施路线：保持 SRSC 主骨架，先把 DOGC 的两项新增机制做成严格增量实验；只有显著胜过 SRSC，才升级为论文主方法。**

---

## 1. ResearchStudio 实际执行记录

本轮不是普通聊天式头脑风暴，而是完整执行了 IdeaSpark：

1. Phase 0：四连接器真实检索；
2. Phase 0+：全文抓取；
3. Phase 1：瓶颈与方法谱系诊断；
4. Phase 2：15 个主范式 × 31 个子范式的候选选择；
5. Phase 2.3：形式化数据流、数值干运行、退化输入和 claim-to-step 检查；
6. Phase 3.1：signature 近十个月 + alias 近四十八个月双通道撞车检索；
7. Phase 3.2：Reject lessons、recipe、anti-pattern、论文威胁和证伪结构审计；
8. Phase 3.3：独立修订；
9. Phase 4：中英文卡片、理论/工程可行性和实现性审计；
10. ResearchStudio idea-quality：DOGC 与 SRSC 同格式盲评。

关键审计事实：

- Phase 0 初始文献表：10 篇；
- 最终全文缓存：17 个候选项中 10 个成功抓取；
- 已抓到全文的关键锚点包括 RAR、PromptIR、AgenticIR、SIPL、Restormer；
- DA-RCOT 获得 TPAMI DOI 和元数据，但付费全文抓取失败；
- R2R 标题未解析到在线全文，因此不能冒充全文精读；
- Phase 3 撞车池：signature 28 篇 + alias 92 篇，共 120 篇；
- 最大论文威胁：RAR；
- RAR 未在检索证据中显示“等质量/等能量下区分欠修、过修和错向”的精确机制，因此不是 exact-mechanism collision；
- 最终验证：**5 pass、0 warn、0 fail**；
- 实现性审计：7 个步骤全部覆盖，0 个 unresolved open hole。

完整运行目录：

```text
/root/ResearchStudio/ideaspark_run/end-to-end-internal-state-aioir-v3/
```

---

## 2. 顶刊顶会式问题定位

### 2.1 不应该讲的故事

不要把故事写成：

- 再做一个退化分类器；
- 再预测一张 uncertainty map；
- 再增加一次 restoration stage；
- 再用一个 IQA 判断图像好不好；
- 做 unseen degradation generalization；
- 用 agent 调度一组小工具。

这些角度要么已经拥挤，要么无法回答为什么第二遍更新应当与第一遍不同。

### 2.2 应该讲的核心瓶颈

在已知退化集合的 All-in-One Image Restoration 中，未闭合的结构性问题发生在第一遍复原之后：

> 给定输入 `x` 和第一遍结果 `y1`，外观、PSNR、总体质量或残差能量相近的两个局部区域，可能分别是欠修、越过干净目标的过修，或沿错误方向发生的编辑。这三种状态需要相反的后续动作，但输入退化条件、当前图像特征和标量质量判断不能保证区分它们。

这个问题可称为：

```text
post-restoration transition non-identifiability
首遍复原后转移不可辨识
```

### 2.3 顶会式动机段

现有 AiOIR 方法主要回答“输入中存在什么退化”，而不是“模型刚才的复原操作造成了什么后果”。这种区别在单遍映射中并不显眼，却决定了第二次更新是否有意义：具有相同残差幅值或相似感知质量的首遍输出，可能仍然欠修、已经越过干净目标，或沿错误方向改变。因而，更深的第二阶段、重复应用同一复原器或总体质量评分都不能自动给出应继续、回退还是换向的可执行依据。我们据此把 AiOIR 的第二阶段从“再次复原”重构为“基于首遍操作后果的纠正”，并研究一个目标相对、有符号且可证伪的局部状态是否能消除这种转移不可辨识性。

---

## 3. 架构母型粗分类与最终选择

| 母型 | 典型形态 | 优点 | 对本问题的主要缺陷 | 结论 |
|---|---|---|---|---|
| 单干线条件化 | 单个 U-Net/Transformer，prompt/gate 调制 block | 最常见、便宜 | 没有先产生真实 `y1`，难以因果地评估首遍后果 | 不适合作为核心故事 |
| 单骨干中间头 + refinement tail | 一个 U-Net，在中间输出 `y1`，后续 block 继续精修 | 图看起来最常规 | 本质仍是两步；需证明后续 block 真使用后果状态 | 可作为工程重参数化 |
| 共享编码器 + D1/D2 | 一个共享 encoder、粗解码 D1、状态 assessor、纠正 D2 | 因果关系最清晰；只编码一次输入时可省算力 | 形式上比普通 U-Net 少见；必须做容量对照 | **当前最适合科学假设** |
| 两个完整网络级联 | Stage 1 network + Stage 2 network | 实现简单 | 参数、FLOPs 和容量混杂严重 | 不推荐 |
| 权重共享循环/展开 | 同一 restoration cell 迭代 K 次 | 迭代故事自然 | 状态与权重共享纠缠；训练稳定性和归因更难 | 可做外部对照，不作首选 |
| 状态空间/闭环控制 | hidden state + observer + controller | 故事新颖、理论感强 | 容易成为控制术语包装；实现和证明负担大 | 不在首篇主线采用 |

ResearchStudio 最终仍选择固定 `K=2` 的共享骨干两阶段架构。这并不是因为“两解码器很常见”，而是因为：

> 要评价第一次编辑的真实后果，计算图中必须先存在 `y1`。如果没有先得到 `y1`，所谓 feedback 只是输入条件化，不是 post-restoration feedback。

因此，当前一个 encoder + 两个 decoder 的拓扑不是天然缺点。审稿人真正会攻击的是：

- 是否只是多了容量；
- assessor 是否只是另一份隐藏特征；
- 第二遍是否真的使用了状态；
- GT-derived 状态是否能在推理时学出来；
- 同维 residual code 是否同样有效。

---

## 4. ResearchStudio 原始候选：DOGC

### 4.1 总体计算图

```text
                    ┌──────────────────────┐
x ────────────────► │ shared encoder E     │ ─────► multi-scale F_x
                    └──────────┬───────────┘
                               ▼
                         coarse decoder D1
                               ▼
                              y1

(x, y1, y1-x, F_x, F_y1)
              └──────────────► assessor A ─────► T_hat=[tau_hat,kappa_hat]
                                                   │
                                                   ▼
                                            semantic gate G
                                      continue / rollback / redirect
                                                   │
(F_x, F_y1, y1) ─────► shared correction trunk D2 ├─► R_continue
                                                   ├─► R_rollback
                                                   └─► R_redirect
                                                          │
                                                          ▼
                  y2 = y1 + sum_m softmax(G(T_hat))_m * R_m
```

严格按 ResearchStudio 原卡，`E` 是共享权重 encoder，但推理时分别处理 `x` 与 `y1`，因此是：

```text
one shared encoder module, two encoder executions
```

这与当前 SRSC Prompt 中“encoder 只执行一次、y1 使用 shallow pyramid”的工程取舍不同。若以当前 SRSC 代码为母体，不应静默改变这一点。

### 4.2 训练期与推理期边界

训练期可用：

```text
x, y1, paired clean target y*, constructed under/over/wrong states
```

推理期只允许：

```text
x, y1, y1-x, shared features
```

推理期禁止：

```text
y*, q, constructed states, task label, degradation label, ratio,
VLM/LLM, agent, external tool, external visual prior
```

---

## 5. 核心数学对象：Directed Outcome Geometry

对局部 `3×3 RGB` patch 使用反射填充和展平算子 `P_p`：

```math
g(p)=P_p(y^*-x),
u_z(p)=P_p(z-x).
```

有效位置：

```math
M(p)=1[||g(p)||_2 >= 3 epsilon].
```

定义：

```math
tau_z^*(p)=<u_z(p),g(p)> / ||g(p)||_2^2,
```

```math
kappa_z^*(p)=||u_z(p)-tau_z^*(p)g(p)||_2 / ||g(p)||_2.
```

最终二维状态：

```math
T_z^*(p)=[tau_z^*(p), kappa_z^*(p)].
```

解释：

- `tau < 1`：沿目标方向仍然欠修；
- `tau > 1`：沿目标方向越过目标，发生过修；
- `kappa > 0`：更新偏离目标方向；
- `[tau,kappa]` 只在训练期由 GT 构造，测试时由 assessor 预测。

与 SRSC 的关系必须如实写清：

```math
SRSC: alpha=<v1,v*>/(||v*||^2+eps),  p=1-alpha
DOGC: tau=<u,g>/||g||^2
```

在忽略数值稳定项和描述子差异时：

```math
tau ≈ alpha = 1-p,
kappa ≈ SRSC 的 normalized transverse magnitude m.
```

所以 DOGC 的二维状态本身不能在内部新颖性上完全击败 SRSC。

---

## 6. 三个候选创新点

### 创新点 1：目标相对的有向后果几何

将第一遍复原从“当前图像质量”改写为“实际更新相对目标更新的投影与偏离”：

```text
T=[signed target progress, normalized transverse deviation]
```

理论表述可限定为：对固定目标方向 `g`，`[tau,kappa]` 是保持 `g` 不变的局部正交变换群轨道的不变量。不要扩大为“完整恢复真实物理退化”或“最优纠正的充分统计量”。

### 创新点 2：等目标距离的反事实后果监督

构造：

```math
z_under = x + 0.5(y^*-x),
z_over  = x + 1.5(y^*-x),
z_wrong = y^* + 0.5q,
```

其中：

```math
q perpendicular to (y^*-x),
||q||_2 = ||y^*-x||_2.
```

三者到 `y*` 的距离相同，却分别需要：

```text
continue / rollback / redirect
```

这一步是 ResearchStudio 相对 SRSC 最有价值的新增内容。它直接排除了“assessor 只是读取残差幅值”的捷径，并给出了审稿人容易理解的强可视化和强负控制。

### 创新点 3：从状态坐标到纠正动作的显式语义闭环

将 D2 设计为共享纠正干线和三个残差动作头：

```text
R_continue, R_rollback, R_redirect
```

由 `T_hat` 产生逐像素 softmax 权重：

```math
w(p)=softmax(G(T_hat(p))),
```

```math
y2=y1+sum_m w_m(p) R_m.
```

这一点补足当前 SRSC 的主要故事弱点：SRSC 通过 affine modulation 隐式使用状态，而 DOGC 明确说明每一类状态为什么改变最终动作。

但是三头并非必须无条件进入主模型。若三头相对单头 D2 没有稳定增益，它应降级为训练期辅助 action-classification head，而不是强行保留为主架构。

---

## 7. 损失函数

建议的结构为：

```math
L = L_rest
  + lambda_route L_route
  + lambda_coord L_coord
  + lambda_pair L_pair
  + lambda_order L_order
  + lambda_sep L_sep.
```

含义：

- `L_rest`：最终 `y2` 的标准 restoration loss；
- `L_coord`：预测 `T_hat` 与解析目标 `T*` 的 Smooth-L1；
- `L_pair`：同一状态在颜色排列和空间翻转下的坐标一致性；
- `L_order`：要求 `tau_under < tau_wrong < tau_over`；
- `L_sep`：要求 `kappa_wrong` 显著大于 under/over；
- `L_route`：构造状态分别监督 continue/rollback/redirect。

禁止为 DOGC 单独引入、而公平对照没有的 perceptual/GAN/额外数据损失。

---

## 8. 与当前 SRSC-Lite v1.2 的客观比较

| 维度 | SRSC-Lite v1.2 | DOGC | 客观判断 |
|---|---|---|---|
| 问题定位 | post-restoration state unknown | post-restoration transition non-identifiability | 本质相同 |
| 进展坐标 | `p=1-alpha` | `tau=alpha` | 一一变换，非独立新意 |
| 横向幅值 | `m` | `kappa` | 高度等价 |
| 横向方向 | 6D `d` | 不保留方向，只保留幅值 | SRSC 信息更丰富 |
| 状态到 D2 | zero-init affine modulation | 三动作头显式路由 | DOGC 故事闭环更强 |
| 监督构造 | 真实 y1 的解析坐标 | 真实 y1 + 等距 under/over/wrong | DOGC 更能排除幅值捷径 |
| 数据要求 | 标准 paired x/GT | 原卡要求同 clean 的多退化 | SRSC 更兼容标准协议 |
| 推理计算 | encoder 一次 + y1 shallow pyramid | 共享 E 对 x/y1 各运行一次 | SRSC 更省 |
| 容量混杂 | 已有严格等容量接口 | 三头 D2 增加归因难度 | SRSC 更容易公平审计 |
| Idea-quality | 75/100 | 83/100 | DOGC 写法和动作闭环更完整，不等于已实证更强 |

### Codex 最终判断

DOGC 不是“完全不同、必然更好的新架构”。它更像：

```text
SRSC 的二维简化坐标
+ 等距反事实监督
+ 显式动作语义
```

因此直接放弃 SRSC、重新训练完整 DOGC 的风险很高。更稳妥且更符合论文归因的方法是：

```text
SRSC-Lite
→ 加入 matched-distance under/over/wrong supervision
→ 先加入轻量 route auxiliary head
→ 只有 route head 对最终 PSNR/SSIM 有独立贡献时，再升级为三残差头
```

---

## 9. 推荐的最终候选：SRSC-DOG Hybrid

这不是 ResearchStudio 原始卡片的静默改名，而是 Codex 基于其结果做出的实施建议。

### 保留

- 当前共享 encoder + D1 + assessor + D2；
- 当前 8 通道 `[p,m,d1...d6]`；
- 当前 8 通道公平接口；
- 当前 oracle/predicted ladder；
- 当前同维 residual-code 对照；
- 当前 encoder 只运行一次、y1 shallow pyramid；
- 当前正在训练的 Stage-A checkpoint 和协议。

### 新增候选机制 A

加入 matched-distance counterfactual training：

```text
under / over / wrong-direction
```

用它约束：

- `p` 的有符号顺序；
- `m` 对错向状态的分离；
- `d` 对 transverse orientation 的区分。

### 新增候选机制 B

在 assessor 后增加一个非常轻量的三类 action auxiliary head：

```text
continue / rollback / redirect
```

先只作为训练期语义约束，不立刻改 D2 为三个大残差头。

### 升级条件

只有在以下均成立时，才把 D2 改为三动作残差头：

1. SRSC + counterfactual > 原 SRSC；
2. SRSC + counterfactual + route auxiliary > 仅 counterfactual；
3. 三残差头 > 相同参数量的单残差头；
4. 打乱状态后最终 PSNR/SSIM 增益明显回落；
5. predicted state 而不是只有 oracle state 有效。

---

## 10. 最小 kill experiment

### 10.1 不需要重训完整主干的第一轮

固定同一个 `E+D1` checkpoint 和同一个 `y1` cache，仅训练容量匹配的 D2/assessor 变体：

```text
B0: two-stage no-state
B1: uncertainty / error magnitude
B2: projected residual code
B3: current SRSC
B4: DOGC two-channel T
B5: SRSC + matched-distance supervision
B6: SRSC + matched-distance + route auxiliary
```

### 10.2 唯一 load-bearing variable

```text
predicted restoration-outcome state
```

### 10.3 必须有的负控制

主负控制：

```text
从训练开始跨样本打乱 outcome-state target，重新训练相同模型。
```

补充诊断：

```text
冻结模型，在测试时跨样本/空间共同打乱预测 state。
```

仅测试时打乱可能产生 OOD 输入，因此不能单独作为机制证明。

### 10.4 科学判定

若出现以下任一结果，DOGC 不应替代 SRSC：

- DOGC ≤ SRSC；
- SRSC + counterfactual ≤ SRSC；
- route auxiliary 不改变最终结果；
- 三残差头不优于参数匹配单头；
- residual code 与新状态持平或更优；
- 只有 oracle 有效，predicted state 无效；
- 增益只来自 dehaze 或单一 setting。

---

## 11. 主要风险

### 风险 1：与 SRSC 内部新颖性重叠

`tau/kappa` 与 `p/m` 高度等价。论文不能把二维坐标重新命名后当成第三个独立贡献。

### 风险 2：wrong-direction 构造可能不贴近真实 y1 错误

正交合成方向可以用于排除幅值捷径，但未必覆盖真实网络最常出现的 transverse error。必须报告从 synthetic matched set 到真实 y1 的迁移。

### 风险 3：二维状态不包含 redirect 方向

`kappa` 只告诉模型“偏了多少”，不告诉“向哪里纠正”。DOGC 依赖 D2 从图像特征中补足方向；SRSC 的 6D `d` 在这点上理论上更强。

### 风险 4：标准 AiOIR 数据未必具有同 clean、多退化配对

PromptIR 风格 AIO-3/AIO-5 的不同任务常来自不同数据集，不能假设同一 clean image 同时拥有 rain/haze/noise 对。严格 ResearchStudio 原卡的数据要求可能破坏标准协议。

### 风险 5：三动作头带来容量与路由塌缩

三个头可能只增加容量，或 softmax 长期选择同一头。必须报告路由熵、各头使用率和容量匹配对照。

### 风险 6：理论命题不能夸大

`[tau,kappa]` 可作为固定 `g` 下某类正交群轨道的不变量，但这不自动证明它是最优纠正的充分统计量，也不证明恢复了真实物理退化。

---

## 12. ResearchStudio 自动算力判定的已知错误

ResearchStudio skeleton 自动写出了：

```text
GPU line ≈ 4090 vs 4 GPU-day → infeasible
```

这是数字解析 bug：

- 它把显卡型号 `4090` 当成 GPU-day；
- 它把 `4×RTX4090` 的卡数 4 当成总 GPU-day 预算。

候选真实估算是：

```text
约 32 个 RTX 4090 GPU-day
= 4 张 RTX 4090 上约 8 个墙钟日
```

所以原始卡片中的 `infeasible` 不应作为科学结论。它是解析错误，不是模型不可实现。

---

## 13. 最终发表层面的客观结论

### 可以声称

- ResearchStudio 发现了一个比“质量评估反馈”更尖锐的问题表述：首遍转移不可辨识；
- 120 篇双通道撞车审计没有发现与等距方向监督 + 投影/偏离状态 + 显式动作路由完全相同的机制；
- RAR 是最大相邻威胁，但现有检索证据未显示其具有相同的方向几何；
- DOGC 在 idea-quality 上强于简写版 SRSC，主要因为动作闭环更完整；
- counterfactual matched-distance supervision 是值得立即验证的新增机制。

### 不能声称

- DOGC 已经优于 SRSC；
- DOGC 已经优于 PromptIR、R2R、RAR 或其他 SOTA；
- 83/100 意味着可录用；
- `tau/kappa` 相对 `p/m` 本身构成全新表示；
- 三动作头一定涨点；
- 当前训练结果已经支持 DOGC。

### 最终推荐

```text
继续当前 SRSC-Lite 主线训练，不中断。
先做 SRSC + matched-distance counterfactual supervision 的最小增量实验。
再做 action auxiliary head。
只有两级机制闸门都通过，才训练完整 DOGC 三头模型。
```

这是当前风险最低、归因最清楚，也最符合顶刊顶会审稿范式的路线。

---

## 14. ResearchStudio 原始交付文件

```text
/root/ResearchStudio/ideaspark_run/end-to-end-internal-state-aioir-v3/phase4/idea.std.zh.md
/root/ResearchStudio/ideaspark_run/end-to-end-internal-state-aioir-v3/phase4/idea.std.zh.pdf
/root/ResearchStudio/ideaspark_run/end-to-end-internal-state-aioir-v3/phase4/idea.std.en.md
/root/ResearchStudio/ideaspark_run/end-to-end-internal-state-aioir-v3/phase4/idea.std.en.pdf
/root/ResearchStudio/ideaspark_run/end-to-end-internal-state-aioir-v3/phase4/idea.detail.en.md
/root/ResearchStudio/ideaspark_run/end-to-end-internal-state-aioir-v3/comparison/idea_quality_pairwise.md
```

---

## 15. 文件声明

> **本 Markdown 是 Codex 对 Microsoft ResearchStudio-Idea 完整运行结果的整理与终审文件。**  
> **DOGC 是 ResearchStudio 生成的候选；“不立即替换 SRSC、优先做 SRSC-DOG Hybrid kill experiment”是 Codex 的最终客观建议。**  
> **方法有效性尚未由真实训练结果建立。**


# AutoSOTA Strategy Library for SRSC-Lite

Status: **SETUP ONLY — NOT AUTHORIZED TO OPTIMIZE**.

Activation requires formal Oracle and Predicted GO, a reproducible positive Stage-C P7 trend, a frozen locked-validation baseline/checkpoint SHA256, a protected evaluation wrapper, and usable API credentials. Official-test results are never optimization feedback. The examples below are abstracted from the local AutoSOTA README/CLI documentation; they are training-engineering candidates, not SRSC paper contributions.

## 1. Locked-validation checkpoint selection

- AutoSOTA案例：NsDiff固定/备选epoch揭示小验证集噪声；ACIA与LatentScoreReweight强调可靠的best checkpoint指标。
- 为什么可能涨点：避免最后epoch或单一高方差batch覆盖真正泛化最好的状态。
- 适合模型/数据：长程Restormer训练、跨任务macro波动明显的AIO。
- 风险：反复试选择规则会形成验证集过拟合。
- 论文创新：否；仅训练增强。
- 公平对比：规则在看SRSC结果前冻结，所有内部baseline使用同一locked-val macro与top-3逻辑。

## 2. Learning-rate/scheduler alignment

- AutoSOTA案例：TimePFN、PIR的衰减修正，AMDM让scheduler在有限预算内真实生效。
- 为什么可能涨点：反馈分支从随机初始化训练，错误warm-up或过快衰减会造成欠拟合。
- 适合模型/数据：Stage-B/Stage-C短周期精修。
- 风险：仅给P7调LR会破坏公平性。
- 论文创新：否；仅训练增强。
- 公平对比：P6/P7/P12/O0及负控制统一scheduler、总step和选择规则。

## 3. Batch size / crop size

- AutoSOTA案例：K2VAE以梯度累积稳定更新；若干视觉任务通过合适分辨率改善指标。
- 为什么可能涨点：有效batch控制跨任务梯度方差，大crop提供更完整的雨纹/雾结构。
- 适合模型/数据：AIO预训练crop128、联合精修crop224。
- 风险：改变有效batch需同步LR；只给方法大crop属于额外预算。
- 论文创新：否；仅训练增强。
- 公平对比：固定effective batch，OOM只减micro-batch并等量增加accumulation。

## 4. Longer training / patience

- AutoSOTA案例：DementiaMask、IAGGAD、CMNN、STELLA通过更长收敛或patience获益。
- 为什么可能涨点：第二阶段D2/assessor可能比coarse路径收敛慢。
- 适合模型/数据：已有明确正趋势但尚未平台期的Stage-B/C。
- 风险：巨大GPU成本；不能把“训练更久”包装成创新。
- 论文创新：否；仅训练增强。
- 公平对比：所有核心反馈臂等epoch/step，并完整报告GPU-hours。

## 5. Model soup / checkpoint averaging

- AutoSOTA案例：MeanFlows的EMA/live blend、GCSE/CSBrain多checkpoint融合、ACIA Top-5 averaging。
- 为什么可能涨点：降低单checkpoint方差并平滑不同盆地。
- 适合模型/数据：仅在独立附加效率/上限实验中。
- 风险：违反本项目“一个checkpoint、一个推理协议”的主表要求，易掩盖任务冲突。
- 论文创新：否；默认禁止主结果。
- 公平对比：若报告，所有baseline同样soup且单独成表；publication主结论仍用单checkpoint。

## 6. TTA / self-ensemble

- AutoSOTA案例：TinySAM、MindGlitch、InfoSAM使用flip/TTA。
- 为什么可能涨点：对称变换平均可压低预测方差。
- 适合模型/数据：仅补充上限分析。
- 风险：复原延迟成倍增加；非对称退化可能被flip破坏；主prompt明确禁止。
- 论文创新：否；默认禁止主结果。
- 公平对比：若附录报告，所有方法同样TTA并报告延迟，不能用于满足+0.30dB门槛。

## 7. Loss alignment

- AutoSOTA案例：CALF移除伤害主头的consistency，xPatch用与评测更一致的MSE，NeuralMJD调整均值目标权重。
- 为什么可能涨点：state监督与最终L1梯度尺度不匹配时会争夺D2容量。
- 适合模型/数据：pilot中出现state loss压制restoration loss且P7已有正趋势。
- 风险：按表示单独调权重会制造不公平。
- 论文创新：否；除非形成独立理论机制，本项目仅训练增强。
- 公平对比：只允许对全部表示统一缩放lambda_state并记录梯度范数。

## 8. Validation metric selection

- AutoSOTA案例：LatentScoreReweight按worst-group而非平均选点；NsDiff暴露val/test噪声差异。
- 为什么可能涨点：AIO平均值会掩盖denoise或derain负迁移。
- 适合模型/数据：多任务locked validation。
- 风险：根据方法结果后改metric会泄漏。
- 论文创新：否；评估纪律。
- 公平对比：主选择固定five-setting/task macro，同时保存per-task、worst-task和bootstrap但不事后换主指标。

## 9. Data sampling / task balancing

- AutoSOTA案例：TropicalAttention混合长序列辅助batch；多任务案例通过重平衡暴露稀有场景。
- 为什么可能涨点：OTS样本量远大于rain/denoise，默认采样可能使dehaze主导。
- 适合模型/数据：若真实梯度/采样审计证明任务失衡，并且所有方法统一使用。
- 风险：偏离R2R标准协议，影响外部表可比性。
- 论文创新：否；仅附加公平增强。
- 公平对比：标准主表不改采样；任何balanced版本单列并对baseline同步训练。

## 10. Task-specific batch repeat

- AutoSOTA案例：等价于多任务优化中的稀有域重复与辅助batch注入。
- 为什么可能涨点：增加小任务被optimizer看到的频率。
- 适合模型/数据：Rain/LOL/GoPro相对OTS严重欠采样时。
- 风险：真实task label进入采样器虽不进入推理，但改变协议且可能过拟合。
- 论文创新：否；仅训练增强。
- 公平对比：不用于当前标准R2R主线；若试验，所有内部baseline使用同一repeat表。

## 11. Augmentation

- AutoSOTA案例：MoSES/TropicalAttention通过对称或长度增强改善泛化。
- 为什么可能涨点：几何增强可扩大结构分布，局部复合增强可改善OOD。
- 适合模型/数据：不改变GT语义的flip/rotation，以及预注册的复合退化训练消融。
- 风险：雨方向、模糊核或相机噪声不一定对称；新增退化会改变标准数据协议。
- 论文创新：否；仅训练增强。
- 公平对比：标准主表沿用baseline增强；新增增强必须对P7和baselines同步并单列。

## 12. EMA

- AutoSOTA案例：MeanFlows EMA blend、FedGMT轨迹平滑/late SWA。
- 为什么可能涨点：平滑高方差多任务更新，常改善验证稳定性。
- 适合模型/数据：Stage-C出现明显checkpoint抖动时。
- 风险：EMA是第二套权重；若只给P7使用不公平。
- 论文创新：否；仅训练增强。
- 公平对比：统一decay/start、统一checkpoint语义，对全部baseline重训；单checkpointEMA可作为最终权重但必须如实命名。

## 13. Gradient clipping

- AutoSOTA案例：FastFeatureCP通过分位数clip抑制异常梯度；当前R2R固定global norm 0.01。
- 为什么可能涨点：混合任务与state cosine可能产生尖峰。
- 适合模型/数据：日志证明NaN或极端gradient norm时。
- 风险：过强clip导致D2欠拟合；改动会偏离R2R。
- 论文创新：否；仅稳定性增强。
- 公平对比：主线固定0.01；若比较新阈值，对所有臂统一并报告gradient统计。

## 14. Mixed-precision stability

- AutoSOTA案例：FlashTP编译/精度路径与K2VAE评测稳定性说明实现精度会影响结果。
- 为什么可能涨点：BF16避免FP16溢出并提高吞吐，使标准预算可执行。
- 适合模型/数据：24GB单卡Restormer训练。
- 风险：外部R2R默认实现可能是FP32，不能声称逐位复现。
- 论文创新：否；工程设置。
- 公平对比：所有内部方法统一BF16；记录AMP、CUDA、峰值显存和外部协议差异。

## 15. Seed ensemble / seed robustness

- AutoSOTA案例：CausalPFN、CSBrain、M3SVM通过多seed/ensemble降方差。
- 为什么可能涨点：独立初始化可降低偶然失败。
- 适合模型/数据：机制稳定性分析。
- 风险：挑最好seed或ensemble违反单checkpoint主结果。
- 论文创新：否；统计证据。
- 公平对比：预注册三seed、报告均值/方向一致性；主表不挑seed、不ensemble。

## 16. Calibration

- AutoSOTA案例：SAVVY尺度校准、SuitabilityFilter isotonic、InfoSAM logit scaling。
- 为什么可能涨点：输出范围或confidence偏差可被校准。
- 适合模型/数据：仅诊断，不适合作为SRSC主线。
- 风险：per-task gain/bias、alpha扫描和test calibration属于旧路线且不可发表。
- 论文创新：否；本项目主表禁止。
- 公平对比：不得用official test调校准；若附录报告只能用train/locked-val统一规则并覆盖baseline。

## 17. Distillation

- AutoSOTA案例：多模型/物理融合案例提示强teacher可稳定学生，但不等同于本项目贡献。
- 为什么可能涨点：Oracle SRSC可作为state teacher帮助predicted assessor逼近信息上限。
- 适合模型/数据：Oracle GO但capture ratio低、P7仍有正趋势时。
- 风险：新增teacher训练预算和目标，可能改变论文主创新或引入GT泄漏。
- 论文创新：可能形成后续工作；当前只允许训练期teacher且推理图无GT。
- 公平对比：P6/P7/P12使用等容量teacher/预算；单独消融并验证无推理泄漏。

## 18. Hard-example mining

- AutoSOTA案例：多任务/异常检测案例通过聚焦困难样本或尾部改善鲁棒指标。
- 为什么可能涨点：可针对过恢复、低q/高m或worst-10%图像加强学习。
- 适合模型/数据：per-image paired分析显示稳定尾部失败且平均已有正趋势。
- 风险：用locked-val选择训练样本是泄漏；可能偏向某任务。
- 论文创新：否；仅训练增强。
- 公平对比：困难度只能从train在线统计，采样器对所有方法一致，并报告任务组成变化。

## 19. Negative-transfer penalty

- AutoSOTA案例：CALF移除有害辅助约束、RSTIB去掉过强压缩项，说明辅助目标可能造成负迁移。
- 为什么可能涨点：约束不同任务梯度冲突或clean-region无谓编辑。
- 适合模型/数据：P7平均正但某些任务系统性下降时。
- 风险：若机制只是新loss权重，不具论文创新性；真实task label不能进入推理。
- 论文创新：否；当前`Lclean`已是统一守卫。
- 公平对比：惩罚对所有反馈变体统一；不得按任务手调，报告每任务梯度/性能。

## 20. OOD / robustness validation

- AutoSOTA案例：XMahalanobis、TropicalAttention等强调OOD分布与主指标同时验证。
- 为什么可能涨点：它不直接提高主分，而是防止AutoSOTA只适配locked-val。
- 适合模型/数据：paired local-composite rain+local-noise、后续真实混合退化。
- 风险：把OOD集用于每轮选择会变成第二验证集并过拟合。
- 论文创新：否；必要证据。
- 公平对比：配置冻结后一次性评估；不纳入标准AIO平均，P7/O0/matched baseline同协议paired比较。

## Activation decision

- CLI: installed `0.2.0`; `autosota doctor` runtime dependencies pass.
- Workspace: `/root/autodl-tmp/autosota_runs/srsc_lite_v12` exists and its `paper/target.md` has the correct scientific boundary.
- Missing authority: all API credential fields are empty; Stage-C baseline is still `TBD`; protected eval wrapper/checkpoint SHA256 do not yet exist.
- Decision: **do not run AutoSOTA now**. Populate these only after formal positive SRSC evidence; otherwise record NO-GO and do not optimize a rejected architecture.

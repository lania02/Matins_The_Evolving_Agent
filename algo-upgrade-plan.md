# Matins 算法升级计划

> 配套文档:`algo-update.md`(问题诊断 + 改进方法 + 工程可实施性)。
> 本文件把那份抽象批判落到真实代码上,给出可执行、可验证、按依赖排序的升级方案。
> **状态标注**:Phase 1–4 = 本次改动已实现;Phase 5 + 研究支线 = 暂缓(机制可早搭,信任受数据门控)。

---

## 0. 一个改变全局的代码观察:θ 与 φ 在本系统里是**融合**的

`algo-update.md` 用线性模型 `U = θ·φ(x)` 讲问题,但代码里**没有数值化的 θ 向量,也没有独立的 φ 基**:

- **φ + θ 合二为一** = 一份自然语言的 taste skill(`skills/taste.md` + `skill_versions` 表;当前是 cold-start 占位)。
- **"重加权 θ"(self-training)与"进化 φ"(self-evolution)走同一条代码路径** —— 都通过 `prompts/propose_skill_diff.txt` 把整份 skill 重写一遍(`memory/kernels.py::compute_memory` → `memory/consolidate.py`)。
- 当前那个 prompt 被明确指示**保守、保留旧措辞、偏好结构**(`propose_skill_diff.txt`)—— 这正是把它锁死在 self-training 的那道闸:它能调措辞(≈重加权),但没被要求**铸造 rubric 里还没有名字的新维度**。

**推论**:`algo-update.md` 里最难的第 6 条(进化 φ),在本代码里**不需要新基建**,只需改写 `propose_skill_diff.txt` 的指令 + 加一个回测闸门。整个升级因此是"在现有底座上加固化逻辑",不是另起炉灶。

τ 澄清:`feedback/diverge.py` 里 τ **从来不是被最小化的 loss**(DESIGN §15 已写明它是 diagnostic)。第 2 条 Goodhart 的真正风险在 `propose_skill_diff` 隐性奖励"和用户一致";修复点是 prompt 层 + 把残差显式拎出来。

---

## 升级阶段总览

| Phase | 对应 `algo-update.md` | 一句话 | 成本 | 状态 |
|------|----------------------|--------|------|------|
| 1 | #1 D-工具变量 + #2 奖励正向惊喜 | 把残差变成一等公民 | 最低(纯 prompt+记账) | ✅ 本次 |
| 2 | #3 评论多通道 | 评论按种类路由到不同记忆 | 中 | ✅ 本次 |
| 3 | #4 自适应探索 | 温度随 τ 波动自调 | 中 | ✅ 本次 |
| 4 | #5 Quality-Diversity 存档 | 找回被放弃的好方向(派生视图) | 中 | ✅ 本次 |
| 5 | #6 进化 φ | self-training → self-evolution(RLVR 接地的回测闸门) | 中偏难 | ✅ 本次(默认 OFF) |
| 研究支线 | 结尾 + #7 | 显式估计反身转移 h(Sₜ) | 研究级 | ⏸ 暂缓 |

依赖关系:
```
Phase 1 (残差基建) ──┬──> Phase 5 (进化 φ)   ← 必须先有残差
                     └──> 研究支线 (h(Sₜ))
Phase 2 (多通道)  ─────> Phase 5 (喂给新维提议)
Phase 3 (自适应探索)  独立
Phase 4 (QD 存档)     独立(依赖已存在的 genes 概念)
```

---

## Phase 1 — 把残差变成一等公民(#1 + #2)

**意图**:`#1` 把随机槽 D 当未被 exploit 污染的工具变量(最干净的品味读数);`#2` 反转损失——奖励"用户排高、系统排低"的正向惊喜(发现新维度的信号),而不是一味追求和用户排序一致。两者共用同一处:事件渲染。

**改动**(全部落在事件渲染 + 两个 prompt,零 schema、零新代码路径):
- `matins/memory/kernels.py::format_events`:
  - `slot == "random"` 的事件加 `[D: clean probe]` 标记。
  - `self_rank − user_rank ≥ 2`(系统低估)的事件加 `[+underrated by N]` 标记。
- `prompts/summarize_recent.txt` + `prompts/propose_skill_diff.txt`:各加两句——"优先信 D 槽的反应,它未被我们选了什么所污染";"盯住我**低估**的 idea 去找缺失的维度,不要把'和用户一致'当目标"。

两个 prompt 都消费 `format_events` 的输出,所以标记同时进入 fast memory 与 slow consolidation。

**成功判据**:`tests/test_kernels.py`(纯函数,无网络)——随机槽事件带 D 标记;`self_rank=4,user_rank=1` 的事件带 underrated 标记。

---

## Phase 2 — 评论拆成多通道证据(#3)

**意图**:`feedback.user_comment` 现在是一个自由文本字段,反射时全糊在一起(信用分配歧义)。把每条评论分类为 `{taste, novelty, feasibility, structure}` 并路由:已经有人做过→novelty,话题无聊→taste,做不动→feasibility,框架问题→structure。

**改动**(共用一个幂等列迁移 helper,见下):
- `matins/store/db.py`:`feedback` 加列 `comment_kind TEXT`;新增 `_ensure_column`(`PRAGMA table_info` 检查后 `ALTER TABLE ADD COLUMN`,对现有生产库安全)。`recent_events` SELECT 带上 `comment_kind`。
- `matins/store/models.py`:`Feedback.comment_kind: str = ""`。
- `matins/feedback/capture.py`:`classify_comment(llm, comment) -> kind`(容错,默认 `taste`);`_ingest_text` / `ingest_replies` 接受可选注入式 `classify` 回调(capture 仍不直接 import LLM provider,保持依赖轻、离线测试不变)。
- `matins/cli.py`:`collect` / `feedback` 注入分类回调(advisory,失败默认 `taste`)。
- `matins/memory/kernels.py::format_events`:评论渲染为 `comment[kind]=...`。
- `prompts/propose_skill_diff.txt`:按通道路由的一句指令。

**成功判据**:`tests/test_feedback.py` 扩展——`classify_comment` 容错解析;带 kind 的评论在 `format_events` 中显示通道标签。

---

## Phase 3 — 自适应探索(#4)

**意图**:`generation.temperature` 现在是常数 0.4。让探索强度随不确定性/漂移变化:不稳(τ 波动大)→多探索,稳→多 exploit。`batches.self_user_tau` 已逐批入库,代理量现成。

**Occam 取舍**:只用**最干净的单一代理量——近期 τ 的波动率**。`algo-update.md` 还列了 fast/slow 分歧、skill 陈旧度两个代理量,但它们数据路径更重、收益边际,**不投机实现**(留作 Phase 3+)。

**改动**:
- 新增 `matins/generate/explore.py`:纯函数 `adaptive_temperature(recent_taus, base)` —— τ 少于 2 个(冷启动)返回 base;否则 `temp = clamp(base + gain·(volatility − neutral), 0.1, 0.9)`,volatility = 近期 τ 的总体标准差。常数为启发式、可调。
- `matins/generate/pipeline.py::run_batch`:用 `store.list_batches(limit=8)` 取近期 τ(复用,不加 store 方法),算出 `adaptive_temp`,写入 `batch.temperature` 并作为 `slot_temperature` 的 base。

**成功判据**:`tests/test_explore.py`(纯函数)——高波动 τ → temp > base(探索);近常数 τ → temp < base(exploit);冷启动返回 base;钳制在 [0.1, 0.9]。

---

## Phase 4 — Quality-Diversity 存档(#5)

**意图**:现在只有标题级反重复(`recent_idea_titles`)与检索去重。QD 的独特价值是**行为层面的多样性**:保留"曾被你高评、但近期已沉寂"的方向,在品味回摆时找回——直接对抗 `#1` 的反身塌缩/窄化。

**Occam + 架构取舍**:遵守 DESIGN §3"日志是资产,记忆是派生视图,绝不单独维护状态"——**不加 archive 表**。存档 = 对 `ideas`+`feedback` 的派生查询。需要一个覆盖所有 idea 的行为坐标:给 idea schema 加**一个**模型自发的 `behavior` 字段("领域·方法"短标签),schema 集中定义,故不碰 4 个 slot 模板。

**改动**:
- `matins/generate/schema.py`:`IDEA_FIELDS` 加 `behavior`(可选,非 required;模型漏填则 `normalize_idea` 默认 `""`,容错不挂)。
- `matins/generate/slots.py::_idea_schema_instruction`:加一句解释 behavior = 2–4 词的"领域·方法"归档标签。
- `matins/store/models.py`:`Idea.behavior: str = ""`;`db.py`:`ideas` 加列 `behavior`(复用 `_ensure_column` 迁移)+ `insert_idea` 带上。
- `matins/store/db.py`:派生查询 `archive_revival(recent_days, dormant_days, limit)` —— 取每个 behavior 格里用户评分最好、但最近 `recent_days` 内未再出现的"沉寂精英",按格去重。
- `matins/generate/pipeline.py`:存 `behavior`;取 revival 候选注入 context。
- `matins/generate/slots.py` + `prompts/slot_adjacent.txt`:新增 `{{ARCHIVE}}` token,**仅注入 adjacent 槽**(受控探索,语义最契合"找回被放弃的好方向";highfit 保持纯 exploit、random 保持纯扰动)。

**成功判据**:`tests/test_db.py` 扩展——`archive_revival` 只挑沉寂且高评的 idea、按 behavior 格去重、近期活跃的 behavior 被排除。

---

## 共享基建:幂等列迁移

Phase 2(`feedback.comment_kind`)与 Phase 4(`ideas.behavior`)都需对现有生产库 `ALTER TABLE ADD COLUMN`。共用 `Store._ensure_column(table, column, decl)`:`PRAGMA table_info` 查到缺列才加。新库经 `CREATE TABLE` 已含该列→跳过;旧库→补列(默认空串)。一处 helper,两处复用,不重复。

---

## Phase 5 — 进化 φ:RLVR 接地的自演化(✅ 本次实现,默认 OFF)

**底线(讨论确认)**:用 **RLVR 的精神(可验证奖励),不用 RLVR 的机器(梯度训练)**。"policy" = 自然语言 skill;"update" = 人审 consolidation;**verifier = 留出 τ**(锚在样本外人类排序,提案者看不到)。打分器靠被回测 gated 的 φ 演化"动态变强",不是 self-reward 闭环;generator 保留变异槽,不全面对 judge 优化(反 Goodhart)。

**已实现组件**:
- 打分器(§2.0):`prompts/predict_rank.txt` + `slots.build_predict_rank_prompt`——预测"用户"的排序(语义区别于生产 `self_rank` 的"客观质量",后者不动)。
- 提案(§2.1):`prompts/propose_dimension.txt` + `evolve._propose_dimension`——从**仅训练窗口**的残差/多通道评论/持续假设里,提一条 rubric 没词描述的新命名维度,或 `NONE`。
- 回测闸门(§2.2):`matins/memory/backtest.py::backtest_dimension`——在留出 batch 上比较 skill ± 候选维度的 τ;`passed` 要求 mean Δτ≥margin、**严格** frac_positive>0.5、且**新颖性检查**:lift 必须出现在 base 判错的 batch 上(纯 reweight 只锐化已对的 → 不过)。
- 固化(§2.3):通过则并入 skill 作**未批准**版本(Assisted),复用现有人审 + 版本化 + 回滚;附 Δτ/base-failure-lift/词汇重叠给人审。

**审计加固(对抗 Workflow 发现并已修)**:
1. **训练/留出隔离**:喂给提案者的假设也只取**训练证据**(`_train_only_hypotheses`),否则全历史假设会把留出信号泄漏进提案。
2. **新颖性可验证化**:gate 检查 lift 集中在 base-failure;另给人审一个词汇重叠信号 + 诚实措辞(高重叠标为"可能 reweight")——不再无条件宣称"evolved"。
3. **打分子集一致**:两折必须是同一 idea 集的完整排列,否则丢弃(防 Δτ 因子集错配失真)。
4. **gate 收紧**:frac_positive 严格 >0.5;数据门控提到 ≥8 可比 batch(留出 ≥3),测试用 4-idea batch 贴近生产。

**诚实的已知限制(延后,非缺陷):**
- **proposer == verifier 同一模型**:留出人类排序*锚定*奖励(自洽但无关人类的重排 Δτ≤0 被拒),但不能*完全解耦* lift 与模型自身先验。**人审是唯一完全独立的检查**。延后项:可选 `verifier_provider`(换一个模型打分)。
- **无 placebo 对照臂**:Δτ 未对"加任何自信指令文本"的泛化抬升做 length-matched 对照。延后(需三倍打分调用);新颖性集中度检查已部分缓解。
- **小样本统计功效低**:~4 idea/天,留出集小,早期 verifier 弱——故默认 **OFF**、数据门控、人审兜底。攒数月日志后再开 `consolidation.evolve_dimensions: true`。

---

## 研究支线 — 估计反身转移 h(Sₜ)(暂缓)

已记录"每天展示了什么"(`batches`/`ideas`)+ D 的随机化。足够随机化的 D 数据 + 因果估计 → 测"系统输出如何移动你的品味",即可证伪的反身漂移数据集。现在不写代码,只确保日志格式不挡路(Phase 1 给 D 打标、Phase 3 的自适应扰动正好为它积累干净数据)。

---

## 验证总纲

- 每个 Phase 配离线、无网络测试(沿用现有 FakeLLM / `Store(":memory:")` 惯例)。
- 每个 Phase 落地后跑全量 `pytest`,保持绿色再进下一个(checkpoint 纪律)。
- 全部完成后做一次全局扫描,删减冗余/投机实体(奥卡姆复核)。


## 基于LLM课堂对话自动分析
1. 关注教师的话语，基于APT框架（academically productive talk);
   * Elaboration: Say more, Revoice
   * Reasoning: Press for reasoning, Challenge
   * Listening: Restate
   * Thinking with others: Agree/disagree, Add on, Explain other
2. 分析逻辑：先判断讲话者（teacher, student), 再判断对谁讲话（同一位学生，还是多位学生），最后判断属于哪一类APT(elaboration, reasoning, restate, thinking with others);
   * Step 1: 教师是在对【同一个学生】继续追问，还是在邀请【其他学生】参与？
   * Step 2: 如果是对同一学生——press for reasoning, challenge, revoice, say more
   * Step 3: 如果是对其他学生——agree/disagree, explain others, restate, add on
4. Say more与多个APT类别重复了，可采用的方案：
   * 增加Say more的说明；
   * 增加较高阶（thinking with others）案例；
   * 把当前话论输入llm，利用情景信息提高准确性；
5. 通过多视角投票来确定最终的编码结果：
   * 视角一：标准 Prompt，要求精确分类（偏低温度）。 
   * 视角二：修改 Prompt，得到高召回结果。 
   * 合并：对于两个视角结果不一致的话轮，单独用第三个 Prompt 进行仲裁，只准选择一个代码。
6. 两轮分析思路 （层级决策树、分布聚合、二次确认） 《- chain-of-though, self-consistency, ensemble
   * 第一轮 (决策树 prompt): 快速决策树判断 → 大部分话轮直接出结果  （needs_review=True）
     * Type A 顺序：revoice → challenge → press → say_more。
     * Type B 顺序：explain_others → agree_disagree → restate → add_on。
   * 第二轮 (精确定义 prompt): 聚焦型确认 → 给出最终判断

## 整体思路1 （准确率一般）
![img.png](readme/img.png)

## 整体思路2
┌─────────────────────────────────────────────┐
│  Stage 1: Trigger + Addressee Detection     │
│  Input: 上下文窗口 + 当前教师话轮            │
│  Output: {trigger, addressee, evidence}      │
│  Prompt: 专注于"是不是follow-up"+"对谁说"   │
│  Few-shot: 大量trigger/non-trigger边界案例   │
└──────────────┬──────────────────┬───────────┘
               │                  │
      addressee =          addressee =
        same_student         other_student
               │                  │
               ▼                  ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Stage 2A: Type A    │  │  Stage 2B: Type B    │
│  Coding              │  │  Coding              │
│                      │  │                      │
│  Input: 上下文 +      │  │  Input: 上下文 +      │
│  已确认的addressee    │  │  已确认的addressee    │
│                      │  │                      │
│  Codes:              │  │  Codes:              │
│  - revoice           │  │  - explain_others    │
│  - challenge         │  │  - agree_disagree    │
│  - press_for_reason  │  │  - restate           │
│  - say_more (默认)   │  │  - add_on (默认)     │
│                      │  │                      │
│  Few-shot: 10-15个   │  │  Few-shot: 10-15个   │
│  Type A 专用案例     │  │  Type B 专用案例     │
└──────────────────────┘  └──────────────────────┘
               │                  │
               ▼                  ▼
┌─────────────────────────────────────────────┐
│  (可选) Stage 3: Confidence Review          │
│  仅对低置信度案例进行复查                     │
└─────────────────────────────────────────────┘

## 准确率记录
测试数据：121 个教师话轮，其中 99 个包含 APT move，22 个不包含 APT move。`restate` 和 `explain_others` 在该数据集中的 gold support 为 0，因此对应 F1 为 N/A，不参与 Macro F1 的平均。

| 版本 | 主要思路 | Macro F1 | 二分类 F1 | 错误话轮 | 主要问题 |
| --- | --- | ---: | ---: | ---: | --- |
| v3 | 单次预测 + 条件复查；一次判断 trigger、addressee 和 code | 0.586 | 0.851 | 50 | Trigger 错误较多，`revoice` 和 `challenge` 误报明显 |
| v5 | 分阶段判断：Stage 1 判断 trigger/addressee，Stage 2 按 Type A/B 分类，Stage 3 复查高风险结果 | 0.654 | 0.927 | 35 | 早期最佳基线；`revoice`、`challenge` 和部分 `add_on` 边界仍不稳定 |
| v6 | 多 code + 结构化 confidence；收紧 discussion continuation 和 addressee 判定 | 0.522 | 0.841 | 57 | 过度收紧导致大量 `add_on` 漏检，Trigger Recall 降至 0.747 |
| v7 | 在 v6 基础上恢复部分 discussion continuation，保留多 code 和复查失败兜底 | 0.598 | 0.906 | 38 | 比 v6 改善，但仍低于 v5；`add_on` Recall 为 0.481，`revoice` F1 为 0.222 |
| v8.1 | 高召回 trigger + immediate response evidence + 单主标签 + 条件复查 | 0.721 | **0.945** | 26 | 先前综合最优；`add_on → agree_disagree`、`add_on → missed` 仍存在 |
| v11 | 双重证据 trigger + risk flags + risk-driven review（首版） | 0.619 | 0.924 | 31 | 结构化输出成功，但 `add_on`、`revoice/challenge` 边界退化明显 |
| v11.1 | 针对 `add_on vs agree_disagree`、`revoice vs challenge` 做 prompt 微调 | 0.682 | **0.945** | 27 | Binary F1 追平 v8.1，但 Macro F1 仍落后 |
| v11.2 | 强化“主动作优先于最后点名对象”，修正 reasoning/addressee 边界 | 0.725 | 0.918 | 27 | 一度成为 Macro F1 最佳；`press_for_reasoning`、`challenge` 提升，但 `add_on` 漏检回升 |
| v11.3 | 在保留 reasoning 改善的前提下，专门救回 peer-linked `add_on` recall | **0.727** | 0.940 | 22 | 当前综合最优；仍有少量 `add_on → missed`、`revoice ↔ challenge` 边界问题 |

### 版本分析

* v3：作为单次预测基线，二分类 F1 为 0.851。
* v5：将任务拆成 trigger/addressee、Type A/Type B 分类和条件复查，Macro F1 达到 0.654，说明分阶段结构有效改善了整体触发判断。
* v6：首次正式支持多个 code，并使用结构化 confidence 控制复查；Macro F1 降至 0.522。
* v7：恢复了部分 active discussion 逻辑，并加入 Stage 3 失败时保留 Stage 2 结果的兜底。错误话轮降至 38，Macro F1 回升至 0.598，但仍未超过 v5。
* v8.1：在 v8 的高召回 trigger 基础上加入 immediate response evidence 和单主标签策略，在 `chris_moon_v1` 上达到 Binary F1 0.945、Macro F1 0.721，一度成为综合最优版本。
* v11：开始引入双重证据 trigger、`risk_flags`、`basis` 和 risk-driven review。首版结构化输出成功，但 Macro F1 仅 0.619，说明结构增强本身不会自动带来更好边界。
* v11.1：围绕 `add_on vs agree_disagree`、`revoice vs challenge` 做 prompt 微调后，Binary F1 回到 0.945，Macro F1 提升到 0.682，证明新结构可继续优化。
* v11.2：进一步强化“主动作优先于最后点名对象”，修正 `press_for_reasoning -> add_on` 等 addressee 边界后，Macro F1 提升到 0.725；但 Binary F1 降到 0.918，说明 peer-linked `add_on` recall 被压低。
* v11.3：继续针对 peer-linked `add_on` 的 trigger recall 做小步微调，在保住 reasoning/challenge 改善的同时，把 Binary F1 拉回到 0.940，并把 Macro F1 提升到 **0.727**，成为当前综合最优版本。

### 当前判断

* 如果只看 Macro F1，`v11.3` 是当前最佳版本。
* 如果只看 Binary F1，`v8.1` 和 `v11.1` 仍略高（0.945），但 `v11.3` 的 0.940 已非常接近。
* 如果综合考虑分类性能、可解释性和后续可迭代性，当前最值得继续使用和扩展的是 `v11.3`。

### v11.3 的主要收益

* `add_on` F1 提升到 **0.852**，明显高于 v11.2 的 0.723。
* `press_for_reasoning` 保持在高位（F1 = **0.894**）。
* `agree_disagree` 达到 **0.857**。
* `risk_flags` / `basis` / risk-driven review 已经进入稳定可用状态，后续误差分析比 v8.1 更强。

### v11.3 剩余问题

* 仍有少量 `add_on -> missed`，代表性话轮包括 Turn 27、45、160。
* `revoice ↔ challenge` 仍有边界混淆，代表性话轮包括 Turn 37、113。
* 个别长话轮（如 Turn 27、47）仍可能触发 Stage 1 JSON 解析失败。

### 下一步优化方向

* 继续以 `v11.3` 为主版本小步迭代，而不是重构框架。
* 优先继续补 `add_on` 的长话轮/解释后邀请场景。
* 单独加强 `revoice` 与 `challenge` 的对照示例。
* 如果后续目标转向泛化能力，应优先在多文件上验证，而不是只看 `chris_moon_v1`。

### 关键实验结果文件

* `coded_discourse/evaluation/v8.1/evaluation_chris_moon_v1_20260826_134636.json`
* `coded_discourse/evaluation/v11/evaluation_chris_moon_v1_20260827_214103.json`
* `coded_discourse/evaluation/v11/evaluation_chris_moon_v1_20260827_225736.json`
* `coded_discourse/evaluation/v11/evaluation_chris_moon_v1_20260828_102542.json`
* `coded_discourse/evaluation/v11/evaluation_chris_moon_v1_20260828_114547.json`
* `coded_discourse/evaluation/v11/error_analysis_chris_moon_v1_20260828_114547.json`

### v8 泛化表现（历史记录）
四个文件已经全部完成 v8 评估。结果说明：v8 在 Chris Moon 上表现不错，但跨课堂泛化明显下降。

文件 | 教师话轮 | Binary F1 | Macro F1 | 错误话轮
--- | ---: | ---: | ---: | ---:
Paper 1 Marking Conference | 82 | 0.615 | 0.324 | 27
Natalie Chan Lesson 1 | 82 | 0.807 | 0.669 | 15
STFACYTSS Charity | 30 | 0.526 | 0.214 | 11
Timothy Lim Lesson 1 | 26 | 0.720 | 0.403 | -

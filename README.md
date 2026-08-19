
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

## 当前实现流程

当前代码采用“单次预测 + 条件复查”的流程，不再使用早期版本的多视角投票方案。

1. **读取和清洗数据**
   * 从 Excel 读取课堂转录、说话人、话轮编号、时间戳和人工编码列。
   * 根据表头自动匹配 APT 代码列，并转换为统一的 JSON 结构。
   * 判断教师话轮，保留教师话轮的人工标注作为评估用 gold labels。
   * 如果清洗后的 JSON 没有过期，则直接复用，避免重复读取 Excel。
2. **构建上下文**
   * 对每个教师话轮提供前后文。
   * 额外识别目标话轮前后的学生，用于判断教师是在继续对话还是转向其他学生。
3. **第一次预测**
   * 使用 `prompts/v3` 中的编码 prompt，要求模型按三个步骤判断：
     1. 教师话轮是否回应了某位学生的具体发言；
     2. 教师面向同一位学生，还是其他学生/全班；
     3. 在对应类型中选择 APT 代码。
   * Type A（同一学生）只能使用 `say_more`、`revoice`、`press_for_reasoning`、`challenge`。
   * Type B（其他学生/全班）只能使用 `restate`、`agree_disagree`、`add_on`、`explain_others`。
4. **置信度判断和条件复查**
   * 对第一次结果进行规则检查，包括触发判断、对象类型、代码合法性和代码类型是否匹配。
   * `add_on` 和 `challenge` 等高风险结果，以及包含犹豫表达的结果，进入第二轮复查。
   * 复查使用更大的上下文窗口和 `confirm_prompt.txt`，重新判断触发条件、对象和最终代码。
   * 复查失败时保留第一次结果，并标记 `needs_review=True`，供人工检查。
5. **保存和评估**
   * 保存清洗后的转录 JSON 和模型预测 JSON。
   * 有人工标注时，计算是否存在 APT 行为的二分类指标，以及每个代码的 Precision、Recall、F1 和 Macro F1。
   * 对预测错误进行进一步分类，统计触发错误、对象类型错误、代码混淆、漏检和误检。

该流程的核心判断原则是：先确认教师是否真正回应了学生的具体想法，再判断教师面向谁，最后在对应的 APT 类型中进行精细分类。仅仅因为教师话轮紧接着学生发言，并不自动视为 APT 行为。

## 整体思路
![img.png](readme/img.png)


## 修改记录
* 多轮投票 -》 单次预测+精准复查
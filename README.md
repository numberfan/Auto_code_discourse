
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
6. 两轮分析思路 （显示决策树、分布聚合、二次确认） 《- chain-of-though, self-consistency, ensemble
   * 第一轮 (决策树 prompt): 快速决策树判断 → 大部分话轮直接出结果  （needs_review=True）
   * 第二轮 (精确定义 prompt): 聚焦型确认 → 给出最终判断

## 整体思路
![img.png](readme/img.png)

## 当前任务清单
优先级 1: 修复技术 bug（文件名拼写、输出格式不匹配、time.sleep）
         → 现在代码根本跑不对

优先级 2: 先用当前方案跑出一个完整结果，看 F1 数据
         → 没有数据之前，所有优化讨论都是猜测

优先级 3: 根据 error analysis 结果，定向优化
         → 如果错误主要集中在 trigger 误判，就调整 trigger 逻辑
         → 如果错误主要集中在 say_more vs press 混淆，就强化 few-shot
         → 如果错误主要集中在多码话轮，再引入软过滤

优先级 4: 消融实验，验证每个组件的边际贡献
         → 确认"分步聚合"真的比"直接投票"好
         → 确认"二次确认"真的提升了 needs_review 话轮的准确率
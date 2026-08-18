
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
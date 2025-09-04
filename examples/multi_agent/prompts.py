## 定义了一些 prompts 的模板，用于生成不同的 prompts


SOLVER_PROMPT_TEMPLATE = """{problem_statement}"""


def generate_rewriter_template(num_solutions: int) -> str:
    """根据解决方案数量动态生成重写模板"""
    solution_sections = []
    for i in range(num_solutions):
        solution_sections.append(f"#### Solution {i+1}\n{{solution{i+1}}}\n\n---")

    solutions_text = "\n".join(solution_sections)

    return f"""### Task: Solution Rewriting Based on Previous Solutions ###
You are being reactivated to revise your mathematical proof. You are provided with two documents:
1.  The problem you need to solve.
2.  Your {num_solutions} different "Previous Solutions".

Your sole task is to generate a new, correct version of your solution based on your previous discoveries in the provided {num_solutions} solutions.

Refer to the following {num_solutions} solutions and solve the problem.
---

### Problem

{{problem_statement}}

---

### Candidates Solution
{solutions_text}
"""


def generate_select_template(num_solutions: int) -> str:
    """根据解决方案数量动态生成选择模板"""
    solution_sections = []
    for i in range(num_solutions):
        solution_sections.append(f"#### Solution {i+1}\n{{solution{i+1}}}\n\n---")

    solutions_text = "\n".join(solution_sections)

    return f"""You will be given a challenging math problem followed by {num_solutions} solutions.
Your task is to systematically analyze these solutions to identify the most mathematically sound approach. 

You are provided with two documents:
1.  The problem you need to solve.
2.  Your {num_solutions} "Candidate Solutions".

Evaluation Process:
1. Initial Screening
- Group solutions by their final answers
- Identify and explain mathematical contradictions between different answers
- Eliminate solutions with clear mathematical errors

2. Detailed Analysis
For remaining solutions, evaluate:
- Mathematical precision and accuracy
- Logical progression of steps
- Completeness of mathematical reasoning
- Handling of edge cases or special conditions
- For solutions containing and addressing errors, evaluate the error identification and correction methodology.

3. Solution Comparison
Compare viable solutions based on:
- Efficiency of approach
- Clarity of mathematical reasoning
- Sophistication of method
- Robustness of solution (works for all cases)

Your response should include:
1. Brief analysis of conflicting answers
2. Detailed evaluation of mathematically sound solutions
3. Justification for eliminating incorrect solutions
4. Clear explanation for selecting the best approach

End your evaluation with exactly:
Judgment: IDX
where IDX is the index 1-{num_solutions} of the best solution

### Problem

{{problem_statement}}

---

### Candidate Solutions
{solutions_text}
"""

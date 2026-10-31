# AI-Use Statement

InternLM2.5-7B-chat and Gemma-2-27B-it are the models this project evaluates, not tools used to produce the work.

ChatGPT was used to brainstorm project rescoping ideas after assignment from Lee Language Lab project leads. Our team revised the M2 proposal ourselves.

Claude was used to help with dependency conflicts when trying to reproduce the LLMs results, using the error tracebacks and used package versions. We confirmed by using transformers 4.51.3 and sentencepiece 0.2.1 to work because the LLMs code fail on newer versions.

ChatGPT was used to suggest possible causes of the suspiciously high classical baseline scores compared to LLMs. One suggestion was the possibility of data artifacts, which helped guide our investigation where we found that the question at the end of each story leaked the answer. We are looking to address it with the question-ablation experiment in this milestone.

Claude was used to suggest code structure for the `run_question_ablation`, using our `run_baselines` function, which was written in the previous milestone, as the template. We wrote and reviewed the code and outputs.

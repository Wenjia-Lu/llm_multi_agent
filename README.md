# Improving Factuality and Reasoning in Language Models through Multiagent Debate

### [Project Page](https://composable-models.github.io/llm_debate/) | [Paper](https://arxiv.org/abs/2305.14325) 

[Yilun Du](https://yilundu.github.io/),
[Shuang Li](https://shuangli59.github.io/),
[Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab),
[Joshua B. Tenenbaum](https://scholar.google.com/citations?user=rRJ9wTJMUB8C&hl=en),
[Igor Mordatch](https://scholar.google.com/citations?user=Vzr1RukAAAAJ&hl=en)

This is a preliminary implementation of the paper "Improving Factuality and Reasoning in Language Models through Multiagent Debate". More tasks and settings will be released soon. 
You may see some additional debate logs [here](https://www.dropbox.com/sh/6kq5ixfnf4zqk09/AABezsYsBhgg1IQAZ12yQ43_a?dl=0).

Also, check out gauss5930's awesome implementation of multiagent debate on opensource LLMs [here](https://github.com/gauss5930/LLM-Agora)!

## Running experiments

The code for running arithmetic, GSM, biographies, and MMLU tasks may be found in the following subfolders

* ./math/ contains code for running math
* ./gsm/ contains code for running gsm
* ./biography/ contains code for running biographies
* ./mmlu/ contains code for running mmlu results.

**Math:**

To generate and evaluated answer for Math problems through multiagent debate, cd into the math directory and run:
	`python gen_math.py`
	
**Grade School Math:**

To generate answers for Grade School Math problems through multiagent debate, cd into the gsm directory and run:
	`python gen_gsm.py`

To evaluate the generated results of Grade School Math problems:
	`python eval_gsm.py`
	
You can download the GSM dataset [here](https://github.com/openai/grade-school-math)


**Biography:**

To generate answers for Biography problems through multiagent debate, cd into the biography directory and run:
	`python gen_conversation.py`

To evaluate the generated results for Biography problems:
	`python eval_conversation.py`
	
**MMLU:**

To generate answers for MMLU through multiagent debate, cd into the MMLU directory and run:
	`python gen_mmlu.py`

To evaluate the generated results of MMLU:
	`python eval_mmlu.py`
	
You can download the MMLU dataset [here](https://github.com/hendrycks/test)

**References:**
[1] Balaji, P. G.; and Srinivasan, D. 2010. An Introduction to Multi-Agent Systems,
 1–27. Berlin, Heidelberg: Springer Berlin Heidelberg. ISBN 978-3-642-14435-6.
[2] Dong, K. 2024. Large Language Model Applied in Multiagent System: A Survey.
 Applied and Computational Engineering, 109: 9–16.
[3] Du, Y.; Li, S.; Torralba, A.; Tenenbaum, J. B.; and Mordatch, I. 2024. Improving 
Factuality and Reasoning in Language Models through Multiagent Debate. 
arXiv:2305.14325.
[4] Eo, S.; Moon, H.; Zi, E. H.; Park, C.; and Lim, H. 2025. Debate Only When 
Necessary: Adaptive Multiagent Collaboration for Efficient LLM Reasoning.
arXiv:2504.05047.
[5] Ye, R.; Liu, X.; Wu, Q.; Pang, X.; Yin, Z.; Bai, L.; and Chen, S. 2025. X-MAS: 
Towards Building Multi-Agent Systems with Heterogeneous LLMs. arXiv:2505.16997.
[6] Zhang, H.; Cui, Z.; Chen, J.; Wang, X.; Zhang, Q.; Wang, Z.; Wu, D.; and Hu, S.
 2025. Stop Overvaluing Multi-Agent
[7] Debate – We Must Rethink Evaluation and Embrace Model Heterogeneity.
 arXiv:2502.08788.

If you would like to cite the paper, here is a bibtex file:
```
TODO
```

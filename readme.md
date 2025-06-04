# EMPO: Fully Unsupervised LLM Reasoning Incentivization


<a href="https://huggingface.co/collections/qingyangzhang/empo-67f9f7ad7817ebff4b664010">🤗 HF Models and Datasets Collection </a> |
<a href="https://arxiv.org/abs/2504.05812"> 📑 Arxiv Preprint </a>

For any questions, feel free to open an issue or directly contact to [QingyangZhang](qingyangzhang@tju.edu.cn), happy to help and disccuss!

If you find this work helpful, please consider to **star🌟** this repo. Thanks for your support!

## 🆕 News

- [2025-04-08] We introduce EMPO, which makes the first attempt on fully unsupervised LLM reasoning incentivization. Check out our arxiv preprint (first released at 2025.04.08): https://arxiv.org/abs/2504.05812
- [2025-04-30] We release the training and evaluation code for both mathematical reasoning and free-form natural reasoning tasks.
- [2025-06-04] We add the baselines suggested by Spurious Rewards. Our previous claim holds.

## 🎯 Overview

EMPO (Entropy Minimized Policy Optimization) does not require any supervised information for incentivizing reasoning capabilities (i.e., neither verifiable reasoning traces, problems with golden answers, nor additional pre-trained reward models). By continuously minimizing the predictive entropy of LLMs on unlabeled user queries in a latent semantic space, EMPO enables purely self-supervised evolution of reasoning capabilities with strong flexibility and practicality.

<p align="center">
<img src="./figs/EMPO.jpg" width="600" height="320">
</p>

## 🏗️ Quick Start
### Requirements

```
pip install -r requirements.txt
```


### Mathematical Reasoning

Training with EMPO:

```
sh empo-1.5B-NM-COT-20K.sh
```

Evaluation:

```
cd eval_math
sh test.sh
```

We directly borrow the evaluation scripts from the Online-DPO-R1 project. Please refer to [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1) for more details.

As suggested by [Spurious Rewards](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) and [Incorrect Baseline](https://safe-lip-9a8.notion.site/Incorrect-Baseline-Evaluations-Call-into-Question-Recent-LLM-RL-Claims-2012f1fbf0ee8094ab8ded1953c15a37#2022f1fbf0ee80cb9b18f7eac460410a), we adpot the same test prompt to both pre-RL Qwen Base models and RL-trained models. Besdies, we add Random+Format Reward Baseline for more comprehensive comparison. You can also modify the default test prompt in [here](https://github.com/QingyangZhang/EMPO/blob/main/eval_math/utils.py#L140) to investigate the influence of different test prompt.

| Model                          | Supervision    | MATH | Minerva Math | Olympiad Bench | AIME24 | AMC23 | Avg. |
|--------------------------------|----------------|------|--------------|----------------|--------|-------|------|
| **1.5B model**                 |                |      |              |                |        |       |      |
| Qwen2.5-Math                   | None           | 52.2 | 10.7         | 25.2           | 10.0   | 42.5  | 28.1 |
| Qwen2.5-Math-Instruct          | $\{q, r, a\}$  | 73.8 | 30.9         | 38.7           | 6.7    | 52.5  | 40.5 |
| Qwen2.5-Math w/SFT             | $\{q, r, a\}$  | 61.8 | 26.1         | 27.1           | 3.3    | 37.5  | 31.2 |
| Qwen2.5-Math w/Rand Format     | $\{q, a\}$     | 65.0 | 26.1         | 30.7           | 10.0   | 55.0  | 37.4 |
| Qwen2.5-Math w/GRPO            | $\{q, a\}$     | 75.2 | 32.0         | 33.6           | 16.7   | 52.5  | 42.0 |
| Qwen2.5-Math w/EMPO            | $\{q\}$        | 73.0 | 32.4         | 36.6           | 13.3   | 55.0  | 42.1 |
| **7B model**                   |                |      |              |                |        |       |      |
| Qwen2.5-Math                   | None           | 64.8 | 15.1         | 26.7           | 6.7    | 40.0  | 30.7 |
| Qwen2.5-Math Instruct          | $\{q, r, a\}$  | 82.8 | 43.8         | 41.2           | 16.7   | 62.5  | 49.4 |
| Qwen2.5-Math w/SFT             | $\{q, r, a\}$  | 72.2 | 34.6         | 33.2           | 10.0   | 45.0  | 39.0 |
| Qwen2.5-Math w/Rand Format     | $\{q, a\}$     | 73.0 | 26.5         | 37.0           | 26.7   | 52.5  | 43.1 |
| Qwen2.5-Math w/ODPO            | $\{q, a\}$     | 76.8 | 30.9         | 37.9           | 26.7   | 62.5  | 47.0 |
| Qwen2.5-Math w/GRPO            | $\{q, a\}$     | 77.8 | 39.7         | 39.1           | 20.0   | 57.5  | 46.8 |
| Qwen2.5-Math w/EMPO            | $\{q\}$        | 78.0 | 40.4         | 37.3           | 20.0   | 65.0  | 48.1 |

Noted that the pre-RL results in our EMPO are similar to that reported by [Absolute-Zero](https://arxiv.org/abs/2505.03335).

### Free-form Natural Reasoning

First, you need to uncomment the code on line 17 in src/open-r1/reward.py and use the verifier (a Small Language Model from [General Reasonor Project](https://huggingface.co/TIGER-Lab/general-verifier))

```
verifier = GeneralVerifier()
```

Very bad workaround here, forgive me :)

Then you can train with EMPO by:

```
sh empo-3B-NR-50K.sh
```

Noted that the verifier will be mapped to the last available GPU.

Assume that you have 8 GPUs in total, the default mapping would be:

GPU 0-5: Training, GPU 6: Generation (vllm), GPU 7: Verifier

Evaluation:

```
cd eval_natural
```
and then
```
sh script/eval/mmlu_pro.sh
```
or
```
sh script/eval/gpqa.sh
```

We adpot the test codebase from AI2's open-instruction for evaluation, with necessary modifications to extract the final answer for reasoning models.



## 🏆 Performance

<p align="center">
<img src="./figs/fancy_plot.jpg" width="300" height="160">
</p>

## 🙏 Acknowlegement

This repo is built upon [Semantic Entropy](https://github.com/jlko/semantic_uncertainty), [PFPO](https://github.com/microsoft/unilm/tree/master/PFPO), [Open-R1](https://github.com/huggingface/open-r1), [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1), and [DAPO](https://dapo-sia.github.io). We thank all these researchers for generously sharing their insights, model weights, data, and codes.

## 📑 Related Works

There are many awesome works related to this paper that you may also interested with:

- LLM Uncertainty Quantification: [Semantic Entropy (ICLR'23, Nature'24)](https://openreview.net/pdf?id=VD-AYtP0dve)
- Test-time Adaption in Computer Vision: [COME (ICLR'25)](https://openreview.net/pdf?id=506BjJ1ziZ)
- Presudo Feedback for LLM Reasoning: [PFPO (ICLR'25 spotlight)](https://arxiv.org/abs/2411.16345)
- Test-time Reinforcement Learning: [TTRL (cocurrent work)](https://arxiv.org/abs/2504.16084)

## 🖊️ Citation

If you find this work helpful, please consider to **star🌟** this repo. Thanks for your support!
```
@article{zhang2025right,
  title={Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization},
  author={Zhang, Qingyang and Wu, Haitao and Zhang, Changqing and Zhao, Peilin and Bian, Yatao},
  journal={arXiv preprint arXiv:2504.05812},
  year={2025}
}
```

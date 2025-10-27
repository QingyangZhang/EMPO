# EMPO: Fully Unsupervised LLM Reasoning Incentivization


<a href="https://huggingface.co/collections/qingyangzhang/empo-67f9f7ad7817ebff4b664010">🤗 HF Models and Datasets Collection </a> |
<a href="https://arxiv.org/abs/2504.05812"> 📑 Arxiv Preprint </a>

For any questions, feel free to open an issue or directly contact to [Qingyang Zhang](qingyangzhang@tju.edu.cn), happy to help and discuss!

If you find this repo helpful, please consider to **star🌟** this repo for support our work.

## News
- [2025-09-20] EMPO has been accepted by NeurIPS as a Spotlight! See you in San Diego!
- [2025-04-30] We release the training and evaluation code for both mathematical reasoning and free-form natural reasoning tasks.
- [2025-04-08] We introduce EMPO, which makes the first attempt on fully unsupervised LLM reasoning incentivization. Check out our arxiv preprint (first released at 2025.04.08): https://arxiv.org/abs/2504.05812

## Table of Contents
- [EMPO: Fully Unsupervised LLM Reasoning Incentivization](#empo-fully-unsupervised-llm-reasoning-incentivization)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [TRL Quick Start](#trl-quick-start)
  - [Verl Quick Start](#verl-quick-start)
  - [Acknowledgement](#acknowledgement)
  - [Related Works](#related-works)
  - [Citation](#citation)


## Overview

EMPO (Entropy Minimized Policy Optimization) does not require any supervised information for incentivizing reasoning capabilities (i.e., neither verifiable reasoning traces, problems with golden answers, nor additional pre-trained reward models). By continuously minimizing the predictive entropy of LLMs on unlabeled user queries, EMPO enables self-supervised RL for reasoning capabilities.

<p align="center">
<img src="./figs/EMPO.jpg" width="600" height="320">
</p>

## Repository Structure

This repository contains **two self-contained implementations** of EMPO:

- [`trl`](./trl/README.md): Based on Hugging Face’s trl, a cutting-edge library designed for post-training foundation models.

&nbsp;&nbsp; ↳ Built on commit [v0.14-release](https://github.com/huggingface/trl/commits/v0.14-release)

- [`verl-empo`](./verl-empo/README.md): Based on VERL, a high-performance RL training library designed for LLMs.

&nbsp;&nbsp; ↳ Built on commit [v0.4x](https://github.com/volcengine/verl/tree/v0.4.x)


Both are licensed under Apache 2.0 and include their respective `LICENSE` and `NOTICE` files.

## TRL Quick Start (deprecated)

> Developed upon trl 0.14.0. See [`trl`](./trl/README.md) for details.

```
cd trl
pip install -r requirements.txt
sh empo-1.5B-NM-COT-20K.sh
```

As trl 0.14.0 is already a relatively outdated training framework. We highly recommend verl for further development for efficiency and compatibility.

## Verl Quick Start
> Developed upon verl==0.4.x. See [`verl`](./verl/README.md) for details.

The recommended docker image with pre-built dependency is verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview.

### Data Preparation

Place the train and val data from ['math_data'](./math_data) in your local path.


### Train with EMPO

```
cd verl-empo
sh recipe/empo/scripts/run_empo_qwen2.5_math_7b.sh
```

Remember to modify the code [here](https://github.com/QingyangZhang/EMPO/blob/c0cfff769a8c926134d10423d6db3d4dd2adb913/verl-empo/verl/utils/tracking.py#L43C13-L43C160) if you want to track the training dynamics with Wandb.

### Evaluation

Load verl checkpoints by modifying trainer.resume_from_path.

Calculate pass@1 accuracy with greedy decoding by setting actor_rollout_ref.rollout.val_kwargs.do_sample=False.

### Experimental Results

We report pass@1 with greedy decoding for all datasets.

| Model                          | Supervision    | MATH500 | Minerva   | Olympiad Bench | AIME24 | AMC23 | Avg. |
|--------------------------------|----------------|------|--------------|----------------|--------|-------|------|
| **1.5B model**                 |                |      |              |                |        |       |      |
| Qwen2.5-Math Base              | None           | 66.4 | 19.1         | 33.8           | 3.3    | 42.5  | 33.0 |
| Qwen2.5-Math Instruct          | $\{q,a\}$      | 75.2 | 33.8         | 42.8           | 6.7    | 52.5  | 42.2 |
| Qwen2.5-Math w/GSPO            | $\{q,a\}$      | 78.0 | 37.1         | 39.1           | 10.0   | 50.0  | 42.8 |
| Qwen2.5-Math w/EMPO            | $\{q\}$        | 77.6 | 36.0         | 39.5           | 10.0   | 50.0  | 42.6 |
| **3B model**                   |                |      |              |                |        |       |      |
| OctoThinker-Long Base          | None           | 15.8 |  2.9         |  7.5           |  0.0   | 12.5  |  7.7 |
| OctoThinker-Long Zero          | $\{q,a\}$      | 69.6 | 27.6         | 32.0           | 13.3   | 42.5  | 37.0 |
| OctoThinker-Long w/GSPO        | $\{q,a\}$      | 65.0 | 23.5         | 27.3           |  6.7   | 32.5  | 31.0 |
| OctoThinker-Long w/EMPO        | $\{q\}$        | 60.6 | 17.3         | 23.6           |  6.7   | 30.0  | 27.6 |
| **7B model**                   |                |      |              |                |        |       |      |
| Qwen2.5-Math Base              | None           | 70.2 | 12.5         | 30.8           | 10.0   | 45.0  | 33.7 |
| Qwen2.5-Math Instruct          | $\{q,a\}$      | 80.8 | 41.9         | 49.2           | 13.3   | 67.5  | 50.5 |
| Qwen2.5-Math w/GSPO            | $\{q,a\}$      | 82.4 | 45.2         | 47.6           | 23.3   | 60.0  | 51.7 |
| Qwen2.5-Math w/EMPO            | $\{q\}$        | 81.4 | 42.3         | 46.1           | 23.3   | 65.0  | 51.6 |

Noted that due to different 1) evaluation proxy 2) RL framework and 3) GPU hardware, the above results are different from those reported in our early preprint.

### Models and Wandb log

|HF Models | Wandb Logs|
|----------|-----------|
|[Qwen2.5-Math-1.5B w/ EMPO](https://huggingface.co/qingyangzhang/EMPO-Qwen2.5-Math-7B)       | [Wandb Report](https://api.wandb.ai/links/zqyoung1127-tianjin-university/kquibtp8)       |
|[Qwen2.5-Math-7B w/ EMPO](https://huggingface.co/qingyangzhang/EMPO-Qwen2.5-Math-1.5B)       | [Wandb Report](https://api.wandb.ai/links/zqyoung1127-tianjin-university/kquibtp8)       |
|[OctoThinker-3B-Long-Base w/ EMPO](https://huggingface.co/qingyangzhang/EMPO-OctoThinker-3B-Long-Base)       | [Wandb Report](https://api.wandb.ai/links/zqyoung1127-tianjin-university/kquibtp8)      |

## Acknowledgement

This repo is built upon [Semantic Entropy](https://github.com/jlko/semantic_uncertainty), [Open-R1](https://github.com/huggingface/open-r1), [Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1), and [TTRL](https://github.com/PRIME-RL/TTRL). We thank all these researchers for generously sharing their insights, model weights, data, and codes.


## Related Works

There are many awesome works related to this paper that you may also interested with:

- LLM Uncertainty Quantification: [Semantic Entropy (ICLR'23, Nature'24)](https://openreview.net/pdf?id=VD-AYtP0dve)
- Test-time Adaption in Computer Vision: [COME (ICLR'25)](https://openreview.net/pdf?id=506BjJ1ziZ)
- Pseudo Feedback for LLM Reasoning: [PFPO (ICLR'25 spotlight)](https://arxiv.org/abs/2411.16345)
- Test-time Reinforcement Learning: [TTRL (cocurrent work)](https://arxiv.org/abs/2504.16084)

More papers are listed in [Awesome Reinforcement Learning with Internal Reward Paper list](https://github.com/QingyangZhang/Label-Free-RLVR).

## Citation

If you find this work helpful, please consider to **star🌟** this repo. Thanks for your support!
```
@article{zhang2025right,
  title={Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization},
  author={Zhang, Qingyang and Wu, Haitao and Zhang, Changqing and Zhao, Peilin and Bian, Yatao},
  journal={Advances in neural information processing systems},
  year={2025}
}
```

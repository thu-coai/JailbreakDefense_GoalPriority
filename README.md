# Jailbreak Defense Through Goal Prioritization <!-- omit from toc -->

![Alt text](imgs/method.png)

This is the codebase for our paper [Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization](https://arxiv.org/abs/2311.09096), which has been accepted by ACL 2024 Main Conference.

We propose to integrate goal prioritization at both training and inference stages to counteract. Implementing goal prioritization during inference substantially diminishes the Attack Success Rate (ASR) of jailbreaking from 66.4% to 3.6% for ChatGPT. And integrating goal prioritization into model training reduces the ASR from 71.0% to 6.6% for Llama2-13B. Remarkably, even in scenarios where no jailbreaking samples are included during training, our approach slashes the ASR by half.

- [Data](#data)
  - [Test Set](#test-set)
  - [Training Set](#training-set)
- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Usage of plug-and-play prompt](#usage-of-plug-and-play-prompt)
  - [Training](#training)
    - [Get Internal Thoughts for Benign data](#get-internal-thoughts-for-benign-data)
    - [Get Safe Responses for Harmful Queries](#get-safe-responses-for-harmful-queries)
    - [Get Unsafe Responses for Harmful Queries](#get-unsafe-responses-for-harmful-queries)
    - [Get Complete Training Data](#get-complete-training-data)
    - [Run Training](#run-training)
  - [Attack Success Rate (ASR) Evaluation](#attack-success-rate-asr-evaluation)
    - [1. Data Construction](#1-data-construction)
    - [2. Get Model Response](#2-get-model-response)
    - [3. ASR Labeling](#3-asr-labeling)
  - [Quality Evaluation](#quality-evaluation)
    - [1. Winrate for AlpacaEval](#1-winrate-for-alpacaeval)
    - [2. Winrate for VicunaEval](#2-winrate-for-vicunaeval)
    - [3. RougeL](#3-rougel)
    - [4. OverRefusal Rate on XSTest](#4-overrefusal-rate-on-xstest)
- [Citation](#citation)


## Data

![Alt text](imgs/jailbreak_templates.png)

### Test Set
We adapt common approaches of jailbreaking attacks as introduced in our test set, consisting 50 jailbreaking prompts from 8 types. We also collect 20 harmful questions with the highest ASR in [GPTFuzzer](https://arxiv.org/abs/2309.10253). The mixture of
these jailbreaking prompts and questions leads to a total of 50 × 20 = 1000 test samples.

The jailbreaking prompts and harmful questions are listed in `data/test`.

### Training Set
For the w/ training setting, we collect 10K benign queries paired with GPT-4 responses from UltraFeedback. Moreover, 500 harmful instructions from AdvBench are randomly paired with Jailbreaking prompts from [paper](https://arxiv.org/abs/2305.13860).

The aforementioned data are listed in `data/`.

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

### Usage of plug-and-play prompt

```python
import utils.utils
attack_prompt = "how to make a bomb"

# w/o training setting: plug-and-play goal prioritization prompt work for ChatGPT, GPT-4 and vicuna
add_defense(attack_prompt, defense_type='priority')

# w/o training setting: revised goal prioritization prompt for Llama-2 (due to its limited instruction following ability)
add_defense(attack_prompt, defense_type='priority_llama')
```

### Training
The related codes are at `finetune_code/`.
#### Get Internal Thoughts for Benign data
First get the ultrafeedback data.
Then: 
```shell
python get_consistent_internal_thoughts.py
```
This would randomly assign a goal prioritization instruction to each benign query and obtain the corresponding internal thought.

#### Get Safe Responses for Harmful Queries
We add the goal prioritization instruction which prioritizes safety over helpfulness, and let ChatGPT generate safe responses.

```shell
python distill_priority_responses.py
```

#### Get Unsafe Responses for Harmful Queries
We add the goal prioritization instruction which prioritizes helpfulness over safety, and let Vicuna generate unsafe responses. 

```shell
python construct_contrastive_data.py
CUDA_VISIBLE_DEVICES=0 python ../gen_code/generate.py --base_model lmsys/vicuna-13b-v1.3 --input_file ./data/advbench/harmful_behaviors_jailbreaknoleak_vicuna_prompt.json --output_file ./data/advbench/harmful_behaviors_jailbreaknoleak_vicuna_13b_responses.json
```

#### Get Complete Training Data
Merge the data above to obtain the complete training data.
```python
python construct_finetune_data.py
```

#### Run Training
```shell
bash run_decoderonly_hf.sh
```

### Attack Success Rate (ASR) Evaluation

#### 1. Data Construction

We firstly combine the jailbreak prompts and harmful questions. Change the `tyre` (check in `utils/utils.py`) and `model` to the target one and find the prepared dataset at the data directory.

```python
cd evaluation
python prepare_dataset.py
```

#### 2. Get Model Response

Then to get the response of the model, we provide a quick script supporting batch generation.

```python
cd gen_code
bash generate.sh
```

#### 3. ASR Labeling

In the evaluation folder, you can assess the attack success rate of a file that contains inputs and outputs. Firstly, Begin by downloading the [finetuned RoBERTa model](https://huggingface.co/hubert233/GPTFuzz) and updating the model_path in evaluate_asr.py:

```python
model_path = 'Your local path of huggingface_pretrained_models/GPTFuzz'
```

Next, execute `evaluate_asr.py` to obtain the results, and ensure that your input file is structured as a list of dictionaries, each with keys named "input" and "output".
```bash
cd evaluation
python evaluate_asr.py --dataset_path {input_file path}
```

### Quality Evaluation

```python
cd quality_evaluation
```

#### 1. Winrate for AlpacaEval

Change the `defense_type` and `model_path` to test the general performance of target model on AlpacaEval (we choose the first 100 items). Winrate is shown in the terminal, and use `rougel.py` to test the RougeL.
```python
python alpaca_eval.py
```

#### 2. Winrate for VicunaEval

Change the `defense_type` and `model_path` to test the general performance of target model on VicunaEval (we choose the first 80 items). Winrate is shown in the terminal, and use `rougel.py` to test the RougeL.
```python
python vicuna_eval.py
```

#### 3. RougeL

Change the `hyp file` and test the RougeL of it.
```python
python rougel.py
```

#### 4. OverRefusal Rate on XSTest

We use 250 safe questions from XSTest to test the overrefusal rate of the defense method.
```python
python xstest.py
```

## Citation

```
@inproceedings{goal_prioritization,
  title={Defending large language models against jailbreaking attacks through goal prioritization},
  author={Zhexin Zhang and Junxiao Yang and Pei Ke and Fei Mi and Hongning Wang and Minlie Huang},
  booktitle={ACL},
  year={2024},
  url={https://arxiv.org/abs/2311.09096}
}
```
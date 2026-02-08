# THE ACHILLES' HEEL OF LLMS

This repository implements our perturbation-based causal identification method for locating critical neurons in Large Language Models (LLMs) that are indispensable for their capabilities.

## Overview

This research investigates whether LLMs contain a small subset of critical neurons similar to biological neural networks. We propose a systematic method to locate such critical neurons and find that disabling as few as three neurons can catastrophically impair a 72B-parameter model with over 1.1 billion neurons, driving perplexity up by 20 orders of magnitude.
![Motivation diagram](assets/motivation.png)


## Key Findings

1. **Ultra-Sparse Vulnerability**: 3-45 critical neurons can collapse billion-parameter models
2. **Architectural Concentration**: Critical neurons cluster in outer layers, particularly in MLP down_proj components
3. **Phase Transition Behavior**: Sharp performance collapse rather than gradual degradation

## Environment Setup

```bash
pip install -r requirements.txt
```

## Model Download

Download models from Hugging Face Hub:

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download example models used in our experiments
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./models/llama-3.2-3b
huggingface-cli download google/gemma-7b --local-dir ./models/gemma-7b  
huggingface-cli download microsoft/DialoGPT-medium --local-dir ./models/dialogpt
```

## Method

Our Perturbation-based Causal Identification of Critical Neurons method consists of a two-stage optimization process that mirrors the spirit of lesion studies to locate critical neurons. The method operates on any text as input and uses controlled perturbations to systematically identify neurons indispensable for the model's function.


### Stage 1: Neuron Importance Evaluation

**File**: `method/1_importance_evaluation.py`

In the first stage, the method injects controlled noise into the model's input and measures the resulting activation differences across neurons, thereby generating a ranked list of candidates most likely to influence model behavior.

```bash
python method/1_importance_evaluation.py 
    --model_path "./models/llama-3.2-3b" 
    --input_text "Later reviews were more positive. In just over 29 minutes, Bookends is stunning in its vision of a bewildered America in search of itself, said AllMusic writer Thom Jurek, who gave it five stars out of five. Pitchfork Media's Stephen M. Deusner called Bookends the moment in which the duo were settling into themselves, losing their folk revival pretensions and emphasizing quirky production techniques to match their soaring vocals. The A.V. Club called it the group's most musically and conceptually daring album." \
    --noise_scale 5 
    --num_samples 100 
    --output_file "./neurons/importance_ranked_neurons.json"
```

**Output**: This stage generates a JSON file containing approximately 10,000 neurons ranked in descending order of importance scores, providing the candidate pool for critical neuron identification.

**Key Parameters:**

- `--noise_scale`: Controls perturbation magnitude (α = 5)
- `--num_samples`: Monte Carlo samples for importance estimation (K = 100)
- `--input_text`: Input text for perturbation analysis (minimum 10 tokens required)

### Stage 2: Critical Neuron Identification

**File**: `method/2_neuron_identification.py`

In the second stage, we sequentially mask the top-ranked neurons in a greedy manner, closely monitoring changes in model perplexity. This allows us to causally determine which neurons are truly indispensable for the model's function.

```bash
python method/2_neuron_identification.py 
    --model_path "./models/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --test_text "Later reviews were more positive. In just over 29 minutes, Bookends is stunning in its vision of a bewildered America in search of itself, said AllMusic writer Thom Jurek, who gave it five stars out of five. Pitchfork Media's Stephen M. Deusner called Bookends the moment in which the duo were settling into themselves, losing their folk revival pretensions and emphasizing quirky production techniques to match their soaring vocals. The A.V. Club called it the group's most musically and conceptually daring album." 
    --output_file "./results/critical_neurons_result.json"
```

**Output**: This stage identifies the minimal critical neuron set (typically 3-45 neurons) that must be masked to cause catastrophic performance degradation. The result determines how many of the top-ranked neurons from the 10,000 candidates need to be masked to trigger model collapse.



## Evaluation Scripts

The evaluation framework includes scripts for both language modeling tasks and downstream benchmarks:

### Language Modeling Evaluation

#### WikiText-103 Evaluation

```bash
python evaluation/wikitext_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --masking_levels "0,4" 
    --max_samples 1801350 
    --output_file "./results/wikitext_results.json"
```

#### C4 Dataset Evaluation

```bash
python evaluation/c4_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --masking_levels "0,4" 
    --max_samples 300000 
    --output_file "./results/c4_results.json"
```

### Downstream Task Evaluation

#### MMLU-Pro (Advanced Multi-domain Knowledge)

```bash
python evaluation/mmlu_pro_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --masking_levels "0,4" 
    --output_file "./results/mmlu_pro_results.json"
```

#### IFEval (Instruction Following)

```bash
python evaluation/ifeval_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --masking_levels "0,4" 
    --output_file "./results/ifeval_results.json"
```

#### GPQA-Diamond (Graduate-level Scientific Reasoning)

```bash
python evaluation/gpqa_evaluator.py
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --masking_levels "0,4" 
    --output_file "./results/gpqa_results.json"
```

#### HumanEval (Code Generation)

```bash
python evaluation/humaneval_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json"
    --masking_levels "0,4" 
    --output_file "./results/humaneval_results.json"
```

#### MATH (Mathematical Problem Solving)

```bash
python evaluation/math_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --masking_levels "0,4" 
    --output_file "./results/math_results.json"
```

#### MGSM (Multilingual Mathematical Reasoning)

```bash
python evaluation/mgsm_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --masking_levels "0,4" 
    --output_file "./results/mgsm_results.json"
```

#### SimpleQA (Factual Question Answering)

```bash
python evaluation/simpleqa_evaluator.py 
    --model_path "../model/llama-3.2-3b"
    --neuron_file "./neurons/importance_ranked_neurons.json" 
    --masking_levels "0,4" 
    --output_file "./results/simpleqa_results.json"
```

### Parameters

**Common Parameters:**

- `--model_path`: Path to the model directory
- `--neuron_file`: JSON file containing identified critical neurons from Stage 2
- `--masking_levels`: Comma-separated list of masking levels (e.g., "0,4" means evaluate with 0 and 4 neurons masked)
- `--output_file`: Path to save evaluation results

**Dataset-specific Parameters:**

- ```
  --max_samples
  ```

  : Maximum number of samples to evaluate (varies by dataset size)

  - WikiText-103: 1,801,350 samples
  - C4: 300,000 samples
  - Downstream tasks: Full evaluation sets



## Comparison Experiments

This section implements comparison studies examining different neuron location strategies to validate the effectiveness of our perturbation-based method against alternative approaches.

### Random Masking

Random neuron selection serves as the baseline to demonstrate that our findings are not due to chance effects.

```bash
python comparison/random_masking.py 
    --model_path "../model/llama-3.2-3b" 
    --max_samples 1000 
    --masking_levels "100,200,300,400,500,600,700,800,900,1000" 
    --num_runs 10 
    --output_file "./results/random_masking_results.json"
```

**Output:** This generates perplexity evaluation results for randomly masked neurons at different masking levels, averaged over 10 trials to ensure statistical reliability. 

**Parameters:**

- `--masking_levels`: Number of randomly selected neurons to mask
- `--num_runs`: Multiple runs for statistical reliability (averaged over 10 trials)
- `--max_samples`: Number of samples for perplexity evaluation

### Activation Magnitude (AM) Ranking

This method ranks neurons by their activation values, representing a static importance measure.

```bash
python comparison/activation_masking.py 
    --model_path "../model/llama-3.2-3b" 
    --input_text "Later reviews were more positive. In just over 29 minutes, Bookends is stunning in its vision of a bewildered America in search of itself, said AllMusic writer Thom Jurek, who gave it five stars out of five. Pitchfork Media's Stephen M. Deusner called Bookends the moment in which the duo were settling into themselves, losing their folk revival pretensions and emphasizing quirky production techniques to match their soaring vocals. The A.V. Club called it the group's most musically and conceptually daring album." 
    --output_file "./neurons/activation_ranked_neurons.json"
```

**Output:** This generates a JSON file containing 10,000 neurons ranked by activation magnitude in descending order.

### Gradient Magnitude (GM) Ranking

This method ranks neurons by the magnitude of perplexity gradients with respect to each neuron.

```bash
python comparison/gradient_masking.py 
    --model_path "../model/llama-3.2-3b" 
    --input_text "Later reviews were more positive. In just over 29 minutes, Bookends is stunning in its vision of a bewildered America in search of itself, said AllMusic writer Thom Jurek, who gave it five stars out of five. Pitchfork Media's Stephen M. Deusner called Bookends the moment in which the duo were settling into themselves, losing their folk revival pretensions and emphasizing quirky production techniques to match their soaring vocals. The A.V. Club called it the group's most musically and conceptually daring album." 
    --output_file "./neurons/gradient_ranked_neurons.json"
```

**Output:** This generates a JSON file containing 10,000 neurons ranked by gradient magnitude in descending order.

## Comparative Evaluation

After obtaining ranked neuron lists from each method, we evaluate their effectiveness by progressively masking neurons and measuring perplexity degradation.

### Activation Magnitude Evaluation

```bash
python method/wikitext_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/activation_ranked_neurons.json" 
    --masking_levels "100,200,300,400,500,600,700,800,900,1000" 
    --max_samples 1000 
    --output_file "./results/am_comparison_results.json"
```

### Gradient Magnitude Evaluation

```bash
python method/wikitext_evaluator.py 
    --model_path "../model/llama-3.2-3b" 
    --neuron_file "./neurons/gradient_ranked_neurons.json" 
    --masking_levels "100,200,300,400,500,600,700,800,900,1000" 
    --max_samples 1000 
    --output_file "./results/gm_comparison_results.json"
```



## Parameter Sensitivity Analysis

### Noise Scale (α) Ablation

The noise scale controls the perturbation magnitude in Stage 1. We test different values to determine the optimal sensitivity threshold.

```bash
# Test different noise scales (α = 1, 3, 5, 7, 10)
for alpha in 1 3 5 7 10; do
    python method/1_importance_evaluation.py 
        --model_path "./models/llama-3-8b" 
        --input_text "Later reviews were more positive. In just over 29 minutes, Bookends is stunning in its vision of a bewildered America in search of itself, said AllMusic writer Thom Jurek, who gave it five stars out of five. Pitchfork Media's Stephen M. Deusner called Bookends the moment in which the duo were settling into themselves, losing their folk revival pretensions and emphasizing quirky production techniques to match their soaring vocals. The A.V. Club called it the group's most musically and conceptually daring album." 
        --noise_scale ${alpha} 
        --num_samples 100 
        --output_file "./ablation/noise_scale_${alpha}_neurons.json"
done
```

### Sample Size (K) Ablation

The number of Monte Carlo samples affects the stability of importance estimation through the Law of Large Numbers.

```bash
# Test different sample sizes (K = 25, 50, 75, 100, 125)
for samples in 25 50 75 100 125; do
    python method/1_importance_evaluation.py 
        --model_path "./models/llama-3-8b" 
        --input_text "Later reviews were more positive. In just over 29 minutes, Bookends is stunning in its vision of a bewildered America in search of itself, said AllMusic writer Thom Jurek, who gave it five stars out of five. Pitchfork Media's Stephen M. Deusner called Bookends the moment in which the duo were settling into themselves, losing their folk revival pretensions and emphasizing quirky production techniques to match their soaring vocals. The A.V. Club called it the group's most musically and conceptually daring album." \
        --noise_scale 5 
        --num_samples ${samples} 
        --output_file "./ablation/num_samples_${samples}_neurons.json"
done
```

### Input Text Robustness

#### Different Text Types

```bash
# Wikipedia text
python method/1_importance_evaluation.py 
    --model_path "./models/llama-3-8b" 
    --input_text "The polar bear (Ursus maritimus) is a large carnivorous mammal native to the Arctic. It primarily hunts seals, relying on sea ice for habitat and hunting." 
    --noise_scale 5 
    --num_samples 100 
    --output_file "./ablation/wikipedia_text_neurons.json"
```


## Citation

If you find this repository or our findings useful in your research, please cite our paper:

```bibtex
@misc{qin2025achillesheelllmsaltering,
      title={The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities}, 
      author={Zixuan Qin and Kunlin Lyu and Qingchen Yu and Yifan Sun and Zhaoxin Fan},
      year={2025},
      eprint={2510.10238},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={[https://arxiv.org/abs/2510.10238]}, 
}
```
















import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import argparse
import logging
from tqdm import tqdm
import math
import random
from collections import defaultdict
from datasets import load_dataset
import pandas as pd
import re
import sys
import os
from typing import Dict, List, Any, Union, Tuple
import string
import transformers

def setup_logging(log_file="simpleqa_evaluation.log", log_level=logging.INFO):
    """Configure logging settings"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleQAEvaluator:
    def __init__(self, model_path, neuron_file=None, device="auto", seed=42, logger=None):
        """
        Initialize SimpleQA evaluator with direct answer approach
        
        Args:
            model_path: Path to the model
            neuron_file: Path to neuron importance file (JSON), optional
            device: Computing device ("auto" for multi-GPU)
            seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        set_seed(seed)
        self.seed = seed
        
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        self.logger.info(f"Available GPUs: {self.num_gpus}")

        self._load_model_multi_gpu(model_path)

        self.neurons = []
        if neuron_file:
            self.logger.info(f"Loading neuron importance data from: {neuron_file}")
            with open(neuron_file, 'r') as f:
                neuron_data = json.load(f)
            
            self.neurons = neuron_data.get('top_neurons', neuron_data.get('neurons', []))
            self.logger.info(f"Loaded {len(self.neurons)} neurons for masking")
        else:
            self.logger.info("No neuron file provided, will only test baseline accuracy")

        self.masking_hooks = []
        self.current_masked_neurons = []

        self._setup_direct_answer_prompts()
        
    def _load_model_multi_gpu(self, model_path):
        """Load model with multi-GPU support"""
        self.logger.info(f"Loading model with multi-GPU: {model_path}")

        self.logger.info(f"Transformers version: {transformers.__version__}")
        self.logger.info(f"PyTorch version: {torch.__version__}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={i: "80GB" for i in range(self.num_gpus)} if self.num_gpus > 0 else None
        )
        
        self.model.eval()
        self.logger.info("Multi-GPU model loading completed")

        self._print_device_map()
        
    def _print_device_map(self):
        """Print model device allocation information"""
        self.logger.info("Model device allocation:")
        if hasattr(self.model, 'hf_device_map'):
            device_summary = defaultdict(int)
            for layer_name, device in self.model.hf_device_map.items():
                device_summary[str(device)] += 1
            
            for device, count in device_summary.items():
                self.logger.info(f"  Device {device}: {count} layers")
        else:
            self.logger.info("  Device map not available")
    
    def _setup_direct_answer_prompts(self):
        """Setup prompt templates for direct answer generation"""
        self.direct_answer_prompts = [
            "Answer the following question with only the direct answer. Do not provide explanations, reasoning, or additional text.\n\nQuestion: {question}\nAnswer:",
            "Question: {question}\nProvide only the answer without any explanation:\nAnswer:",
            "{question}\n\nGive me just the answer:",
            "Q: {question}\nA:",
        ]
        
        self.logger.info("Setup direct answer prompt templates")
    
    def load_dataset(self, split='test'):
        """Load SimpleQA dataset using HuggingFace datasets"""
        self.logger.info(f"Loading SimpleQA dataset from HuggingFace, split: {split}")
        
        try:
            ds = load_dataset("basicv8vc/SimpleQA", split=split)
            self.logger.info(f"Dataset loaded successfully from HuggingFace. Total items: {len(ds)}")
            
            samples = []
            for item in tqdm(ds, desc=f"Loading {split} data"):
                if isinstance(item, dict):
                    samples.append(item)
                else:
                    samples.append(dict(item))
            
            self.logger.info(f"Loaded {len(samples)} samples from SimpleQA {split}")

            if samples:
                sample = samples[0]
                self.logger.info(f"Sample structure: {list(sample.keys())}")
                if 'problem' in sample:
                    self.logger.info(f"Sample problem: {sample['problem'][:200]}...")
                if 'answer' in sample:
                    self.logger.info(f"Sample answer: {sample['answer'][:100]}...")
                for field in ['question', 'query', 'text', 'input']:
                    if field in sample:
                        self.logger.info(f"Sample {field}: {str(sample[field])[:200]}...")
                        break
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Failed to load SimpleQA dataset from HuggingFace: {e}")
            raise
    
    def is_response_meaningful(self, response):
        """Check if response is meaningful"""
        if not response or not response.strip():
            return False
        
        cleaned = response.strip()

        if len(cleaned) < 1:
            return False

        alphanumeric_chars = sum(1 for c in cleaned if c.isalnum())
        if alphanumeric_chars < 1:  
            return False

        if len(cleaned) > 3:
            unique_chars = len(set(cleaned.lower()))
            if unique_chars < 2: 
                return False

        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(cleaned)):
            if cleaned[i] == cleaned[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        if max_consecutive > 10:
            return False
        
        return True
    
    def create_direct_answer_prompt(self, question: str, template_idx: int = 0) -> str:
        """Create a direct answer prompt using the specified template"""
        template = self.direct_answer_prompts[template_idx % len(self.direct_answer_prompts)]
        return template.format(question=question)
    
    def normalize_answer(self, text: str) -> str:
        """Normalize text for answer comparison"""
        if not text:
            return ""

        normalized = text.lower().strip()

        normalized = ' '.join(normalized.split())
        
        normalized = re.sub(r'[.!?;,"\']', '', normalized)
        
        return normalized
    
    def extract_answer_words(self, text: str) -> set:
        """Extract meaningful words from answer for comparison"""
        normalized = self.normalize_answer(text)
        if not normalized:
            return set()

        words = normalized.split()
        meaningful_words = set()
        
        for word in words:
            word_clean = re.sub(r'[^\w\s]', '', word)
            if len(word_clean) >= 1: 
                meaningful_words.add(word_clean)
        
        return meaningful_words
    
    def evaluate_answer_match(self, predicted: str, ground_truth: str) -> Dict:
        """
        Evaluate if predicted answer contains the ground truth answer
        using word-level matching
        """
        if not predicted or not ground_truth:
            return {
                'contains_answer': False,
                'exact_match': False,
                'word_overlap_ratio': 0.0,
                'confidence': 0.0
            }
        
        pred_normalized = self.normalize_answer(predicted)
        gt_normalized = self.normalize_answer(ground_truth)

        exact_match = pred_normalized == gt_normalized

        pred_words = self.extract_answer_words(predicted)
        gt_words = self.extract_answer_words(ground_truth)
        
        if not gt_words:
            return {
                'contains_answer': True,
                'exact_match': exact_match,
                'word_overlap_ratio': 1.0,
                'confidence': 1.0
            }

        overlap = pred_words & gt_words
        overlap_ratio = len(overlap) / len(gt_words)

        contains_answer = overlap_ratio > 0

        response_length_penalty = min(1.0, 20.0 / len(predicted.split())) if predicted else 0.0
        confidence = overlap_ratio * response_length_penalty
        
        return {
            'contains_answer': contains_answer,
            'exact_match': exact_match,
            'word_overlap_ratio': overlap_ratio,
            'confidence': confidence
        }
    
    def _setup_masking_hooks(self, masked_neurons):
        """Set up hooks to mask specific neurons during forward pass"""
        self._clear_masking_hooks()

        neurons_by_layer = defaultdict(list)
        for neuron in masked_neurons:
            neurons_by_layer[neuron['layer_name']].append(neuron)
        
        hook_count = 0

        for layer_name, layer_neurons in neurons_by_layer.items():
            def create_layer_mask_hook(layer_name, layer_neurons):
                def hook(module, input, output):
                    if not isinstance(output, torch.Tensor):
                        return output
                    
                    masked_output = output.clone()
                    
                    for neuron in layer_neurons:
                        try:
                            if 'batch_idx' in neuron and 'seq_idx' in neuron and 'hidden_idx' in neuron:
                                batch_idx = neuron['batch_idx']
                                seq_idx = neuron['seq_idx']
                                hidden_idx = neuron['hidden_idx']
                                
                                if (batch_idx < masked_output.shape[0] and 
                                    hidden_idx < masked_output.shape[-1]):
                                    if masked_output.dim() == 3:
                                        masked_output[batch_idx, :, hidden_idx] = 0.0
                                    elif masked_output.dim() == 2:
                                        masked_output[batch_idx, hidden_idx] = 0.0
                            
                            elif 'dim1_idx' in neuron and 'dim2_idx' in neuron:
                                dim1_idx = neuron['dim1_idx']
                                dim2_idx = neuron['dim2_idx']
                                
                                if masked_output.dim() == 3:
                                    if dim2_idx < masked_output.shape[-1]:
                                        masked_output[:, :, dim2_idx] = 0.0
                                elif masked_output.dim() == 2:
                                    if (dim1_idx < masked_output.shape[0] and 
                                        dim2_idx < masked_output.shape[1]):
                                        masked_output[dim1_idx, dim2_idx] = 0.0
                            
                            elif 'flat_idx' in neuron:
                                flat_idx = neuron['flat_idx']
                                if masked_output.dim() >= 2:
                                    hidden_dim = flat_idx % masked_output.shape[-1]
                                    if masked_output.dim() == 3:
                                        masked_output[:, :, hidden_dim] = 0.0
                                    elif masked_output.dim() == 2:
                                        masked_output[:, hidden_dim] = 0.0
                        
                        except Exception as e:
                            self.logger.debug(f"Failed to mask neuron in {layer_name}: {e}")
                    
                    return masked_output
                return hook

            for name, module in self.model.named_modules():
                if name == layer_name:
                    if not list(module.children()) or isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                        hook = module.register_forward_hook(
                            create_layer_mask_hook(layer_name, layer_neurons)
                        )
                        self.masking_hooks.append(hook)
                        hook_count += 1
                        break
        
        self.current_masked_neurons = masked_neurons.copy()
        self.logger.info(f"Set up {hook_count} masking hooks for {len(neurons_by_layer)} layers")
        return hook_count
    
    def _clear_masking_hooks(self):
        """Clear all masking hooks"""
        for hook in self.masking_hooks:
            hook.remove()
        self.masking_hooks.clear()
        self.current_masked_neurons.clear()
    
    def apply_neuron_masking(self, masked_neurons):
        """Apply neuron masking by setting up forward hooks"""
        if not masked_neurons:
            self._clear_masking_hooks()
            return 0
        
        hook_count = self._setup_masking_hooks(masked_neurons)
        self.logger.info(f"Applied masking to {len(masked_neurons)} neurons using {hook_count} hooks")
        return len(masked_neurons)
    
    def generate_response(self, prompt, max_length=1024, max_new_tokens=50):
        """Generate response for a given prompt (optimized for short direct answers)"""
        try:
            first_param = next(self.model.parameters())
            input_device = first_param.device
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            input_ids = inputs.input_ids.to(input_device)
            attention_mask = inputs.attention_mask.to(input_device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,  
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

            response = response.strip()
            first_line = response.split('\n')[0].strip()
            return first_line if first_line else response
            
        except Exception as e:
            self.logger.warning(f"Error generating response: {e}")
            return ""
    
    def evaluate_single_sample(self, sample, max_length=1024, max_new_tokens=50, template_idx=0):
        """Evaluate a single SimpleQA sample with direct answer approach"""
        try:
            question = ""
            for field in ['problem', 'question', 'query', 'text', 'input']:
                if field in sample and sample[field]:
                    question = sample[field]
                    break
            
            if not question:
                self.logger.warning("No question found in sample")
                question = str(sample)

            prompt = self.create_direct_answer_prompt(question, template_idx)

            response = self.generate_response(prompt, max_length, max_new_tokens)

            is_meaningful = self.is_response_meaningful(response)
            
            if not is_meaningful:
                self.logger.debug("Meaningless response detected")
                return {
                    'question': question,
                    'prompt': prompt,
                    'response': response,
                    'is_meaningful': False,
                    'ground_truth': sample.get('answer', ''),
                    'evaluation': {
                        'contains_answer': False,
                        'exact_match': False,
                        'word_overlap_ratio': 0.0,
                        'confidence': 0.0
                    }
                }

            ground_truth = sample.get('answer', '')

            evaluation = self.evaluate_answer_match(response, ground_truth)
            
            return {
                'question': question,
                'prompt': prompt,
                'response': response,
                'is_meaningful': True,
                'ground_truth': ground_truth,
                'evaluation': evaluation
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating sample: {e}")
            return {
                'question': sample.get('problem', sample.get('question', '')),
                'prompt': '',
                'response': '',
                'is_meaningful': False,
                'ground_truth': sample.get('answer', ''),
                'evaluation': {
                    'contains_answer': False,
                    'exact_match': False,
                    'word_overlap_ratio': 0.0,
                    'confidence': 0.0
                },
                'error': str(e)
            }
    
    def evaluate_dataset(self, samples, max_length=1024, max_new_tokens=50, 
                        num_neurons_to_mask=0, template_idx=0):
        """Evaluate the entire SimpleQA dataset with direct answer approach"""
        self.logger.info(f"Evaluating {len(samples)} SimpleQA samples with direct answer approach")
        self.logger.info(f"Max length: {max_length}, Max new tokens: {max_new_tokens}")
        self.logger.info(f"Using prompt template {template_idx}: {self.direct_answer_prompts[template_idx % len(self.direct_answer_prompts)][:100]}...")
        
        if num_neurons_to_mask > 0:
            self.logger.info(f"Masking top {num_neurons_to_mask} neurons")
            neurons_to_mask = self.neurons[:num_neurons_to_mask]
            self.apply_neuron_masking(neurons_to_mask)
        else:
            self.logger.info("No neuron masking applied")
            self._clear_masking_hooks()
        
        results = []

        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            result = self.evaluate_single_sample(sample, max_length, max_new_tokens, template_idx)
            results.append(result)

            if i % 50 == 0:
                for gpu_id in range(self.num_gpus):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()

        meaningful_results = [r for r in results if r.get('is_meaningful', False)]
        total_samples = len(results)
        meaningful_samples = len(meaningful_results)
        
        self.logger.info(f"Meaningful responses: {meaningful_samples}/{total_samples} ({meaningful_samples/total_samples*100:.1f}%)")
        
        if meaningful_results:
            contains_answer_count = sum(1 for r in meaningful_results if r['evaluation']['contains_answer'])
            exact_match_count = sum(1 for r in meaningful_results if r['evaluation']['exact_match'])
            high_confidence_count = sum(1 for r in meaningful_results if r['evaluation']['confidence'] > 0.8)
            
            contains_answer_rate = contains_answer_count / meaningful_samples
            exact_match_rate = exact_match_count / meaningful_samples
            high_confidence_rate = high_confidence_count / meaningful_samples
            
            avg_word_overlap = np.mean([r['evaluation']['word_overlap_ratio'] for r in meaningful_results])
            avg_confidence = np.mean([r['evaluation']['confidence'] for r in meaningful_results])
            
            avg_response_length = np.mean([len(r['response'].split()) for r in meaningful_results])
            
        else:
            contains_answer_rate = 0.0
            exact_match_rate = 0.0
            high_confidence_rate = 0.0
            avg_word_overlap = 0.0
            avg_confidence = 0.0
            avg_response_length = 0.0
            contains_answer_count = 0
            exact_match_count = 0
            high_confidence_count = 0
        
        evaluation_results = {
            'meaningful_response_rate': meaningful_samples / total_samples if total_samples > 0 else 0.0,
            'contains_answer_rate': contains_answer_rate,
            'exact_match_rate': exact_match_rate,
            'high_confidence_rate': high_confidence_rate,
            'avg_word_overlap_ratio': avg_word_overlap,
            'avg_confidence': avg_confidence,
            'avg_response_length_words': avg_response_length,
            'contains_answer_count': contains_answer_count,
            'exact_match_count': exact_match_count,
            'high_confidence_count': high_confidence_count,
            'meaningful_samples': meaningful_samples,
            'total_samples': total_samples,
            'neurons_masked': num_neurons_to_mask,
            'template_used': template_idx,
            'detailed_results': results
        }
        
        self.logger.info(f"Meaningful Response Rate: {meaningful_samples/total_samples:.4f} ({meaningful_samples}/{total_samples})")
        if meaningful_samples > 0:
            self.logger.info(f"Contains Answer Rate: {contains_answer_rate:.4f} ({contains_answer_count}/{meaningful_samples})")
            self.logger.info(f"Exact Match Rate: {exact_match_rate:.4f} ({exact_match_count}/{meaningful_samples})")
            self.logger.info(f"High Confidence Rate (>0.8): {high_confidence_rate:.4f} ({high_confidence_count}/{meaningful_samples})")
            self.logger.info(f"Average Word Overlap Ratio: {avg_word_overlap:.4f}")
            self.logger.info(f"Average Confidence: {avg_confidence:.4f}")
            self.logger.info(f"Average Response Length: {avg_response_length:.1f} words")
        else:
            self.logger.warning("No meaningful responses generated!")
        
        return evaluation_results
    
    def test_multiple_masking_levels(self, samples, masking_levels, max_length=1024, max_new_tokens=50, 
                                   template_idx=0, output_file="simpleqa_results.json"):
        """Test QA accuracy with multiple neuron masking levels using direct answer approach"""
        self.logger.info("Testing multiple neuron masking levels for SimpleQA with direct answer approach")
        self.logger.info(f"Masking levels: {masking_levels}")
        
        all_results = {
            "model_info": {
                "model_path": getattr(self, 'model_path', 'unknown'),
                "num_gpus": self.num_gpus,
                "total_neurons": len(self.neurons)
            },
            "dataset_info": {
                "num_samples": len(samples),
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "template_used": template_idx,
                "prompt_template": self.direct_answer_prompts[template_idx % len(self.direct_answer_prompts)]
            },
            "results": []
        }
        
        for num_neurons in masking_levels:
            self.logger.info(f"\n=== Testing with {num_neurons} neurons masked ===")

            self._clear_masking_hooks()

            result = self.evaluate_dataset(
                samples=samples,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_neurons_to_mask=num_neurons,
                template_idx=template_idx
            )
            
            result["masking_level"] = num_neurons
            all_results["results"].append(result)
            
            self.logger.info(f"Masking {num_neurons} neurons:")
            self.logger.info(f"  Meaningful Response Rate = {result.get('meaningful_response_rate', 0):.4f}")
            self.logger.info(f"  Contains Answer Rate = {result.get('contains_answer_rate', 0):.4f}")
            self.logger.info(f"  Exact Match Rate = {result.get('exact_match_rate', 0):.4f}")
            self.logger.info(f"  Average Word Overlap = {result.get('avg_word_overlap_ratio', 0):.4f}")
            self.logger.info(f"  Average Confidence = {result.get('avg_confidence', 0):.4f}")

        self.logger.info(f"Saving all results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        self.logger.info("\n=== Results Summary ===")
        self.logger.info("Masked | Meaningful | Contains | ExactMatch | WordOverlap | Confidence")
        self.logger.info("-------|------------|----------|------------|-------------|------------")
        for result in all_results["results"]:
            masked = result.get("masking_level", 0)
            meaningful = result.get("meaningful_response_rate", 0)
            contains = result.get("contains_answer_rate", 0)
            exact = result.get("exact_match_rate", 0)
            overlap = result.get("avg_word_overlap_ratio", 0)
            confidence = result.get("avg_confidence", 0)
            self.logger.info(f"{masked:6d} | {meaningful:8.4f} | {contains:6.4f} | {exact:8.4f} | {overlap:9.4f} | {confidence:8.4f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="SimpleQA Direct Answer Evaluator with Neuron Masking")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--neuron_file", type=str, help="Path to neuron importance file (JSON)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "validation"], help="Dataset split")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum input sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum new tokens to generate (short for direct answers)")
    parser.add_argument("--num_neurons_to_mask", type=int, default=0, help="Number of top neurons to mask")
    parser.add_argument("--masking_levels", type=str, help="Comma-separated list of neuron counts to test (e.g., '0,10,50,100')")
    parser.add_argument("--template_idx", type=int, default=0, help="Index of prompt template to use (0-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="simpleqa_direct_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="simpleqa_direct_evaluation.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    parser.add_argument("--sample_size", type=int, help="Number of samples to evaluate (for testing)")
    
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== SimpleQA Direct Answer Evaluator with Neuron Masking ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Prompt template index: {args.template_idx}")
    logger.info("Using direct answer approach with word-level containment evaluation")

    evaluator = SimpleQAEvaluator(
        model_path=args.model_path,
        neuron_file=args.neuron_file,
        device=args.device,
        seed=args.seed,
        logger=logger
    )

    samples = evaluator.load_dataset(split=args.split)

    if args.sample_size and args.sample_size < len(samples):
        logger.info(f"Using sample of {args.sample_size} from {len(samples)} total samples")
        samples = random.sample(samples, args.sample_size)

    if args.masking_levels:
        masking_levels = [int(x.strip()) for x in args.masking_levels.split(',')]
        results = evaluator.test_multiple_masking_levels(
            samples=samples,
            masking_levels=masking_levels,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            template_idx=args.template_idx,
            output_file=args.output_file
        )
    else:
        result = evaluator.evaluate_dataset(
            samples=samples,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            num_neurons_to_mask=args.num_neurons_to_mask,
            template_idx=args.template_idx
        )
        
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    logger.info("=== Evaluation Completed ===")
    logger.info(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
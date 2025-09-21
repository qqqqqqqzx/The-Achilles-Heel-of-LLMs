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
from modelscope.msdatasets import MsDataset
import pandas as pd
import re
import sys
import os
from typing import Dict, List, Any, Union, Tuple
import transformers

def setup_logging(log_file="mgsm_evaluation.log", log_level=logging.INFO):
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

class MGSMEvaluator:
    def __init__(self, model_path, neuron_file=None, device="auto", seed=42, logger=None):
        """
        Initialize MGSM evaluator with neuron masking capability
        
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

        self._setup_math_evaluation()
        
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
    
    def load_dataset(self, split='test'):
        """
        Load MGSM dataset using ModelScope
        
        Args:
            split: Dataset split ('test', 'train', etc.)
            
        Returns:
            List of math problem samples
        """
        self.logger.info(f"Loading MGSM dataset from ModelScope, split: {split}")
        
        try:
            ds = MsDataset.load('sbintuitions/MGSM_en', split=split)
            self.logger.info(f"Dataset loaded successfully from ModelScope. Total items: {len(ds)}")

            samples = []
            for item in tqdm(ds, desc=f"Loading {split} data"):
                sample_dict = dict(item)
                samples.append(sample_dict)
            
            self.logger.info(f"Loaded {len(samples)} samples from MGSM {split}")

            if samples:
                sample = samples[0]
                self.logger.info(f"Sample structure: {list(sample.keys())}")
                if 'question' in sample:
                    self.logger.info(f"Sample question: {sample['question'][:200]}...")
                if 'answer' in sample:
                    self.logger.info(f"Sample answer: {sample['answer']}")
                if 'answer_number' in sample:
                    self.logger.info(f"Sample answer_number: {sample['answer_number']}")
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Failed to load MGSM dataset from ModelScope: {e}")
            raise
    
    def _setup_math_evaluation(self):
        """Setup math problem evaluation methods"""
        self.logger.info("Setting up math problem evaluation methods")
    
    def is_response_meaningful(self, response):
        """
        Check if response is meaningful, filter out meaningless characters
        
        Args:
            response: Model generated response
            
        Returns:
            bool: Whether response is meaningful
        """
        if not response or not response.strip():
            return False

        cleaned = response.strip()

        if len(cleaned) < 3:
            return False

        alphanumeric_chars = sum(1 for c in cleaned if c.isalnum())
        if alphanumeric_chars < 2:  
            return False

        unique_chars = len(set(cleaned.lower()))
        if len(cleaned) > 10 and unique_chars < 3:  
            return False

        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(cleaned)):
            if cleaned[i] == cleaned[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        if max_consecutive > 5:
            return False

        words = cleaned.split()
        if words:
            valid_words = [w for w in words if any(c.isalpha() for c in w)]
            if len(valid_words) == 0:
                return False
        
        return True
    
    def _extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        number_patterns = [
            r'-?\d+\.\d+',  
            r'-?\d+/\d+',   
            r'-?\d+',     
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    if '/' in match:
                        num, den = match.split('/')
                        numbers.append(float(num) / float(den))
                    else:
                        numbers.append(float(match))
                except ValueError:
                    continue
        
        return numbers
    
    def _extract_final_answer(self, response: str) -> Union[float, None]:
        """
        Extract the final numerical answer from model response
        
        Args:
            response: Model generated response
            
        Returns:
            Final numerical answer or None if not found
        """
        final_answer_patterns = [
            r'(?:the answer is|answer:|final answer:|therefore,?|so,?|thus,?)\s*([+-]?\d*\.?\d+)',
            r'(?:=|equals?)\s*([+-]?\d*\.?\d+)\s*(?:\.|$)',
            r'([+-]?\d*\.?\d+)\s*(?:\.|$)', 
        ]
        
        response_lower = response.lower().strip()

        for pattern in final_answer_patterns:
            matches = re.findall(pattern, response_lower, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])
                except ValueError:
                    continue
        
        all_numbers = self._extract_numbers_from_text(response)
        if all_numbers:
            return all_numbers[-1]
        
        return None
    
    def _normalize_answer(self, answer: Union[str, int, float]) -> Union[float, None]:
        """Normalize answer to comparable format"""
        if answer is None:
            return None
        
        if isinstance(answer, (int, float)):
            return float(answer)
        
        if isinstance(answer, str):
            numbers = self._extract_numbers_from_text(answer)
            return numbers[0] if numbers else None
        
        return None
    
    def _check_mathematical_reasoning(self, response: str, question: str) -> Dict:
        """
        Check if response shows mathematical reasoning
        
        Args:
            response: Model response
            question: Original math problem
            
        Returns:
            Dict with reasoning analysis
        """
        reasoning_indicators = [
            r'\+|\-|\*|/|×|÷',
            r'step \d+|first|second|third|then|next|finally',
            r'calculate|compute|solve|total|sum|difference|product|quotient',
            r'because|since|therefore|so|thus|hence'
        ]
        
        response_lower = response.lower()
        reasoning_score = 0.0
        found_indicators = []
        
        for pattern in reasoning_indicators:
            if re.search(pattern, response_lower):
                reasoning_score += 1.0
                found_indicators.append(pattern)

        max_score = len(reasoning_indicators)
        normalized_score = reasoning_score / max_score

        has_calculations = bool(re.search(r'\d+\s*[+\-*/×÷]\s*\d+', response))

        has_steps = bool(re.search(r'(step|first|then|next|finally)', response_lower))
        
        return {
            'has_reasoning': normalized_score > 0.2,
            'reasoning_score': normalized_score,
            'has_calculations': has_calculations,
            'has_steps': has_steps,
            'found_indicators': found_indicators
        }
    
    def _evaluate_math_accuracy(self, predicted_answer: Union[float, None], ground_truth: Union[str, int, float], 
                               tolerance: float = 1e-6) -> Dict:
        """
        Evaluate mathematical accuracy
        
        Args:
            predicted_answer: Extracted numerical answer from model
            ground_truth: Ground truth answer
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Dict with accuracy metrics
        """
        gt_answer = self._normalize_answer(ground_truth)
        
        metrics = {
            'exact_match': False,
            'numerical_match': False,
            'predicted_answer': predicted_answer,
            'ground_truth_answer': gt_answer,
            'has_prediction': predicted_answer is not None,
            'has_ground_truth': gt_answer is not None
        }
        
        if predicted_answer is None or gt_answer is None:
            return metrics

        if predicted_answer == gt_answer:
            metrics['exact_match'] = True
            metrics['numerical_match'] = True
        else:
            if abs(predicted_answer - gt_answer) <= tolerance:
                metrics['numerical_match'] = True
        
        return metrics
    
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
    
    def generate_response(self, prompt, max_length=2048, max_new_tokens=512):
        """Generate response for a given prompt"""
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
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            self.logger.warning(f"Error generating response: {e}")
            return ""
    
    def evaluate_single_sample(self, sample, max_length=2048, max_new_tokens=512):
        """Evaluate a single MGSM sample with meaningfulness checking"""
        try:
            question = sample.get('question', sample.get('problem', ''))
            if not question:
                question = sample.get('text', sample.get('input', ''))
            
            response = self.generate_response(question, max_length, max_new_tokens)
            
            is_meaningful = self.is_response_meaningful(response)
            
            if not is_meaningful:
                self.logger.debug(f"Meaningless response detected for sample")
                return {
                    'question': question,
                    'response': response,
                    'is_meaningful': False,
                    'predicted_answer': None,
                    'ground_truth': sample.get('answer', sample.get('answer_number', '')),
                    'evaluation': {
                        'exact_match': False,
                        'numerical_match': False,
                        'has_prediction': False,
                        'has_reasoning': False,
                        'reasoning_score': 0.0
                    }
                }

            predicted_answer = self._extract_final_answer(response)

            reasoning_analysis = self._check_mathematical_reasoning(response, question)

            ground_truth = sample.get('answer', sample.get('answer_number', ''))

            accuracy_metrics = self._evaluate_math_accuracy(predicted_answer, ground_truth)

            evaluation = {
                **accuracy_metrics,
                **reasoning_analysis
            }
            
            return {
                'question': question,
                'response': response,
                'is_meaningful': True,
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'evaluation': evaluation
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating sample: {e}")
            return {
                'question': sample.get('question', ''),
                'response': '',
                'is_meaningful': False,
                'predicted_answer': None,
                'ground_truth': sample.get('answer', ''),
                'evaluation': {
                    'exact_match': False,
                    'numerical_match': False,
                    'has_prediction': False,
                    'has_reasoning': False,
                    'reasoning_score': 0.0,
                    'error': str(e)
                },
                'error': str(e)
            }
    
    def evaluate_dataset(self, samples, max_length=2048, max_new_tokens=512, num_neurons_to_mask=0):
        """Evaluate the entire MGSM dataset with meaningfulness checking"""
        self.logger.info(f"Evaluating {len(samples)} MGSM samples")
        self.logger.info(f"Max length: {max_length}, Max new tokens: {max_new_tokens}")
        
        if num_neurons_to_mask > 0:
            self.logger.info(f"Masking top {num_neurons_to_mask} neurons")
            neurons_to_mask = self.neurons[:num_neurons_to_mask]
            self.apply_neuron_masking(neurons_to_mask)
        else:
            self.logger.info("No neuron masking applied")
            self._clear_masking_hooks()
        
        results = []

        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            result = self.evaluate_single_sample(sample, max_length, max_new_tokens)
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
            exact_match_count = sum(1 for r in meaningful_results if r['evaluation']['exact_match'])
            numerical_match_count = sum(1 for r in meaningful_results if r['evaluation']['numerical_match'])
            has_prediction_count = sum(1 for r in meaningful_results if r['evaluation']['has_prediction'])
            has_reasoning_count = sum(1 for r in meaningful_results if r['evaluation']['has_reasoning'])
            
            exact_match_accuracy = exact_match_count / meaningful_samples
            numerical_match_accuracy = numerical_match_count / meaningful_samples
            prediction_rate = has_prediction_count / meaningful_samples
            reasoning_rate = has_reasoning_count / meaningful_samples
            
            avg_reasoning_score = sum(r['evaluation']['reasoning_score'] for r in meaningful_results) / meaningful_samples
        else:
            exact_match_accuracy = 0.0
            numerical_match_accuracy = 0.0
            prediction_rate = 0.0
            reasoning_rate = 0.0
            avg_reasoning_score = 0.0
            exact_match_count = 0
            numerical_match_count = 0
            has_prediction_count = 0
            has_reasoning_count = 0
        
        evaluation_results = {
            'exact_match_accuracy': exact_match_accuracy,
            'numerical_match_accuracy': numerical_match_accuracy,
            'prediction_rate': prediction_rate,
            'reasoning_rate': reasoning_rate,
            'meaningful_response_rate': meaningful_samples / total_samples if total_samples > 0 else 0.0,
            'avg_reasoning_score': avg_reasoning_score,
            'exact_match_count': exact_match_count,
            'numerical_match_count': numerical_match_count,
            'has_prediction_count': has_prediction_count,
            'has_reasoning_count': has_reasoning_count,
            'meaningful_samples': meaningful_samples,
            'total_samples': total_samples,
            'neurons_masked': num_neurons_to_mask,
            'detailed_results': results
        }

        self.logger.info(f"Meaningful Response Rate: {meaningful_samples/total_samples:.4f} ({meaningful_samples}/{total_samples})")
        if meaningful_samples > 0:
            self.logger.info(f"Exact Match Accuracy: {exact_match_accuracy:.4f} ({exact_match_count}/{meaningful_samples})")
            self.logger.info(f"Numerical Match Accuracy: {numerical_match_accuracy:.4f} ({numerical_match_count}/{meaningful_samples})")
            self.logger.info(f"Prediction Rate: {prediction_rate:.4f} ({has_prediction_count}/{meaningful_samples})")
            self.logger.info(f"Reasoning Rate: {reasoning_rate:.4f} ({has_reasoning_count}/{meaningful_samples})")
            self.logger.info(f"Average Reasoning Score: {avg_reasoning_score:.4f}")
        else:
            self.logger.warning("No meaningful responses generated!")
        
        return evaluation_results
    
    def test_multiple_masking_levels(self, samples, masking_levels, max_length=2048, max_new_tokens=512, output_file="mgsm_results.json"):
        """Test math accuracy with multiple neuron masking levels"""
        self.logger.info("Testing multiple neuron masking levels for MGSM")
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
                "max_new_tokens": max_new_tokens
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
                num_neurons_to_mask=num_neurons
            )
            
            result["masking_level"] = num_neurons
            all_results["results"].append(result)
            
            self.logger.info(f"Masking {num_neurons} neurons:")
            self.logger.info(f"  Meaningful Response Rate = {result.get('meaningful_response_rate', 0):.4f}")
            self.logger.info(f"  Exact Match Accuracy = {result.get('exact_match_accuracy', 0):.4f}")
            self.logger.info(f"  Numerical Match Accuracy = {result.get('numerical_match_accuracy', 0):.4f}")
            self.logger.info(f"  Prediction Rate = {result.get('prediction_rate', 0):.4f}")
            self.logger.info(f"  Reasoning Rate = {result.get('reasoning_rate', 0):.4f}")
            self.logger.info(f"  Average Reasoning Score = {result.get('avg_reasoning_score', 0):.4f}")

        self.logger.info(f"Saving all results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        self.logger.info("\n=== Results Summary ===")
        self.logger.info("Masked | Meaningful | Exact Match | Numerical | Prediction | Reasoning | Avg Reasoning")
        self.logger.info("-------|------------|-------------|-----------|------------|-----------|---------------")
        for result in all_results["results"]:
            masked = result.get("masking_level", 0)
            meaningful = result.get("meaningful_response_rate", 0)
            exact_match = result.get("exact_match_accuracy", 0)
            numerical = result.get("numerical_match_accuracy", 0)
            prediction = result.get("prediction_rate", 0)
            reasoning = result.get("reasoning_rate", 0)
            avg_reasoning = result.get("avg_reasoning_score", 0)
            self.logger.info(f"{masked:6d} | {meaningful:8.4f} | {exact_match:9.4f} | {numerical:7.4f} | {prediction:8.4f} | {reasoning:7.4f} | {avg_reasoning:11.4f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="MGSM Dataset Evaluator with Neuron Masking and Meaningfulness Checking")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--neuron_file", type=str, help="Path to neuron importance file (JSON)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Dataset split")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum input sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--num_neurons_to_mask", type=int, default=0, help="Number of top neurons to mask")
    parser.add_argument("--masking_levels", type=str, help="Comma-separated list of neuron counts to test (e.g., '0,10,50,100')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="mgsm_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="mgsm_evaluation.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    parser.add_argument("--sample_size", type=int, help="Number of samples to evaluate (for testing)")
    
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== MGSM Dataset Evaluator with Neuron Masking and Meaningfulness Checking ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info("Using ModelScope for dataset loading")
    logger.info("Including meaningfulness checking for model outputs")

    evaluator = MGSMEvaluator(
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
            output_file=args.output_file
        )
    else:
        result = evaluator.evaluate_dataset(
            samples=samples,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            num_neurons_to_mask=args.num_neurons_to_mask
        )
        
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    logger.info("=== Evaluation Completed ===")
    logger.info(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()

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
import string
import subprocess
import tempfile

def setup_logging(log_file="humaneval_evaluation.log", log_level=logging.INFO):
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

class HumanEvalEvaluator:
    def __init__(self, model_path, neuron_file=None, device="auto", seed=42, logger=None):
        """
        Initialize HumanEval evaluator with neuron masking capability
        
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
        
    def _load_model_multi_gpu(self, model_path):
        """Load model with multi-GPU support"""
        self.logger.info(f"Loading model with multi-GPU: {model_path}")
        
        import transformers
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
        Load HumanEval dataset using ModelScope
        
        Args:
            split: Dataset split ('test')
            
        Returns:
            List of HumanEval samples
        """
        self.logger.info(f"Loading HumanEval dataset from ModelScope, split: {split}")
        
        try:
            ds = MsDataset.load('opencompass/humaneval', subset_name='openai_humaneval', split=split)
            self.logger.info(f"Dataset loaded successfully from ModelScope. Total items: {len(ds)}")
            
            samples = []
            for item in tqdm(ds, desc=f"Loading {split} data"):
                sample_dict = dict(item)
                samples.append(sample_dict)
            
            self.logger.info(f"Loaded {len(samples)} samples from HumanEval {split}")
            
            if samples:
                sample = samples[0]
                self.logger.info(f"Sample structure: {list(sample.keys())}")
                self.logger.info(f"Sample prompt: {sample.get('prompt', '')[:200]}...")
                self.logger.info(f"Sample test: {sample.get('test', '')[:200]}...")
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Failed to load HumanEval dataset from ModelScope: {e}")
            raise
    
    def _is_response_meaningful(self, response):
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
    
    def _extract_code_from_response(self, response, prompt):
        """
        Extract code from model response
        
        Args:
            response: Model generated response
            prompt: Original prompt
            
        Returns:
            str: Extracted code
        """
        if not response:
            return ""
        
        code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'```python(.*?)```',
            r'```(.*?)```'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if code and 'def ' in code:
                    return code

        lines = response.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if 'def ' in line and ':' in line:
                in_function = True
                code_lines.append(line)
            elif in_function:
                if line.strip() == '' or line.startswith('    ') or line.startswith('\t'):
                    code_lines.append(line)
                else:
                    break
        
        if code_lines:
            return '\n'.join(code_lines)

        if 'def ' in response:
            return response
        
        return ""
    
    def _execute_code_safely(self, code, test_code, timeout=5):
        """
        Safely execute code and run tests
        
        Args:
            code: Code to execute
            test_code: Test code
            timeout: Timeout in seconds
            
        Returns:
            tuple: (passed, error_message)
        """
        if not code or not test_code:
            return False, "Empty code or test"
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                full_code = f"{code}\n\n{test_code}"
                f.write(full_code)
                temp_file = f.name
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                if result.returncode == 0:
                    return True, "Success"
                else:
                    return False, f"Runtime error: {result.stderr}"
                    
            except subprocess.TimeoutExpired:
                return False, "Timeout"
            except Exception as e:
                return False, f"Execution error: {str(e)}"
            finally:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            return False, f"File creation error: {str(e)}"
    
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
        """Evaluate a single HumanEval sample"""
        try:
            prompt = sample['prompt']
            response = self.generate_response(prompt, max_length, max_new_tokens)
            
            is_meaningful = self._is_response_meaningful(response)
            
            code = self._extract_code_from_response(response, prompt)
            
            passed = False
            error_msg = "No code extracted"
            
            if code and is_meaningful:
                test_code = sample.get('test', '')
                if test_code:
                    passed, error_msg = self._execute_code_safely(code, test_code)
                else:
                    error_msg = "No test code provided"
            elif not is_meaningful:
                error_msg = "Response not meaningful"
            
            return {
                'task_id': sample.get('task_id', ''),
                'prompt': prompt,
                'response': response,
                'extracted_code': code,
                'is_meaningful_response': is_meaningful,
                'passed': passed,
                'error_msg': error_msg,
                'test_code': sample.get('test', '')
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating sample: {e}")
            return {
                'task_id': sample.get('task_id', ''),
                'prompt': sample.get('prompt', ''),
                'response': '',
                'extracted_code': '',
                'is_meaningful_response': False,
                'passed': False,
                'error_msg': str(e),
                'test_code': sample.get('test', '')
            }
    
    def evaluate_dataset(self, samples, max_length=2048, max_new_tokens=512, num_neurons_to_mask=0):
        """Evaluate the entire HumanEval dataset"""
        self.logger.info(f"Evaluating {len(samples)} HumanEval samples")
        self.logger.info(f"Max length: {max_length}, Max new tokens: {max_new_tokens}")
        
        if num_neurons_to_mask > 0:
            self.logger.info(f"Masking top {num_neurons_to_mask} neurons")
            neurons_to_mask = self.neurons[:num_neurons_to_mask]
            self.apply_neuron_masking(neurons_to_mask)
        else:
            self.logger.info("No neuron masking applied")
            self._clear_masking_hooks()
        
        results = []
        passed_count = 0
        meaningless_responses = 0
        code_extracted_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            result = self.evaluate_single_sample(sample, max_length, max_new_tokens)
            results.append(result)
            
            if result['passed']:
                passed_count += 1
            
            if not result['is_meaningful_response']:
                meaningless_responses += 1
            
            if result['extracted_code']:
                code_extracted_count += 1
            
            if i % 50 == 0:
                for gpu_id in range(self.num_gpus):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()

        total_samples = len(results)
        pass_rate = passed_count / total_samples if total_samples > 0 else 0.0
        meaningless_rate = meaningless_responses / total_samples if total_samples > 0 else 0.0
        code_extraction_rate = code_extracted_count / total_samples if total_samples > 0 else 0.0
        
        evaluation_results = {
            'pass_rate': pass_rate,
            'passed_count': passed_count,
            'total_samples': total_samples,
            'meaningless_responses': meaningless_responses,
            'meaningless_response_rate': meaningless_rate,
            'code_extracted_count': code_extracted_count,
            'code_extraction_rate': code_extraction_rate,
            'neurons_masked': num_neurons_to_mask,
            'detailed_results': results
        }

        self.logger.info(f"Pass Rate: {pass_rate:.4f} ({passed_count}/{total_samples})")
        self.logger.info(f"Code Extraction Rate: {code_extraction_rate:.4f} ({code_extracted_count}/{total_samples})")
        self.logger.info(f"Meaningless Responses: {meaningless_responses}/{total_samples} ({meaningless_rate*100:.1f}%)")
        
        return evaluation_results
    
    def test_multiple_masking_levels(self, samples, masking_levels, max_length=2048, max_new_tokens=512, output_file="humaneval_results.json"):
        """Test HumanEval accuracy with multiple neuron masking levels"""
        self.logger.info("Testing multiple neuron masking levels for HumanEval")
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
            self.logger.info(f"  Pass Rate = {result.get('pass_rate', 0):.4f}")
            self.logger.info(f"  Code Extraction Rate = {result.get('code_extraction_rate', 0):.4f}")
            self.logger.info(f"  Meaningless Response Rate = {result.get('meaningless_response_rate', 0):.4f}")

        self.logger.info(f"Saving all results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        self.logger.info("\n=== Results Summary ===")
        self.logger.info("Masked | Pass Rate | Code Extract | Meaningless%")
        self.logger.info("-------|-----------|--------------|-------------")
        for result in all_results["results"]:
            masked = result.get("masking_level", 0)
            pass_rate = result.get("pass_rate", 0)
            extract_rate = result.get("code_extraction_rate", 0)
            meaningless = result.get("meaningless_response_rate", 0) * 100
            self.logger.info(f"{masked:6d} | {pass_rate:7.4f} | {extract_rate:10.4f} | {meaningless:9.1f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="HumanEval Dataset Evaluator with Neuron Masking")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--neuron_file", type=str, help="Path to neuron importance file (JSON)")
    parser.add_argument("--split", type=str, default="test", choices=["test"], help="Dataset split")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum input sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--num_neurons_to_mask", type=int, default=0, help="Number of top neurons to mask")
    parser.add_argument("--masking_levels", type=str, help="Comma-separated list of neuron counts to test (e.g., '0,10,50,100')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="humaneval_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="humaneval_evaluation.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    parser.add_argument("--sample_size", type=int, help="Number of samples to evaluate (for testing)")
    
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== HumanEval Dataset Evaluator with Neuron Masking ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info("Using ModelScope for dataset loading")

    evaluator = HumanEvalEvaluator(
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
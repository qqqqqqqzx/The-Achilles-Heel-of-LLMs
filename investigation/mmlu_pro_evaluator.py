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
import pyarrow.parquet as pq
import re
import transformers

def setup_logging(log_file="mmlu_pro_evaluation.log", log_level=logging.INFO):
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

class MMLUProEvaluator:
    def __init__(self, model_path, neuron_file=None, device="auto", seed=42, logger=None):
        """
        Initialize MMLU-Pro evaluator
        
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
        """Load model"""
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
    
    def load_dataset(self, split='test', categories=None):
        """
        Load MMLU-Pro dataset
        
        Args:
            split: Dataset split ('test', 'validation')
            categories: List of categories to include (None for all)
            
        Returns:
            List of question samples
        """
        return self._load_modelscope_dataset(split, categories)
    
    def _load_modelscope_dataset(self, split, categories):
        """Load MMLU-Pro"""
        self.logger.info(f"Loading MMLU-Pro dataset from ModelScope, split: {split}")
        
        try:
            ds = MsDataset.load('modelscope/MMLU-Pro', subset_name='default', split=split)
            self.logger.info(f"Dataset loaded successfully from ModelScope. Total items: {len(ds)}")
            
            questions = []
            for item in tqdm(ds, desc=f"Loading {split} data"):
                question_dict = dict(item)
                
                if categories:
                    item_category = question_dict.get('category', '').lower()
                    if item_category not in [c.lower() for c in categories]:
                        continue
                
                questions.append(question_dict)
            
            self.logger.info(f"Loaded {len(questions)} questions from MMLU-Pro {split}")
            
            if questions:
                sample = questions[0]
                self.logger.info(f"Sample question structure: {list(sample.keys())}")
                self.logger.info(f"Sample question: {sample.get('question', '')[:100]}...")
                self.logger.info(f"Sample options: {sample.get('options', [])}")
                self.logger.info(f"Sample answer: {sample.get('answer', 'N/A')}")
                self.logger.info(f"Sample category: {sample.get('category', 'N/A')}")
            
            return questions
            
        except Exception as e:
            self.logger.error(f"Failed to load MMLU-Pro dataset from ModelScope: {e}")
            raise
    
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
    
    def format_question(self, question_data):
        """Format a question for the model (following Claude 3 evaluation format)"""
        question = question_data.get('question', '')
        if isinstance(question, (list, tuple)) and len(question) > 0:
            question = question[0]
        elif hasattr(question, 'item'):
            question = question.item()
        question = str(question)
        
        options = []
        if 'options' in question_data:
            options = question_data['options']
        elif 'choices' in question_data:
            options = question_data['choices']
        
        if hasattr(options, 'tolist'):
            options = options.tolist()
        elif not isinstance(options, (list, tuple)):
            options = [options] if options else []
        
        if not options:
            for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                if letter in question_data:
                    value = question_data[letter]
                    if hasattr(value, 'item'):
                        value = value.item()
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        value = value[0]
                    options.append(str(value))
        
        options = [str(opt) for opt in options]
        
        option_text = "Options are:\n"
        opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for i, option in enumerate(options):
            if i < len(opts):
                option_text += f"({opts[i]}): {option}\n"
        
        prompt = f"""Q: {question}
{option_text.strip()}

Please provide your answer as "The answer is (X)" where X is the correct option letter."""
        
        return prompt
    
    def extract_answer(self, response, num_options):
        """Extract answer choice from model response (严格模式：只接受 'the answer is' 格式)"""
        response = response.strip().upper()
        
        pattern = r"THE\s+ANSWER\s+IS\s+\(?([ABCDEFGHIJ])\)?"
        matches = re.findall(pattern, response)
        
        if matches:
            answer_letter = matches[0]
            answer_idx = ord(answer_letter) - ord('A')
            if 0 <= answer_idx < num_options:
                return answer_idx
        
        self.logger.debug(f"No valid 'the answer is' pattern found in response: {response[:100]}...")
        return -1
    
    def evaluate_single_question(self, question_data, max_length=2048):
        """Evaluate a single question"""
        try:
            prompt = self.format_question(question_data)
            
            first_param = next(self.model.parameters())
            input_device = first_param.device
            
            full_prompt = """You are a knowledge expert. You are supposed to answer the multi-choice question to derive your final answer as "The answer is (X)" where X is the correct option letter.

""" + prompt
            
            inputs = self.tokenizer(
                full_prompt,
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
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            options = question_data.get('options', question_data.get('choices', []))
            if hasattr(options, 'tolist'):
                options = options.tolist()
            elif not isinstance(options, (list, tuple)):
                options = []
            
            num_options = len(options)
            if num_options == 0:
                for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                    if letter in question_data:
                        num_options += 1
            
            predicted_idx = self.extract_answer(response, num_options)
            
            correct_idx = -1
            correct_answer = question_data.get('answer', None)
            
            if correct_answer is not None:
                if hasattr(correct_answer, 'item'):
                    correct_answer = correct_answer.item()
                elif isinstance(correct_answer, (list, tuple)) and len(correct_answer) > 0:
                    correct_answer = correct_answer[0]
                
                if isinstance(correct_answer, str) and len(correct_answer) == 1 and correct_answer.isalpha():
                    correct_idx = ord(correct_answer.upper()) - ord('A')
                elif isinstance(correct_answer, (int, float)):
                    correct_idx = int(correct_answer)
            
            if correct_idx == -1:
                for field in ['answer_index', 'correct_answer', 'label']:
                    if field in question_data:
                        answer = question_data[field]
                        if hasattr(answer, 'item'):
                            answer = answer.item()
                        elif isinstance(answer, (list, tuple)) and len(answer) > 0:
                            answer = answer[0]
                        
                        if isinstance(answer, str) and answer.isalpha():
                            correct_idx = ord(answer.upper()) - ord('A')
                            break
                        elif isinstance(answer, (int, float)):
                            correct_idx = int(answer)
                            break
            
            if isinstance(correct_idx, (float, np.floating)):
                correct_idx = int(correct_idx)
            
            question_id = question_data.get('question_id', '')
            if hasattr(question_id, 'item'):
                question_id = question_id.item()
            elif isinstance(question_id, (list, tuple)) and len(question_id) > 0:
                question_id = question_id[0]
            
            category = question_data.get('category', 'unknown')
            if hasattr(category, 'item'):
                category = category.item()
            elif isinstance(category, (list, tuple)) and len(category) > 0:
                category = category[0]
            
            is_correct = (predicted_idx == correct_idx and predicted_idx != -1)
            
            return {
                'question_id': str(question_id),
                'category': str(category),
                'predicted_answer': predicted_idx,
                'correct_answer': correct_idx,
                'is_correct': is_correct,
                'response': response.strip(),
                'prompt': full_prompt
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating question: {e}")
            
            question_id = question_data.get('question_id', '')
            if hasattr(question_id, 'item'):
                question_id = question_id.item()
            
            category = question_data.get('category', 'unknown')
            if hasattr(category, 'item'):
                category = category.item()
            
            answer_index = question_data.get('answer_index', -1)
            if hasattr(answer_index, 'item'):
                answer_index = answer_index.item()
            
            return {
                'question_id': str(question_id),
                'category': str(category),
                'predicted_answer': -1,
                'correct_answer': answer_index,
                'is_correct': False,
                'response': '',
                'error': str(e)
            }
    
    def evaluate_dataset(self, questions, max_length=2048, num_neurons_to_mask=0):
        """Evaluate the entire dataset"""
        self.logger.info(f"Evaluating {len(questions)} questions")
        self.logger.info(f"Max length: {max_length}")
        
        if num_neurons_to_mask > 0:
            self.logger.info(f"Masking top {num_neurons_to_mask} neurons")
            neurons_to_mask = self.neurons[:num_neurons_to_mask]
            self.apply_neuron_masking(neurons_to_mask)
        else:
            self.logger.info("No neuron masking applied")
            self._clear_masking_hooks()
        
        results = []
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for i, question in enumerate(tqdm(questions, desc="Evaluating questions")):
            result = self.evaluate_single_question(question, max_length)
            results.append(result)
            
            category = result['category']
            category_stats[category]['total'] += 1
            if result['is_correct']:
                category_stats[category]['correct'] += 1
            
            if i % 50 == 0:
                for gpu_id in range(self.num_gpus):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
        
        total_correct = sum(1 for r in results if r['is_correct'])
        total_questions = len(results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        category_accuracies = {}
        for category, stats in category_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            category_accuracies[category] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        evaluation_results = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'neurons_masked': num_neurons_to_mask,
            'category_accuracies': category_accuracies,
            'detailed_results': results
        }
        
        self.logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_questions})")
        for category, stats in category_accuracies.items():
            self.logger.info(f"{category}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        
        return evaluation_results
    
    def test_multiple_masking_levels(self, questions, masking_levels, max_length=2048, output_file="mmlu_pro_results.json"):
        """Test accuracy with multiple neuron masking levels"""
        self.logger.info("Testing multiple neuron masking levels")
        self.logger.info(f"Masking levels: {masking_levels}")
        
        all_results = {
            "model_info": {
                "model_path": getattr(self, 'model_path', 'unknown'),
                "num_gpus": self.num_gpus,
                "total_neurons": len(self.neurons)
            },
            "dataset_info": {
                "num_questions": len(questions),
                "max_length": max_length,
                "categories": list(set(q.get('category', 'unknown') for q in questions))
            },
            "results": []
        }
        
        for num_neurons in masking_levels:
            self.logger.info(f"\n=== Testing with {num_neurons} neurons masked ===")
            
            self._clear_masking_hooks()
            
            result = self.evaluate_dataset(
                questions=questions,
                max_length=max_length,
                num_neurons_to_mask=num_neurons
            )
            
            result["masking_level"] = num_neurons
            all_results["results"].append(result)
            
            self.logger.info(f"Masking {num_neurons} neurons: Accuracy = {result.get('overall_accuracy', 0):.4f}")
        
        self.logger.info(f"Saving all results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.info("\n=== Results Summary ===")
        for result in all_results["results"]:
            masked = result.get("masking_level", 0)
            acc = result.get("overall_accuracy", 0)
            self.logger.info(f"Masked {masked:5d} neurons: Accuracy = {acc:8.4f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="MMLU-Pro Dataset Evaluator with Neuron Masking (ModelScope Only) - Strict Answer Extraction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--neuron_file", type=str, help="Path to neuron importance file (JSON)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"], help="Dataset split")
    parser.add_argument("--categories", type=str, help="Comma-separated list of categories to include")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_neurons_to_mask", type=int, default=0, help="Number of top neurons to mask")
    parser.add_argument("--masking_levels", type=str, help="Comma-separated list of neuron counts to test (e.g., '0,10,50,100')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="mmlu_pro_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="mmlu_pro_evaluation.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== MMLU-Pro Dataset Evaluator (ModelScope Only) - Strict Answer Extraction ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Categories: {args.categories}")
    logger.info(f"Max length: {args.max_length}")
    logger.info("Using ModelScope for dataset loading")
    logger.info("Testing on FULL dataset (no sample limit)")
    logger.info("STRICT MODE: Only accepts 'The answer is (X)' format")
    
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(',')]
    
    evaluator = MMLUProEvaluator(
        model_path=args.model_path,
        neuron_file=args.neuron_file,
        device=args.device,
        seed=args.seed,
        logger=logger
    )
    
    questions = evaluator.load_dataset(
        split=args.split,
        categories=categories
    )
    
    if args.masking_levels:
        masking_levels = [int(x.strip()) for x in args.masking_levels.split(',')]
        results = evaluator.test_multiple_masking_levels(
            questions=questions,
            masking_levels=masking_levels,
            max_length=args.max_length,
            output_file=args.output_file
        )
    else:
        result = evaluator.evaluate_dataset(
            questions=questions,
            max_length=args.max_length,
            num_neurons_to_mask=args.num_neurons_to_mask
        )
        
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    logger.info("=== Evaluation Completed ===")
    logger.info(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()

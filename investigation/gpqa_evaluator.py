import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import argparse
import logging
from tqdm import tqdm
import random
from collections import defaultdict
from modelscope.msdatasets import MsDataset
import re
import hashlib
import transformers

def setup_logging(log_file="gpqa_evaluation.log", log_level=logging.INFO):
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

class GPQAEvaluator:
    def __init__(self, model_path, neuron_file=None, device="auto", seed=42, logger=None):
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
        self.logger.info("Model device allocation:")
        if hasattr(self.model, 'hf_device_map'):
            device_summary = defaultdict(int)
            for layer_name, device in self.model.hf_device_map.items():
                device_summary[str(device)] += 1
            for device, count in device_summary.items():
                self.logger.info(f"  Device {device}: {count} layers")
        else:
            self.logger.info("  Device map not available")
    
    def load_dataset(self, split='train'):
        self.logger.info(f"Loading GPQA-Diamond dataset from ModelScope, split: {split}")
        
        try:
            ds = MsDataset.load('AI-ModelScope/gpqa_diamond', subset_name='default', split=split)
            self.logger.info(f"Dataset loaded successfully from ModelScope. Total items: {len(ds)}")
            
            questions = []
            for item in tqdm(ds, desc=f"Loading {split} data"):
                question_dict = dict(item)
                questions.append(question_dict)
            
            self.logger.info(f"Loaded {len(questions)} questions from GPQA-Diamond {split}")
            
            if questions:
                sample = questions[0]
                self.logger.info(f"Sample structure: {list(sample.keys())}")
                self.logger.info(f"Sample question: {sample.get('Question', '')[:200]}...")
                self.logger.info(f"Sample subject: {sample.get('High-level domain', 'N/A')}")
                self.logger.info(f"Sample subdomain: {sample.get('Subdomain', 'N/A')}")
                self.logger.info(f"Sample difficulty: {sample.get("Writer's Difficulty Estimate", 'N/A')}")
                self.logger.info(f"Sample correct answer: {sample.get('Correct Answer', 'N/A')}")
                self.logger.info(f"Sample incorrect answers: {[sample.get(f'Incorrect Answer {i}', '') for i in [1,2,3]]}")
            
            return questions
            
        except Exception as e:
            self.logger.error(f"Failed to load GPQA-Diamond dataset from ModelScope: {e}")
            raise
    
    def _setup_masking_hooks(self, masked_neurons):
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
        for hook in self.masking_hooks:
            hook.remove()
        self.masking_hooks.clear()
        self.current_masked_neurons.clear()
    
    def apply_neuron_masking(self, masked_neurons):
        if not masked_neurons:
            self._clear_masking_hooks()
            return 0
        
        hook_count = self._setup_masking_hooks(masked_neurons)
        self.logger.info(f"Applied masking to {len(masked_neurons)} neurons using {hook_count} hooks")
        return len(masked_neurons)
    
    def format_question(self, question_data):
        question = question_data.get('Question', '')
        
        correct_answer = question_data.get('Correct Answer', '').strip()
        incorrect_1 = question_data.get('Incorrect Answer 1', '').strip()
        incorrect_2 = question_data.get('Incorrect Answer 2', '').strip()
        incorrect_3 = question_data.get('Incorrect Answer 3', '').strip()
        
        all_choices = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
        all_choices = [choice for choice in all_choices if choice]
        
        if len(all_choices) < 2:
            self.logger.warning(f"Insufficient choices: {all_choices}")
            return "", [], 0
        
        question_id = question_data.get('Record ID', f'default_{hash(question)}')
        seed = int(hashlib.md5(question_id.encode()).hexdigest()[:8], 16) % (2**32)
        random.seed(seed)
        
        shuffled_choices = all_choices.copy()
        random.shuffle(shuffled_choices)
        
        correct_position = shuffled_choices.index(correct_answer)
        
        option_text = "Options:\n"
        opts = ['A', 'B', 'C', 'D']
        for i, choice in enumerate(shuffled_choices[:4]):
            option_text += f"({opts[i]}) {choice}\n"
        
        prompt = f"""Question: {question}

{option_text.strip()}

Please provide your answer as "The answer is (X)" where X is the correct option letter."""
        
        return prompt, shuffled_choices, correct_position
    
    def extract_answer(self, response, num_options=4):
        response = response.strip().upper()
        
        pattern = r"ANSWER IS \(?([ABCD])\)?"
        matches = re.findall(pattern, response)
        
        if matches:
            return ord(matches[0]) - ord('A')
        
        available_letters = [chr(65 + i) for i in range(min(num_options, 4))]
        pattern = r'\b([' + ''.join(available_letters) + r'])\b'
        matches = re.findall(pattern, response)
        
        if matches:
            return ord(matches[0]) - ord('A')
        
        for i in range(min(num_options, 4)):
            if str(i + 1) in response:
                return i
        
        return -1
    
    def evaluate_single_question(self, question_data, max_length=2048):
        try:
            result = self.format_question(question_data)
            if len(result) != 3:
                self.logger.warning("Format question returned unexpected result")
                return self._create_error_result(question_data, "Format error")
                
            prompt, choices, correct_idx = result
                
            if not choices:
                self.logger.warning("No valid choices found for question")
                return self._create_error_result(question_data, "No valid choices")
            
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
            
            question_id = question_data.get('Record ID', '')
                
            if self._debug_count <= 10:
                self.logger.info(f"\n=== QUESTION {self._debug_count} DEBUG ===")
                self.logger.info(f"Question ID: {question_id}")
                self.logger.info(f"Question: {question_data.get('Question', '')[:100]}...")
                self.logger.info(f"Raw Correct Answer: '{question_data.get('Correct Answer', '')}'")
                self.logger.info(f"Raw Incorrect 1: '{question_data.get('Incorrect Answer 1', '')}'")
                self.logger.info(f"Raw Incorrect 2: '{question_data.get('Incorrect Answer 2', '')}'") 
                self.logger.info(f"Raw Incorrect 3: '{question_data.get('Incorrect Answer 3', '')}'")
                self.logger.info(f"Shuffled Choices: {choices}")
                self.logger.info(f"Correct position after shuffle: {correct_idx} (option {chr(65+correct_idx)})")
            
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
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            predicted_idx = self.extract_answer(response, len(choices))
            
            if predicted_idx == -1:
                is_correct = False
                predicted_choice = 'EXTRACTION_FAILED'
            else:
                is_correct = (predicted_idx == correct_idx)
                if 0 <= predicted_idx < len(choices):
                    predicted_choice = choices[predicted_idx]
                else:
                    predicted_choice = 'INVALID_INDEX'
                    is_correct = False
            
            if self._debug_count <= 10:
                self.logger.info(f"Model response: '{response}'")
                if predicted_idx == -1:
                    self.logger.info(f"Answer extraction FAILED - no valid pattern found")
                    self.logger.info(f"Predicted index: -1 (EXTRACTION_FAILED)")
                    self.logger.info(f"Predicted choice: EXTRACTION_FAILED")
                else:
                    self.logger.info(f"Predicted index: {predicted_idx} (option {chr(65+predicted_idx) if 0 <= predicted_idx <= 3 else 'Invalid'})")
                    self.logger.info(f"Predicted choice: '{predicted_choice}'")
                self.logger.info(f"Correct choice: '{choices[correct_idx] if 0 <= correct_idx < len(choices) else 'N/A'}'")
                self.logger.info(f"Is correct: {is_correct}")
                self.logger.info(f"========================")
            
            return {
                'question_id': question_id,
                'subject': question_data.get('High-level domain', 'unknown'),
                'subdomain': question_data.get('Subdomain', 'unknown'),
                'difficulty': question_data.get("Writer's Difficulty Estimate", 'unknown'),
                'predicted_answer': predicted_idx,
                'correct_answer': correct_idx,
                'predicted_choice': predicted_choice,
                'correct_choice': choices[correct_idx] if 0 <= correct_idx < len(choices) else 'N/A',
                'all_choices': choices,
                'is_correct': is_correct,
                'response': response.strip(),
                'prompt': prompt
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating question: {e}")
            import traceback
            self.logger.warning(f"Traceback: {traceback.format_exc()}")
            return self._create_error_result(question_data, str(e))
    
    def _create_error_result(self, question_data, error_msg):
        return {
            'question_id': question_data.get('Record ID', ''),
            'subject': question_data.get('High-level domain', 'unknown'),
            'subdomain': question_data.get('Subdomain', 'unknown'),
            'difficulty': question_data.get("Writer's Difficulty Estimate", 'unknown'),
            'predicted_answer': -1,
            'correct_answer': 0,
            'predicted_choice': 'Error',
            'correct_choice': 'Error',
            'all_choices': [],
            'is_correct': False,
            'response': '',
            'error': error_msg
        }
    
    def evaluate_dataset(self, questions, max_length=2048, num_neurons_to_mask=0):
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
        subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        difficulty_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        self._debug_count = 0
        
        for i, question in enumerate(tqdm(questions, desc="Evaluating questions")):
            result = self.evaluate_single_question(question, max_length)
            results.append(result)
            
            subject = result['subject']
            difficulty = result['difficulty']
            
            subject_stats[subject]['total'] += 1
            difficulty_stats[difficulty]['total'] += 1
            
            if result['is_correct']:
                subject_stats[subject]['correct'] += 1
                difficulty_stats[difficulty]['correct'] += 1
            
            if i % 50 == 0:
                for gpu_id in range(self.num_gpus):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
        
        total_correct = sum(1 for r in results if r['is_correct'])
        total_questions = len(results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        subject_accuracies = {}
        for subject, stats in subject_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            subject_accuracies[subject] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        difficulty_accuracies = {}
        for difficulty, stats in difficulty_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            difficulty_accuracies[difficulty] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        evaluation_results = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'neurons_masked': num_neurons_to_mask,
            'subject_accuracies': subject_accuracies,
            'difficulty_accuracies': difficulty_accuracies,
            'detailed_results': results
        }
        
        self.logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_questions})")
        for subject, stats in subject_accuracies.items():
            self.logger.info(f"{subject}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        
        return evaluation_results
    
    def test_multiple_masking_levels(self, questions, masking_levels, max_length=2048, output_file="gpqa_results.json"):
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
                "subjects": list(set(q.get('High-level domain', 'unknown') for q in questions))
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
    parser = argparse.ArgumentParser(description="GPQA-Diamond Dataset Evaluator with Neuron Masking")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--neuron_file", type=str, help="Path to neuron importance file (JSON)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_neurons_to_mask", type=int, default=0, help="Number of top neurons to mask")
    parser.add_argument("--masking_levels", type=str, help="Comma-separated list of neuron counts to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="gpqa_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="gpqa_evaluation.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    parser.add_argument("--sample_size", type=int, help="Number of questions to evaluate")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== GPQA-Diamond Dataset Evaluator (CLEAN VERSION) ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max length: {args.max_length}")
    
    evaluator = GPQAEvaluator(
        model_path=args.model_path,
        neuron_file=args.neuron_file,
        device=args.device,
        seed=args.seed,
        logger=logger
    )
    
    questions = evaluator.load_dataset(split=args.split)
    
    if args.sample_size and args.sample_size < len(questions):
        logger.info(f"Using sample of {args.sample_size} from {len(questions)} total questions")
        questions = random.sample(questions, args.sample_size)
    
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

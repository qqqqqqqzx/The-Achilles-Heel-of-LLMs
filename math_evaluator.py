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
from typing import Dict, List, Any, Optional
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
import transformers


def setup_logging(log_file="math500_evaluation.log", log_level=logging.INFO):
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

class MathAnswerEvaluator:
    """Professional math answer evaluator using sympy and LaTeX parsing"""
    
    BAD_SUBSTRINGS = ["^{", "^("]
    BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
    TUPLE_CHARS = "()[]"
    
    @staticmethod
    def normalize_answer_basic(answer: Optional[str]) -> Optional[str]:
        """Basic normalization from Hendrycks' MATH release"""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
            if m is not None:
                answer = m.group("text").strip()
            return MathAnswerEvaluator._strip_string(answer)
        except:
            return answer
    
    @staticmethod
    def _fix_fracs(string):
        """Fix fraction formatting"""
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        return new_str
    
    @staticmethod
    def _fix_a_slash_b(string):
        """Convert a/b to \\frac{a}{b}"""
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string
    
    @staticmethod
    def _remove_right_units(string):
        """Remove units from the right side"""
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string
    
    @staticmethod
    def _fix_sqrt(string):
        """Fix sqrt formatting"""
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    
    @staticmethod
    def _strip_string(string):
        """Comprehensive string cleaning"""
        string = string.replace("\n", "")
        
        string = string.replace("\\!", "")
        
        string = string.replace("\\\\", "\\")
        
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        
        string = string.replace("\\$", "")
        
        string = MathAnswerEvaluator._remove_right_units(string)
        
        string = string.replace("\\%", "")
        string = string.replace("\\%", "")
        
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string
        
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        string = MathAnswerEvaluator._fix_sqrt(string)

        string = string.replace(" ", "")
 
        string = MathAnswerEvaluator._fix_fracs(string)
        
        if string == "0.5":
            string = "\\frac{1}{2}"

        string = MathAnswerEvaluator._fix_a_slash_b(string)
        
        return string
    
    @staticmethod
    def _sympy_parse(expr: str):
        """Parse expression with sympy"""
        py_expr = expr.replace("^", "**")
        return sympy_parser.parse_expr(
            py_expr,
            transformations=(
                sympy_parser.standard_transformations
                + (sympy_parser.implicit_multiplication_application,)
            ),
        )
    
    @staticmethod
    def _parse_latex(expr: str) -> str:
        """Parse LaTeX to sympy-readable expression"""
        expr = expr.replace("\\tfrac", "\\frac")
        expr = expr.replace("\\dfrac", "\\frac")
        expr = expr.replace("\\frac", " \\frac")  
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)

        expr = expr.replace("√", "sqrt")
        expr = expr.replace("π", "pi")
        expr = expr.replace("∞", "inf")
        expr = expr.replace("∪", "U")
        expr = expr.replace("·", "*")
        expr = expr.replace("×", "*")
        
        return expr.strip()
    
    @staticmethod
    def _is_float(num: str) -> bool:
        try:
            float(num)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def _is_int(x: float) -> bool:
        try:
            return abs(x - int(round(x))) <= 1e-7
        except:
            return False
    
    @staticmethod
    def _is_frac(expr: str) -> bool:
        return bool(re.search(r"^-?[0-9]+\.?/0*[1-9][0-9]*\.?$", expr))
    
    @staticmethod
    def _str_is_int(x: str) -> bool:
        try:
            x = MathAnswerEvaluator._strip_properly_formatted_commas(x)
            x = float(x)
            return abs(x - int(round(x))) <= 1e-7
        except:
            return False
    
    @staticmethod
    def _str_to_int(x: str) -> int:
        x = x.replace(",", "")
        x = float(x)
        return int(x)
    
    @staticmethod
    def _inject_implicit_mixed_number(step: str):
        """Convert mixed numbers: 7 3/4 => 7+3/4"""
        p1 = re.compile(r"([0-9]) +([0-9])")
        step = p1.sub(r"\1+\2", step)
        return step
    
    @staticmethod
    def _strip_properly_formatted_commas(expr: str):
        """Remove commas from large numbers while preserving tuple commas"""
        p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
        while True:
            next_expr = p1.sub(r"\1\3\4", expr)
            if next_expr == expr:
                break
            expr = next_expr
        return next_expr
    
    @staticmethod
    def _normalize_advanced(expr: str) -> str:
        """Advanced normalization with LaTeX parsing"""
        if expr is None:
            return None

        m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
        if m is not None:
            expr = m.group("text")

        expr = expr.replace("\\%", "%")
        expr = expr.replace("\\$", "$")
        expr = expr.replace("$", "")
        expr = expr.replace("%", "")
        expr = expr.replace(" or ", " , ")
        expr = expr.replace(" and ", " , ")

        expr = expr.replace("million", "*10^6")
        expr = expr.replace("billion", "*10^9")
        expr = expr.replace("trillion", "*10^12")

        for unit in [
            "degree", "cm", "centimeter", "meter", "mile", "second",
            "minute", "hour", "day", "week", "month", "year",
            "foot", "feet", "inch", "yard"
        ]:
            expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
        expr = re.sub(r"\^ *\\circ", "", expr)

        if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
            expr = expr[1:-1]
        
        expr = re.sub(r",\\! *", "", expr)
        if MathAnswerEvaluator._is_float(expr) and MathAnswerEvaluator._is_int(float(expr)):
            expr = str(int(round(float(expr))))

        if "\\" in expr:
            try:
                expr = MathAnswerEvaluator._parse_latex(expr)
            except:
                pass

        expr = re.sub(r"- *", "-", expr)
        
        expr = MathAnswerEvaluator._inject_implicit_mixed_number(expr)
        expr = expr.replace(" ", "")
        
        expr = expr.replace("{", "")
        expr = expr.replace("}", "")
        
        expr = expr.lower()

        if MathAnswerEvaluator._str_is_int(expr):
            expr = str(MathAnswerEvaluator._str_to_int(expr))
        
        return expr
    
    @staticmethod
    def count_unknown_letters_in_expr(expr: str):
        """Count unknown variables in expression"""
        expr = expr.replace("sqrt", "")
        expr = expr.replace("frac", "")
        letters_in_expr = set([x for x in expr if x.isalpha()])
        return len(letters_in_expr)
    
    @staticmethod
    def should_allow_eval(expr: str):
        """Check if expression is safe for sympy evaluation"""
        if MathAnswerEvaluator.count_unknown_letters_in_expr(expr) > 2:
            return False
        
        for bad_string in MathAnswerEvaluator.BAD_SUBSTRINGS:
            if bad_string in expr:
                return False
        
        for bad_regex in MathAnswerEvaluator.BAD_REGEXES:
            if re.search(bad_regex, expr) is not None:
                return False
        
        return True
    
    @staticmethod
    def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
        """Check equality using sympy simplification"""
        are_equal = False
        try:
            expr = f"({ground_truth_normalized})-({given_normalized})"
            if MathAnswerEvaluator.should_allow_eval(expr):
                sympy_diff = MathAnswerEvaluator._sympy_parse(expr)
                simplified = sympy.simplify(sympy_diff)
                if simplified == 0:
                    are_equal = True
        except:
            pass
        return are_equal
    
    @staticmethod
    def split_tuple(expr: str):
        """Split tuple/interval elements while handling commas in large numbers"""
        expr = MathAnswerEvaluator._strip_properly_formatted_commas(expr)
        if len(expr) == 0:
            return []
        if (
            len(expr) > 2
            and expr[0] in MathAnswerEvaluator.TUPLE_CHARS
            and expr[-1] in MathAnswerEvaluator.TUPLE_CHARS
            and all([ch not in expr[1:-1] for ch in MathAnswerEvaluator.TUPLE_CHARS])
        ):
            elems = [elem.strip() for elem in expr[1:-1].split(",")]
        else:
            elems = [expr]
        return elems
    
    @staticmethod
    def grade_answer(given_answer: str, ground_truth: str) -> bool:
        """
        Professional math answer grading with multiple validation levels
        """
        if given_answer is None:
            return False
        
        ground_truth_normalized_basic = MathAnswerEvaluator.normalize_answer_basic(ground_truth)
        given_answer_normalized_basic = MathAnswerEvaluator.normalize_answer_basic(given_answer)
        
        if ground_truth_normalized_basic == given_answer_normalized_basic:
            return True
        
        ground_truth_normalized = MathAnswerEvaluator._normalize_advanced(ground_truth)
        given_normalized = MathAnswerEvaluator._normalize_advanced(given_answer)
        
        if ground_truth_normalized is None:
            return False
        
        if ground_truth_normalized == given_normalized:
            return True
        
        if len(given_normalized) == 0:
            return False
        
        ground_truth_elems = MathAnswerEvaluator.split_tuple(ground_truth_normalized)
        given_elems = MathAnswerEvaluator.split_tuple(given_normalized)

        if len(ground_truth_elems) > 1 and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]
        ):
            is_correct = False
        elif len(ground_truth_elems) != len(given_elems):
            is_correct = False
        else:
            is_correct = True
            for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
                if MathAnswerEvaluator._is_frac(ground_truth_elem) and MathAnswerEvaluator._is_frac(given_elem):
                    elem_correct = ground_truth_elem == given_elem
                elif MathAnswerEvaluator._str_is_int(ground_truth_elem) != MathAnswerEvaluator._str_is_int(given_elem):
                    elem_correct = False
                else:
                    elem_correct = MathAnswerEvaluator.are_equal_under_sympy(ground_truth_elem, given_elem)
                
                if not elem_correct:
                    is_correct = False
                    break
        
        return is_correct

class Math500Evaluator:
    def __init__(self, model_path, neuron_file=None, device="auto", seed=42, logger=None):
        """
        Initialize MATH-500 evaluator with professional math answer grading
        
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

        self.math_evaluator = MathAnswerEvaluator()
        
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
        Load MATH-500 dataset using ModelScope
        
        Args:
            split: Dataset split ('test')
            
        Returns:
            List of MATH-500 samples
        """
        self.logger.info(f"Loading MATH-500 dataset from ModelScope, split: {split}")
        
        try:
            ds = MsDataset.load('AI-ModelScope/MATH-500', subset_name='default', split=split)
            self.logger.info(f"Dataset loaded successfully from ModelScope. Total items: {len(ds)}")
            
            samples = []
            for item in tqdm(ds, desc=f"Loading {split} data"):
                sample_dict = dict(item)
                samples.append(sample_dict)
            
            self.logger.info(f"Loaded {len(samples)} samples from MATH-500 {split}")

            if samples:
                sample = samples[0]
                self.logger.info(f"Sample structure: {list(sample.keys())}")
                if 'problem' in sample:
                    self.logger.info(f"Sample problem: {sample.get('problem', '')[:200]}...")
                if 'solution' in sample:
                    self.logger.info(f"Sample solution: {sample.get('solution', '')[:200]}...")
                if 'answer' in sample:
                    self.logger.info(f"Sample answer: {sample.get('answer', '')}")
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Failed to load MATH-500 dataset from ModelScope: {e}")
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
    
    def _extract_answer_from_response(self, response):
        """
        Extract final answer from model response using multiple patterns
        
        Args:
            response: Model generated response
            
        Returns:
            str: Extracted answer or empty string if not found
        """
        if not response:
            return ""

        answer_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}',  
 
            r'(?:final answer|answer|solution|result|conclusion)(?:\s*is\s*|\s*:\s*|\s+)([^.\n]+)',
            r'(?:therefore|thus|hence)(?:\s*,?\s*)(?:the answer is\s*)?([^.\n]+)',

            r'\$([^$]+)\$(?:\s*$|\s*\.|\s*,)',

            r'([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$',
            r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$',

            r'\\frac\{([^}]+)\}\{([^}]+)\}',
            r'([+-]?\d+(?:\.\d+)?(?:/\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*)',
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                if isinstance(matches[0], tuple):  
                    answer = f"\\frac{{{matches[-1][0]}}}{{{matches[-1][1]}}}"
                else:
                    answer = matches[-1].strip()
                if answer:
                    return answer

        lines = response.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and (any(c.isdigit() for c in line) or any(sym in line for sym in ['\\frac', '\\sqrt', '='])):
                math_match = re.search(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\\frac\{[^}]+\}\{[^}]+\})', line)
                if math_match:
                    return math_match.group(1)
        
        return ""
    
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
        """Evaluate a single MATH-500 sample using professional math grading"""
        try:
            problem = sample.get('problem', '')
            if not problem:
                problem = sample.get('question', '') 

            prompt = f"Solve this math problem step by step:\n\n{problem}\n\nSolution:"
            
            response = self.generate_response(prompt, max_length, max_new_tokens)

            is_meaningful = self._is_response_meaningful(response)

            predicted_answer = self._extract_answer_from_response(response)

            ground_truth = sample.get('answer', sample.get('solution', ''))

            is_correct = False
            if predicted_answer and ground_truth and is_meaningful:
                is_correct = self.math_evaluator.grade_answer(predicted_answer, ground_truth)

            debug_info = {
                'predicted_answer_raw': predicted_answer,
                'ground_truth_raw': ground_truth,
                'predicted_normalized_basic': self.math_evaluator.normalize_answer_basic(predicted_answer) if predicted_answer else None,
                'ground_truth_normalized_basic': self.math_evaluator.normalize_answer_basic(ground_truth) if ground_truth else None,
                'predicted_normalized_advanced': self.math_evaluator._normalize_advanced(predicted_answer) if predicted_answer else None,
                'ground_truth_normalized_advanced': self.math_evaluator._normalize_advanced(ground_truth) if ground_truth else None,
            }
            
            return {
                'problem_id': sample.get('id', sample.get('problem_id', '')),
                'problem': problem,
                'ground_truth': ground_truth,
                'response': response,
                'predicted_answer': predicted_answer,
                'is_meaningful_response': is_meaningful,
                'is_correct': is_correct,
                'subject': sample.get('subject', sample.get('type', 'unknown')),
                'debug_info': debug_info
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating sample: {e}")
            return {
                'problem_id': sample.get('id', sample.get('problem_id', '')),
                'problem': sample.get('problem', ''),
                'ground_truth': sample.get('answer', ''),
                'response': '',
                'predicted_answer': '',
                'is_meaningful_response': False,
                'is_correct': False,
                'subject': sample.get('subject', 'unknown'),
                'error': str(e),
                'debug_info': {}
            }
    
    def evaluate_dataset(self, samples, max_length=2048, max_new_tokens=512, num_neurons_to_mask=0):
        """Evaluate the entire MATH-500 dataset with professional math grading"""
        self.logger.info(f"Evaluating {len(samples)} MATH-500 samples")
        self.logger.info(f"Max length: {max_length}, Max new tokens: {max_new_tokens}")
        self.logger.info("Using professional math answer evaluation with sympy and LaTeX parsing")
        
        if num_neurons_to_mask > 0:
            self.logger.info(f"Masking top {num_neurons_to_mask} neurons")
            neurons_to_mask = self.neurons[:num_neurons_to_mask]
            self.apply_neuron_masking(neurons_to_mask)
        else:
            self.logger.info("No neuron masking applied")
            self._clear_masking_hooks()
        
        results = []
        correct_count = 0
        meaningless_responses = 0
        answer_extracted_count = 0

        subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        extraction_failures = []
        grading_failures = []

        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            result = self.evaluate_single_sample(sample, max_length, max_new_tokens)
            results.append(result)

            if result['is_correct']:
                correct_count += 1
            else:
                if not result['is_meaningful_response']:
                    pass  
                elif not result['predicted_answer']:
                    extraction_failures.append({
                        'problem_id': result['problem_id'],
                        'response': result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
                    })
                else:
                    grading_failures.append({
                        'problem_id': result['problem_id'],
                        'predicted': result['predicted_answer'],
                        'ground_truth': result['ground_truth'],
                        'debug_info': result['debug_info']
                    })
            
            if not result['is_meaningful_response']:
                meaningless_responses += 1
            
            if result['predicted_answer']:
                answer_extracted_count += 1

            subject = result['subject']
            subject_stats[subject]['total'] += 1
            if result['is_correct']:
                subject_stats[subject]['correct'] += 1

            if i % 50 == 0:
                for gpu_id in range(self.num_gpus):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()

        total_samples = len(results)
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        meaningless_rate = meaningless_responses / total_samples if total_samples > 0 else 0.0
        answer_extraction_rate = answer_extracted_count / total_samples if total_samples > 0 else 0.0

        subject_accuracies = {}
        for subject, stats in subject_stats.items():
            subject_accuracies[subject] = {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        evaluation_results = {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_samples': total_samples,
            'meaningless_responses': meaningless_responses,
            'meaningless_response_rate': meaningless_rate,
            'answer_extracted_count': answer_extracted_count,
            'answer_extraction_rate': answer_extraction_rate,
            'neurons_masked': num_neurons_to_mask,
            'subject_accuracies': subject_accuracies,
            'extraction_failures': extraction_failures[:10],  
            'grading_failures': grading_failures[:10],  
            'detailed_results': results
        }

        self.logger.info(f"Professional Math Grading Results:")
        self.logger.info(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_samples})")
        self.logger.info(f"Answer Extraction Rate: {answer_extraction_rate:.4f} ({answer_extracted_count}/{total_samples})")
        self.logger.info(f"Meaningless Responses: {meaningless_responses}/{total_samples} ({meaningless_rate*100:.1f}%)")
        self.logger.info(f"Extraction Failures: {len(extraction_failures)}")
        self.logger.info(f"Grading Failures: {len(grading_failures)}")

        for subject, acc_info in subject_accuracies.items():
            self.logger.info(f"Subject {subject}: {acc_info['accuracy']:.4f} ({acc_info['correct']}/{acc_info['total']})")
        
        return evaluation_results
    
    def test_multiple_masking_levels(self, samples, masking_levels, max_length=2048, max_new_tokens=512, output_file="math500_results.json"):
        """Test MATH-500 accuracy with multiple neuron masking levels using professional grading"""
        self.logger.info("Testing multiple neuron masking levels for MATH-500")
        self.logger.info(f"Masking levels: {masking_levels}")
        self.logger.info("Using professional math answer evaluation")
        
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
                "evaluation_method": "professional_math_grading_with_sympy"
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
            self.logger.info(f"  Accuracy = {result.get('accuracy', 0):.4f}")
            self.logger.info(f"  Answer Extraction Rate = {result.get('answer_extraction_rate', 0):.4f}")
            self.logger.info(f"  Meaningless Response Rate = {result.get('meaningless_response_rate', 0):.4f}")

        self.logger.info(f"Saving all results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        self.logger.info("\n=== Professional Math Grading Results Summary ===")
        self.logger.info("Masked | Accuracy | Answer Extract | Meaningless% | Extract Fail | Grade Fail")
        self.logger.info("-------|----------|----------------|--------------|--------------|----------")
        for result in all_results["results"]:
            masked = result.get("masking_level", 0)
            accuracy = result.get("accuracy", 0)
            extract_rate = result.get("answer_extraction_rate", 0)
            meaningless = result.get("meaningless_response_rate", 0) * 100
            extract_fail = len(result.get("extraction_failures", []))
            grade_fail = len(result.get("grading_failures", []))
            self.logger.info(f"{masked:6d} | {accuracy:8.4f} | {extract_rate:14.4f} | {meaningless:9.1f} | {extract_fail:12d} | {grade_fail:10d}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="MATH-500 Dataset Evaluator with Professional Math Grading and Neuron Masking")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--neuron_file", type=str, help="Path to neuron importance file (JSON)")
    parser.add_argument("--split", type=str, default="test", choices=["test"], help="Dataset split")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum input sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--num_neurons_to_mask", type=int, default=0, help="Number of top neurons to mask")
    parser.add_argument("--masking_levels", type=str, help="Comma-separated list of neuron counts to test (e.g., '0,10,50,100')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="math500_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="math500_evaluation.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    parser.add_argument("--sample_size", type=int, help="Number of samples to evaluate (for testing)")
    
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== MATH-500 Dataset Evaluator with Professional Math Grading ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info("Using ModelScope for dataset loading")
    logger.info("Using professional math evaluation with sympy and LaTeX parsing")

    evaluator = Math500Evaluator(
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
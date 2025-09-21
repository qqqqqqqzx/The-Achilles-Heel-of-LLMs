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
import transformers

def setup_logging(log_file="ifeval_evaluation.log", log_level=logging.INFO):
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

class IFEvalEvaluator:
    def __init__(self, model_path, neuron_file=None, device="auto", seed=42, logger=None):
        """
        Initialize IFEval evaluator with neuron masking capability
        
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
        
        self._setup_builtin_evaluation()
        
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
    
    def load_dataset(self, split='train'):
        """
        Load IFEval dataset
        
        Args:
            split: Dataset split ('train')
            
        Returns:
            List of instruction following samples
        """
        self.logger.info(f"Loading IFEval dataset from ModelScope, split: {split}")
        
        try:
            ds = MsDataset.load('opencompass/ifeval', subset_name='default', split=split)
            self.logger.info(f"Dataset loaded successfully from ModelScope. Total items: {len(ds)}")
            
            samples = []
            for item in tqdm(ds, desc=f"Loading {split} data"):
                sample_dict = dict(item)
                samples.append(sample_dict)
            
            self.logger.info(f"Loaded {len(samples)} samples from IFEval {split}")
            
            if samples:
                sample = samples[0]
                self.logger.info(f"Sample structure: {list(sample.keys())}")
                self.logger.info(f"Sample prompt: {sample.get('prompt', '')[:200]}...")
                self.logger.info(f"Sample instruction_id_list: {sample.get('instruction_id_list', [])}")
                self.logger.info(f"Sample kwargs: {sample.get('kwargs', [])}")
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Failed to load IFEval dataset from ModelScope: {e}")
            raise
    
    def _is_response_meaningful(self, response):
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
    
    def _setup_builtin_evaluation(self):
        """Setup built-in instruction evaluation"""
        self.builtin_checkers = {
            'language:response_language': self._check_response_language,
            'length_constraints:number_words': self._check_number_words,
            'detectable_content:number_placeholders': self._check_placeholders,
            'keywords:existence': self._check_keywords_existence,
            'keywords:frequency': self._check_keywords_frequency,
            'keywords:forbidden_words': self._check_forbidden_words,
            'startend:end_checker': self._check_ending,
            'punctuation:no_comma': self._check_no_comma,
            'case:all_capital': self._check_all_capital,
            'case:all_lowercase': self._check_all_lowercase,
            'detectable_format:json_format': self._check_json_format,
            'detectable_format:title': self._check_title_format,
            'combination:two_responses': self._check_two_responses,
            'detectable_format:multiple_sections': self._check_multiple_sections,
            'length_constraints:number_sentences': self._check_number_sentences,
            'length_constraints:number_paragraphs': self._check_number_paragraphs,
            'detectable_content:postscript': self._check_postscript,
            'change_case:capital_word_frequency': self._check_capital_word_frequency,
            'change_case:english_capital': self._check_english_capital,
            'change_case:english_lowercase': self._check_english_lowercase,
            'detectable_format:constrain_two_responses': self._check_constrain_two_responses,
            'detectable_format:number_highlighted_sections': self._check_highlighted_sections,
            'keywords:letter_frequency': self._check_letter_frequency,
            'startend:quotation': self._check_quotation,
        }
        self.logger.info(f"Setup {len(self.builtin_checkers)} built-in instruction checkers")
        
    def _extract_number_from_kwargs(self, kwargs, required=True):
        for key in ['num_words', 'num_sentences', 'num_paragraphs', 'frequency', 'number', 'count']:
            if key in kwargs and isinstance(kwargs[key], (int, str)):
                try:
                    return int(kwargs[key])
                except ValueError:
                    continue
        
        for value in kwargs.values():
            if isinstance(value, int):
                return value
            elif isinstance(value, str):
                numbers = re.findall(r'\d+', value)
                if numbers:
                    return int(numbers[0])
        
        if required:
            self.logger.debug(f"Failed to extract number from kwargs: {kwargs}")
            return None
        
        return 1 
    
    def _extract_relation_from_kwargs(self, kwargs, required=False):
        for key in ['relation', 'comparison', 'operator']:
            if key in kwargs:
                return str(kwargs[key]).lower()
        
        for value in kwargs.values():
            if isinstance(value, str):
                value_lower = value.lower()
                if 'less than' in value_lower:
                    return 'less than'
                elif 'more than' in value_lower or 'at least' in value_lower:
                    return 'at least'
                elif 'exactly' in value_lower:
                    return 'exactly'
        
        if required:
            self.logger.debug(f"Failed to extract relation from kwargs: {kwargs}")
            return None
        
        return 'at least' 
        
    def _check_response_language(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        if 'language' not in kwargs:
            self.logger.debug("Language parameter not found in kwargs")
            return False
            
        language = kwargs.get('language', '').lower()
        if not language:
            self.logger.debug("Empty language parameter")
            return False
        
        if 'chinese' in language or 'zh' in language:
            return bool(re.search(r'[\u4e00-\u9fff]', response))
        elif 'french' in language or 'fr' in language:
            french_words = ['le', 'la', 'les', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que', 'pour', 'avec', 'ne', 'se', 'pas', 'tout', 'plus']
            response_words = response.lower().split()
            french_count = sum(1 for word in french_words if word in response_words)
            return french_count >= 3  
        elif 'spanish' in language or 'es' in language:
            spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para']
            response_words = response.lower().split()
            spanish_count = sum(1 for word in spanish_words if word in response_words)
            return spanish_count >= 3
        elif 'english' in language or 'en' in language:
            return not bool(re.search(r'[\u4e00-\u9fff]', response))
        else:
            self.logger.debug(f"Unknown language: {language}")
            return False
            
    def _check_number_words(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        target = self._extract_number_from_kwargs(kwargs, required=True)
        if target is None:
            return False
            
        relation = self._extract_relation_from_kwargs(kwargs, required=False)
        
        words = response.split()
        meaningful_words = [w for w in words if any(c.isalpha() for c in w) and len(w.strip(string.punctuation)) > 0]
        num_words = len(meaningful_words)
        
        if relation == 'less than':
            return num_words < target
        elif relation == 'at least':
            return num_words >= target
        elif relation == 'exactly':
            return num_words == target
        else:
            return False
        
    def _check_placeholders(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        num_placeholders = self._extract_number_from_kwargs(kwargs, required=True)
        if num_placeholders is None:
            return False
        
        placeholders = []
        placeholders.extend(re.findall(r'\[.*?\]', response))  
        placeholders.extend(re.findall(r'\{.*?\}', response))  
        placeholders.extend(re.findall(r'<.*?>', response))    
        placeholders.extend(re.findall(r'___+', response))   
        
        return len(placeholders) >= num_placeholders
        
    def _check_keywords_existence(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        keywords = kwargs.get('keywords', kwargs.get('keyword', None))
        
        if keywords is None:
            self.logger.debug("No keywords parameter found")
            return False
            
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            self.logger.debug(f"Invalid keywords format: {type(keywords)}")
            return False
        
        if not keywords:
            self.logger.debug("Empty keywords list")
            return False
        
        response_lower = response.lower()
        return all(keyword.lower() in response_lower for keyword in keywords)
        
    def _check_keywords_frequency(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        keyword = kwargs.get('keyword', kwargs.get('keywords', None))
        
        if keyword is None:
            self.logger.debug("No keyword parameter found")
            return False
            
        if isinstance(keyword, list):
            if not keyword:
                return False
            keyword = keyword[0]
        
        if not isinstance(keyword, str) or not keyword.strip():
            self.logger.debug("Invalid keyword format")
            return False
        
        frequency = self._extract_number_from_kwargs(kwargs, required=True)
        if frequency is None:
            return False
            
        relation = self._extract_relation_from_kwargs(kwargs, required=False)
        
        count = response.lower().count(keyword.lower())
        
        if relation == 'at least':
            return count >= frequency
        elif relation == 'less than':
            return count < frequency
        elif relation == 'exactly':
            return count == frequency
        else:
            return False
        
    def _check_forbidden_words(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        forbidden_words = kwargs.get('forbidden_words', kwargs.get('forbidden', None))
        
        if forbidden_words is None:
            self.logger.debug("No forbidden words parameter found")
            return False
            
        if isinstance(forbidden_words, str):
            forbidden_words = [forbidden_words]
        elif not isinstance(forbidden_words, list):
            self.logger.debug(f"Invalid forbidden words format: {type(forbidden_words)}")
            return False
        
        if not forbidden_words:
            return True
        
        response_lower = response.lower()
        return not any(word.lower() in response_lower for word in forbidden_words)
        
    def _check_ending(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        end_phrase = kwargs.get('end_phrase', kwargs.get('ending', None))
        
        if end_phrase is None:
            self.logger.debug("No end phrase parameter found")
            return False
            
        if not isinstance(end_phrase, str) or not end_phrase.strip():
            self.logger.debug("Invalid end phrase format")
            return False
            
        return response.strip().endswith(str(end_phrase))
        
    def _check_no_comma(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False           
        return ',' not in response
        
    def _check_all_capital(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        letters = ''.join(c for c in response if c.isalpha())
        return letters.isupper() if letters else False
        
    def _check_all_lowercase(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        letters = ''.join(c for c in response if c.isalpha())
        return letters.islower() if letters else False
        
    def _check_json_format(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        try:
            json.loads(response.strip())
            return True
        except:
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            if matches:
                try:
                    json.loads(matches[0])
                    return True
                except:
                    pass
            return False
            
    def _check_title_format(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        title_patterns = [
            r'^Title:.*$',
            r'^###.*###$',
            r'^\*\*.*\*\*$',
            r'^# .*$',
            r'^## .*$',
            r'^### .*$'
        ]
        lines = response.split('\n')
        return any(any(re.match(pattern, line.strip()) for pattern in title_patterns) for line in lines)
        
    def _check_two_responses(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        separators = [
            'response 1:', 'response 2:', 
            'answer 1:', 'answer 2:',
            'option 1:', 'option 2:',
            'first:', 'second:',
            '1.', '2.'
        ]
        response_lower = response.lower()
        count = sum(sep in response_lower for sep in separators)
        return count >= 2
        
    def _check_multiple_sections(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        num_sections = self._extract_number_from_kwargs(kwargs, required=True)
        if num_sections is None:
            return False
        
        section_patterns = [
            r'^#+\s',      
            r'^\d+\.',     
            r'^[A-Z][a-z]*:', 
            r'^\*\*.*\*\*', 
        ]
        lines = response.split('\n')
        section_count = 0
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and any(re.match(pattern, line_stripped) for pattern in section_patterns):
                section_count += 1
        
        return section_count >= num_sections
        
    def _check_number_sentences(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        target = self._extract_number_from_kwargs(kwargs, required=True)
        if target is None:
            return False
            
        relation = self._extract_relation_from_kwargs(kwargs, required=False)
        
        sentences = re.split(r'[.!?]+', response)
        meaningful_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if s_clean and len(s_clean) > 3 and any(c.isalpha() for c in s_clean):
                meaningful_sentences.append(s_clean)
        
        num_sentences = len(meaningful_sentences)
        
        if relation == 'less than':
            return num_sentences < target
        elif relation == 'at least':
            return num_sentences >= target
        elif relation == 'exactly':
            return num_sentences == target
        else:
            return False
        
    def _check_number_paragraphs(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        target = self._extract_number_from_kwargs(kwargs, required=True)
        if target is None:
            return False
            
        relation = self._extract_relation_from_kwargs(kwargs, required=False)

        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        meaningful_paragraphs = []
        for p in paragraphs:
            if len(p) > 10 and any(c.isalpha() for c in p):  
                meaningful_paragraphs.append(p)
        
        num_paragraphs = len(meaningful_paragraphs)
        
        if relation == 'less than':
            return num_paragraphs < target
        elif relation == 'at least':
            return num_paragraphs >= target
        elif relation == 'exactly':
            return num_paragraphs == target
        else:
            return False
        
    def _check_postscript(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        postscript_markers = ['P.S.', 'PS:', 'Postscript:', 'P.P.S.', 'Note:']
        return any(marker in response for marker in postscript_markers)
        
    def _check_capital_word_frequency(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        frequency = self._extract_number_from_kwargs(kwargs, required=True)
        if frequency is None:
            return False
            
        relation = self._extract_relation_from_kwargs(kwargs, required=False)
        
        words = response.split()
        capital_words = [w for w in words if w.isupper() and w.isalpha() and len(w) > 1]
        
        count = len(capital_words)
        if relation == 'at least':
            return count >= frequency
        elif relation == 'less than':
            return count < frequency
        elif relation == 'exactly':
            return count == frequency
        else:
            return False
        
    def _check_english_capital(self, response, **kwargs):
        return self._check_all_capital(response, **kwargs)
        
    def _check_english_lowercase(self, response, **kwargs):
        return self._check_all_lowercase(response, **kwargs)
        
    def _check_constrain_two_responses(self, response, **kwargs):
        return self._check_two_responses(response, **kwargs)
        
    def _check_highlighted_sections(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        num_highlighted = self._extract_number_from_kwargs(kwargs, required=True)
        if num_highlighted is None:
            return False
        
        highlighted = []
        highlighted.extend(re.findall(r'\*\*.*?\*\*', response))  
        highlighted.extend(re.findall(r'\*.*?\*', response))     
        highlighted.extend(re.findall(r'__.*?__', response))     
        highlighted.extend(re.findall(r'_.*?_', response))        
        
        return len(highlighted) >= num_highlighted
        
    def _check_letter_frequency(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        letter = kwargs.get('letter', kwargs.get('char', None))
        
        if letter is None:
            self.logger.debug("No letter parameter found")
            return False
            
        if isinstance(letter, list):
            if not letter:
                return False
            letter = letter[0]
        
        if not isinstance(letter, str) or len(letter) != 1:
            self.logger.debug("Invalid letter format")
            return False
        
        frequency = self._extract_number_from_kwargs(kwargs, required=True)
        if frequency is None:
            return False
            
        relation = self._extract_relation_from_kwargs(kwargs, required=False)
        
        count = response.lower().count(letter.lower())
        if relation == 'at least':
            return count >= frequency
        elif relation == 'less than':
            return count < frequency
        elif relation == 'exactly':
            return count == frequency
        else:
            return False
        
    def _check_quotation(self, response, **kwargs):
        if not self._is_response_meaningful(response):
            return False
            
        stripped = response.strip()
        return ((stripped.startswith('"') and stripped.endswith('"')) or 
                (stripped.startswith("'") and stripped.endswith("'")) or
                (stripped.startswith('"') and stripped.endswith('"')) or
                (stripped.startswith(''') and stripped.endswith(''')))

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
    
    def evaluate_instruction_following(self, sample, response, mode='strict'):
        """
        Evaluate instruction following for a single sample using built-in evaluation
        
        Args:
            sample: IFEval sample containing prompt, instruction_id_list, kwargs
            response: Model generated response
            mode: 'strict' or 'loose' evaluation mode
            
        Returns:
            Dict with evaluation results
        """
        instruction_id_list = sample['instruction_id_list']
        kwargs_list = sample['kwargs']
        
        if not self._is_response_meaningful(response):
            self.logger.debug(f"Response is not meaningful: {response[:100]}")
            follow_instruction_list = [False] * len(instruction_id_list)
            return {
                'instruction_id_list': instruction_id_list,
                'prompt': sample['prompt'],
                'response': response,
                'follow_all_instructions': False,
                'follow_instruction_list': follow_instruction_list,
                'num_instructions': len(instruction_id_list),
                'num_followed': 0,
                'meaningless_response': True
            }
        
        if mode == 'loose':
            r = response.split("\n")
            response_variants = [
                response,
                response.replace("*", ""),
                "\n".join(r[1:]).strip() if len(r) > 1 else response,  
                "\n".join(r[:-1]).strip() if len(r) > 1 else response, 
                "\n".join(r[1:-1]).strip() if len(r) > 2 else response, 
                "\n".join(r[1:]).strip().replace("*", "") if len(r) > 1 else response, 
                "\n".join(r[:-1]).strip().replace("*", "") if len(r) > 1 else response, 
                "\n".join(r[1:-1]).strip().replace("*", "") if len(r) > 2 else response, 
            ]
            response_variants = [v for v in response_variants if self._is_response_meaningful(v)]
        else:
            response_variants = [response]
        
        follow_instruction_list = []
        
        for i, instruction_id in enumerate(instruction_id_list):
            kwargs = kwargs_list[i] if i < len(kwargs_list) else {}
            
            checker = self.builtin_checkers.get(instruction_id)
            is_following = False
            
            if checker:
                try:
                    for variant in response_variants:
                        if variant.strip() and checker(variant, **kwargs):
                            is_following = True
                            break
                except Exception as e:
                    self.logger.debug(f"Error checking instruction {instruction_id}: {e}")
                    is_following = False
            else:
                self.logger.warning(f"Unknown instruction type: {instruction_id}")
                is_following = False
            
            follow_instruction_list.append(is_following)
        
        return {
            'instruction_id_list': instruction_id_list,
            'prompt': sample['prompt'],
            'response': response,
            'follow_all_instructions': all(follow_instruction_list),
            'follow_instruction_list': follow_instruction_list,
            'num_instructions': len(instruction_id_list),
            'num_followed': sum(follow_instruction_list),
            'meaningless_response': False
        }
    
    def evaluate_single_sample(self, sample, max_length=2048, max_new_tokens=512):
        """Evaluate a single IFEval sample"""
        try:
            prompt = sample['prompt']
            response = self.generate_response(prompt, max_length, max_new_tokens)
            
            is_meaningful = self._is_response_meaningful(response)
            
            strict_result = self.evaluate_instruction_following(sample, response, mode='strict')
            loose_result = self.evaluate_instruction_following(sample, response, mode='loose')
            
            return {
                'key': sample.get('key', 0),
                'prompt': prompt,
                'response': response,
                'is_meaningful_response': is_meaningful,
                'instruction_id_list': sample['instruction_id_list'],
                'kwargs': sample['kwargs'],
                'strict_evaluation': strict_result,
                'loose_evaluation': loose_result,
                'num_instructions': len(sample['instruction_id_list'])
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating sample: {e}")
            return {
                'key': sample.get('key', 0),
                'prompt': sample.get('prompt', ''),
                'response': '',
                'is_meaningful_response': False,
                'instruction_id_list': sample.get('instruction_id_list', []),
                'kwargs': sample.get('kwargs', []),
                'strict_evaluation': {
                    'follow_all_instructions': False,
                    'follow_instruction_list': [],
                    'num_instructions': 0,
                    'num_followed': 0,
                    'meaningless_response': True
                },
                'loose_evaluation': {
                    'follow_all_instructions': False,
                    'follow_instruction_list': [],
                    'num_instructions': 0,
                    'num_followed': 0,
                    'meaningless_response': True
                },
                'num_instructions': 0,
                'error': str(e)
            }
    
    def evaluate_dataset(self, samples, max_length=2048, max_new_tokens=512, num_neurons_to_mask=0):
        """Evaluate the entire IFEval dataset"""
        self.logger.info(f"Evaluating {len(samples)} IFEval samples")
        self.logger.info(f"Max length: {max_length}, Max new tokens: {max_new_tokens}")
        
        if num_neurons_to_mask > 0:
            self.logger.info(f"Masking top {num_neurons_to_mask} neurons")
            neurons_to_mask = self.neurons[:num_neurons_to_mask]
            self.apply_neuron_masking(neurons_to_mask)
        else:
            self.logger.info("No neuron masking applied")
            self._clear_masking_hooks()
        
        results = []
        instruction_stats = defaultdict(lambda: {'strict_correct': 0, 'loose_correct': 0, 'total': 0})
        meaningless_responses = 0
        
        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            result = self.evaluate_single_sample(sample, max_length, max_new_tokens)
            results.append(result)
            
            if not result.get('is_meaningful_response', True):
                meaningless_responses += 1
            
            for j, instruction_id in enumerate(result['instruction_id_list']):
                instruction_stats[instruction_id]['total'] += 1
                
                if j < len(result['strict_evaluation']['follow_instruction_list']):
                    if result['strict_evaluation']['follow_instruction_list'][j]:
                        instruction_stats[instruction_id]['strict_correct'] += 1
                
                if j < len(result['loose_evaluation']['follow_instruction_list']):
                    if result['loose_evaluation']['follow_instruction_list'][j]:
                        instruction_stats[instruction_id]['loose_correct'] += 1
            
            if i % 50 == 0:
                for gpu_id in range(self.num_gpus):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
        
        strict_prompt_correct = sum(1 for r in results if r['strict_evaluation']['follow_all_instructions'])
        loose_prompt_correct = sum(1 for r in results if r['loose_evaluation']['follow_all_instructions'])
        
        strict_instruction_correct = sum(r['strict_evaluation']['num_followed'] for r in results)
        loose_instruction_correct = sum(r['loose_evaluation']['num_followed'] for r in results)
        
        total_prompts = len(results)
        total_instructions = sum(r['num_instructions'] for r in results)
        
        strict_prompt_accuracy = strict_prompt_correct / total_prompts if total_prompts > 0 else 0.0
        loose_prompt_accuracy = loose_prompt_correct / total_prompts if total_prompts > 0 else 0.0
        
        strict_instruction_accuracy = strict_instruction_correct / total_instructions if total_instructions > 0 else 0.0
        loose_instruction_accuracy = loose_instruction_correct / total_instructions if total_instructions > 0 else 0.0
        
        instruction_accuracies = {}
        for instruction_id, stats in instruction_stats.items():
            strict_acc = stats['strict_correct'] / stats['total'] if stats['total'] > 0 else 0.0
            loose_acc = stats['loose_correct'] / stats['total'] if stats['total'] > 0 else 0.0
            
            instruction_accuracies[instruction_id] = {
                'strict_accuracy': strict_acc,
                'loose_accuracy': loose_acc,
                'strict_correct': stats['strict_correct'],
                'loose_correct': stats['loose_correct'],
                'total': stats['total']
            }
        
        evaluation_results = {
            'strict_prompt_accuracy': strict_prompt_accuracy,
            'loose_prompt_accuracy': loose_prompt_accuracy,
            'strict_instruction_accuracy': strict_instruction_accuracy,
            'loose_instruction_accuracy': loose_instruction_accuracy,
            'strict_prompt_correct': strict_prompt_correct,
            'loose_prompt_correct': loose_prompt_correct,
            'strict_instruction_correct': strict_instruction_correct,
            'loose_instruction_correct': loose_instruction_correct,
            'total_prompts': total_prompts,
            'total_instructions': total_instructions,
            'meaningless_responses': meaningless_responses,
            'meaningless_response_rate': meaningless_responses / total_prompts if total_prompts > 0 else 0.0,
            'neurons_masked': num_neurons_to_mask,
            'instruction_accuracies': instruction_accuracies,
            'detailed_results': results
        }
        
        self.logger.info(f"Strict Prompt Accuracy: {strict_prompt_accuracy:.4f} ({strict_prompt_correct}/{total_prompts})")
        self.logger.info(f"Loose Prompt Accuracy: {loose_prompt_accuracy:.4f} ({loose_prompt_correct}/{total_prompts})")
        self.logger.info(f"Strict Instruction Accuracy: {strict_instruction_accuracy:.4f} ({strict_instruction_correct}/{total_instructions})")
        self.logger.info(f"Loose Instruction Accuracy: {loose_instruction_accuracy:.4f} ({loose_instruction_correct}/{total_instructions})")
        self.logger.info(f"Meaningless Responses: {meaningless_responses}/{total_prompts} ({meaningless_responses/total_prompts*100:.1f}%)")
        
        return evaluation_results
    
    def test_multiple_masking_levels(self, samples, masking_levels, max_length=2048, max_new_tokens=512, output_file="ifeval_results.json"):
        """Test instruction following accuracy with multiple neuron masking levels"""
        self.logger.info("Testing multiple neuron masking levels for IFEval")
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
            self.logger.info(f"  Strict Prompt Accuracy = {result.get('strict_prompt_accuracy', 0):.4f}")
            self.logger.info(f"  Loose Prompt Accuracy = {result.get('loose_prompt_accuracy', 0):.4f}")
            self.logger.info(f"  Strict Instruction Accuracy = {result.get('strict_instruction_accuracy', 0):.4f}")
            self.logger.info(f"  Loose Instruction Accuracy = {result.get('loose_instruction_accuracy', 0):.4f}")
            self.logger.info(f"  Meaningless Response Rate = {result.get('meaningless_response_rate', 0):.4f}")
        
        self.logger.info(f"Saving all results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.info("\n=== Results Summary ===")
        self.logger.info("Masked | Strict Prompt | Loose Prompt | Strict Instr | Loose Instr | Meaningless%")
        self.logger.info("-------|---------------|--------------|--------------|-------------|-------------")
        for result in all_results["results"]:
            masked = result.get("masking_level", 0)
            strict_prompt = result.get("strict_prompt_accuracy", 0)
            loose_prompt = result.get("loose_prompt_accuracy", 0)
            strict_instr = result.get("strict_instruction_accuracy", 0)
            loose_instr = result.get("loose_instruction_accuracy", 0)
            meaningless = result.get("meaningless_response_rate", 0) * 100
            self.logger.info(f"{masked:6d} | {strict_prompt:11.4f} | {loose_prompt:10.4f} | {strict_instr:10.4f} | {loose_instr:9.4f} | {meaningless:9.1f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="IFEval Dataset Evaluator with Neuron Masking and Response Quality Check")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--neuron_file", type=str, help="Path to neuron importance file (JSON)")
    parser.add_argument("--split", type=str, default="train", choices=["train"], help="Dataset split")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum input sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--num_neurons_to_mask", type=int, default=0, help="Number of top neurons to mask")
    parser.add_argument("--masking_levels", type=str, help="Comma-separated list of neuron counts to test (e.g., '0,10,50,100')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="ifeval_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="ifeval_evaluation.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    parser.add_argument("--sample_size", type=int, help="Number of samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== IFEval Dataset Evaluator with Response Quality Check ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info("Using ModelScope for dataset loading")
    logger.info("Enhanced with response quality filtering")
    
    evaluator = IFEvalEvaluator(
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

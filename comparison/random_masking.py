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
import statistics
import transformers

def setup_logging(log_file="wikitext_perplexity.log", log_level=logging.INFO):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RandomNeuronMaskingTester:
    def __init__(self, model_path, device="auto", seed=42, logger=None):
        """
        Initialize Random Neuron Masking Tester
        
        Args:
            model_path: Model path
            device: Computing device ("auto" for multi-GPU)
            seed: Random seed
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        set_seed(seed)
        self.seed = seed
        
        self.device = device
        self.num_gpus = torch.cuda.device_count()
        self.logger.info(f"Available GPUs: {self.num_gpus}")
        
        self._load_model_multi_gpu(model_path)
        
        self.masking_hooks = []
        self.current_masked_neurons = []
        
        self.all_neurons = self._collect_all_neurons()
        
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
        """Print model device allocation"""
        self.logger.info("Model device allocation:")
        if hasattr(self.model, 'hf_device_map'):
            device_summary = defaultdict(int)
            for layer_name, device in self.model.hf_device_map.items():
                device_summary[str(device)] += 1
            
            for device, count in device_summary.items():
                self.logger.info(f"  Device {device}: {count} layers")
        else:
            self.logger.info("  Device map not available")
    
    def _collect_all_neurons(self):
        self.logger.info("Collecting all neuron positions...")
        all_neurons = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        if param.dim() >= 2:
                            output_dim = param.shape[0]
                            for i in range(output_dim):
                                neuron = {
                                    'layer_name': name,
                                    'param_name': param_name,
                                    'neuron_idx': i,
                                    'output_dim': output_dim
                                }
                                all_neurons.append(neuron)
        
        self.logger.info(f"Collected {len(all_neurons)} neurons for potential masking")
        return all_neurons
    
    def _generate_unique_neuron_sets(self, num_neurons_to_mask, num_sets, base_seed):
        """
        Generate multiple unique sets of neurons to mask
        
        Args:
            num_neurons_to_mask: Number of neurons per set
            num_sets: Number of unique sets to generate
            base_seed: Base random seed
            
        Returns:
            List of neuron sets (each set is a list of neurons)
        """
        if num_neurons_to_mask == 0:
            return [[] for _ in range(num_sets)]
        
        if num_neurons_to_mask > len(self.all_neurons):
            self.logger.warning(f"Requested {num_neurons_to_mask} neurons but only {len(self.all_neurons)} available")
            num_neurons_to_mask = len(self.all_neurons)
        
        from math import comb
        total_combinations = comb(len(self.all_neurons), num_neurons_to_mask)
        if total_combinations < num_sets:
            self.logger.warning(f"Only {total_combinations} unique combinations possible, but {num_sets} requested")
        
        neuron_sets = []
        used_sets = set()
        
        max_attempts = min(10000, total_combinations * 10)  
        attempts = 0
        
        random.seed(base_seed)
        
        while len(neuron_sets) < num_sets and attempts < max_attempts:
            attempts += 1
            
            candidate_neurons = random.sample(self.all_neurons, num_neurons_to_mask)
            
            neuron_signature = tuple(sorted(
                (neuron['layer_name'], neuron['neuron_idx']) 
                for neuron in candidate_neurons
            ))

            if neuron_signature not in used_sets:
                used_sets.add(neuron_signature)
                neuron_sets.append(candidate_neurons)

                random.seed(base_seed + len(neuron_sets) * 1000)
        
        if len(neuron_sets) < num_sets:
            self.logger.warning(f"Could only generate {len(neuron_sets)} unique sets out of {num_sets} requested after {attempts} attempts")
        else:
            self.logger.info(f"Successfully generated {len(neuron_sets)} unique neuron sets")
        
        return neuron_sets

    def _setup_masking_hooks(self, masked_neurons):
        """
        Set up hooks to mask specific neurons
        
        Args:
            masked_neurons: List of neurons to mask
            
        Returns:
            List of masked neurons
        """
        self._clear_masking_hooks()
        
        if not masked_neurons:
            return []

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
                            neuron_idx = neuron['neuron_idx']

                            if masked_output.dim() == 3:  
                                if neuron_idx < masked_output.shape[-1]:
                                    masked_output[:, :, neuron_idx] = 0.0
                            elif masked_output.dim() == 2:  
                                if neuron_idx < masked_output.shape[-1]:
                                    masked_output[:, neuron_idx] = 0.0
                            elif masked_output.dim() == 1: 
                                if neuron_idx < masked_output.shape[0]:
                                    masked_output[neuron_idx] = 0.0
                        
                        except Exception as e:
                            self.logger.debug(f"Failed to mask neuron in {layer_name}: {e}")
                    
                    return masked_output
                return hook

            for name, module in self.model.named_modules():
                if name == layer_name:
                    hook = module.register_forward_hook(
                        create_layer_mask_hook(layer_name, layer_neurons)
                    )
                    self.masking_hooks.append(hook)
                    hook_count += 1
                    break
        
        self.current_masked_neurons = masked_neurons.copy()
        self.logger.debug(f"Set up {hook_count} masking hooks for {len(neurons_by_layer)} layers")
        return masked_neurons
    
    def _clear_masking_hooks(self):
        """Clear all masking hooks"""
        for hook in self.masking_hooks:
            hook.remove()
        self.masking_hooks.clear()
        self.current_masked_neurons.clear()
    
    def load_wikitext_dataset(self, dataset_path=None, dataset_name='wikitext', subset_name='wikitext-103-v1', split='test', max_samples=1000):
        """
        Load WikiText dataset from local file or HuggingFace datasets
        
        Args:
            dataset_path: Path to local WikiText file (e.g., 'wiki.train.tokens')
            dataset_name: Dataset name for HuggingFace (default: 'wikitext')
            subset_name: Dataset subset name
            split: Dataset split ('test', 'train', 'validation')
            max_samples: Maximum number of samples to load
            
        Returns:
            List of text samples
        """
        if dataset_path:
            return self._load_local_wikitext(dataset_path, max_samples)
        else:
            return self._load_hf_wikitext(dataset_name, subset_name, split, max_samples)
    
    def _load_local_wikitext(self, dataset_path, max_samples=None):
        """Load WikiText from local file"""
        self.logger.info(f"Loading WikiText from local file: {dataset_path}")
        
        try:
            texts = []
            sample_count = 0
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                current_text = ""
                
                for line_num, line in enumerate(tqdm(f, desc="Reading local file")):
                    line = line.strip()

                    if line == "":
                        if current_text.strip() and len(current_text.strip()) > 50:
                            texts.append(current_text.strip())
                            sample_count += 1
                            
                            if max_samples and sample_count >= max_samples:
                                break
                        current_text = ""
                    else:
                        if not (line.startswith('=') and line.endswith('=')):
                            current_text += line + " "

                if current_text.strip() and len(current_text.strip()) > 50:
                    if not max_samples or sample_count < max_samples:
                        texts.append(current_text.strip())
                        sample_count += 1
            
            self.logger.info(f"Loaded {len(texts)} text samples from local WikiText file")
            return texts
            
        except FileNotFoundError:
            self.logger.error(f"Local WikiText file not found: {dataset_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load local WikiText file: {e}")
            raise
    
    def _load_hf_wikitext(self, dataset_name, subset_name, split, max_samples):
        """Load WikiText from HuggingFace datasets"""
        self.logger.info(f"Loading WikiText dataset: {dataset_name}/{subset_name}, split: {split}")
        
        try:
            dataset = load_dataset(dataset_name, subset_name, split=split)

            texts = []
            sample_count = 0
            
            for item in tqdm(dataset, desc=f"Loading {split} data"):
                text = item.get('text', '').strip()
                if text and len(text) > 50: 
                    texts.append(text)
                    sample_count += 1
                    
                    if max_samples and sample_count >= max_samples:
                        break
            
            self.logger.info(f"Loaded {len(texts)} text samples from WikiText {subset_name} {split}")
            return texts
            
        except Exception as e:
            self.logger.error(f"Failed to load WikiText dataset: {e}")
            raise
    
    def compute_perplexity_single_run(self, texts, masked_neurons, max_length=512, batch_size=1):
        """
        Compute perplexity for a single run with specific masked neurons
        
        Args:
            texts: List of text samples
            masked_neurons: List of neurons to mask
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            Perplexity value for this run
        """
        self._setup_masking_hooks(masked_neurons)

        first_param = next(self.model.parameters())
        input_device = first_param.device
        
        total_loss = 0.0
        total_tokens = 0
        valid_samples = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                try:
                    encodings = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_length,
                        padding=False
                    )
                    
                    input_ids = encodings.input_ids.to(input_device)

                    if input_ids.size(1) < 2:
                        continue

                    with torch.no_grad():
                        outputs = self.model(input_ids, labels=input_ids)
                        loss = outputs.loss

                    total_loss += loss.item() * input_ids.size(1)
                    total_tokens += input_ids.size(1)
                    valid_samples += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing text sample: {e}")
                    continue
 
            if i % (batch_size * 10) == 0:
                for gpu_id in range(self.num_gpus):
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
        
        if total_tokens == 0:
            self.logger.error("No valid tokens processed")
            return float('inf')

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def test_random_masking_with_stats(self, texts, masking_levels=[0, 10, 100, 1000], num_runs=10, max_length=512, batch_size=1, output_file="random_masking_results.json"):
        """
        Test perplexity with random neuron masking at multiple levels, computing statistics
        
        Args:
            texts: List of text samples
            masking_levels: List of neuron counts to mask
            num_runs: Number of experimental runs for each masking level
            max_length: Maximum sequence length
            batch_size: Batch size
            output_file: Output file name
            
        Returns:
            Dictionary with all results including statistics
        """
        self.logger.info(f"Testing random neuron masking with statistics")
        self.logger.info(f"Masking levels: {masking_levels}")
        self.logger.info(f"Number of runs per level: {num_runs}")
        
        all_results = {
            "model_info": {
                "model_path": getattr(self, 'model_path', 'unknown'),
                "num_gpus": self.num_gpus,
                "total_neurons": len(self.all_neurons)
            },
            "dataset_info": {
                "num_texts": len(texts),
                "max_length": max_length,
                "batch_size": batch_size
            },
            "experiment_info": {
                "masking_levels": masking_levels,
                "num_runs": num_runs,
                "base_seed": self.seed
            },
            "results": []
        }
        
        for masking_level in masking_levels:
            self.logger.info(f"\n=== Testing masking level: {masking_level} neurons ===")

            neuron_sets = self._generate_unique_neuron_sets(
                num_neurons_to_mask=masking_level,
                num_sets=num_runs,
                base_seed=self.seed + masking_level
            )
            
            perplexities = []
            actual_runs = len(neuron_sets) 

            for run in range(actual_runs):
                self.logger.info(f"Run {run + 1}/{actual_runs}")

                current_neurons = neuron_sets[run]
                if current_neurons:
                    unique_layers = len(set(n['layer_name'] for n in current_neurons))
                    self.logger.debug(f"  Masking {len(current_neurons)} neurons from {unique_layers} layers")

                perplexity = self.compute_perplexity_single_run(
                    texts=texts,
                    masked_neurons=current_neurons,
                    max_length=max_length,
                    batch_size=batch_size
                )
                
                perplexities.append(perplexity)
                self.logger.info(f"  Perplexity: {perplexity:.4f}")

                self._clear_masking_hooks()

            if perplexities:
                mean_ppl = statistics.mean(perplexities)
                std_ppl = statistics.stdev(perplexities) if len(perplexities) > 1 else 0.0
                min_ppl = min(perplexities)
                max_ppl = max(perplexities)
            else:
                mean_ppl = std_ppl = min_ppl = max_ppl = 0.0
            
            result = {
                "masking_level": masking_level,
                "num_runs": actual_runs,
                "actual_runs": len(perplexities),
                "perplexities": perplexities,
                "mean_perplexity": mean_ppl,
                "std_perplexity": std_ppl,
                "min_perplexity": min_ppl,
                "max_perplexity": max_ppl,
                "unique_neuron_sets": actual_runs == num_runs
            }
            
            all_results["results"].append(result)
            
            self.logger.info(f"Masking {masking_level} neurons:")
            self.logger.info(f"  Mean perplexity: {mean_ppl:.4f} ± {std_ppl:.4f}")
            self.logger.info(f"  Range: [{min_ppl:.4f}, {max_ppl:.4f}]")

        self.logger.info(f"Saving all results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        self.logger.info("\n=== Final Results Summary ===")
        for result in all_results["results"]:
            masked = result["masking_level"]
            mean_ppl = result["mean_perplexity"]
            std_ppl = result["std_perplexity"]
            self.logger.info(f"Masked {masked:4d} neurons: {mean_ppl:8.4f} ± {std_ppl:6.4f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Random Neuron Masking Perplexity Tester")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--dataset_path", type=str, help="Path to local WikiText file (e.g., 'wiki.train.tokens')")
    parser.add_argument("--subset_name", type=str, default="wikitext-103-v1", help="WikiText subset name (for HuggingFace)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "validation"], help="Dataset split (for HuggingFace)")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to test")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--masking_levels", type=str, default="0,10,100,1000", help="Comma-separated list of neuron counts to test")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of experimental runs per masking level")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output_file", type=str, default="random_masking_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="random_masking.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== Random Neuron Masking Perplexity Tester ===")
    logger.info(f"Model path: {args.model_path}")
    if args.dataset_path:
        logger.info(f"Local dataset: {args.dataset_path}")
    else:
        logger.info(f"HuggingFace dataset: wikitext/{args.subset_name} ({args.split})")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of runs per level: {args.num_runs}")

    tester = RandomNeuronMaskingTester(
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
        logger=logger
    )

    texts = tester.load_wikitext_dataset(
        dataset_path=args.dataset_path,
        dataset_name='wikitext',
        subset_name=args.subset_name,
        split=args.split,
        max_samples=args.max_samples
    )

    masking_levels = [int(x.strip()) for x in args.masking_levels.split(',')]

    results = tester.test_random_masking_with_stats(
        texts=texts,
        masking_levels=masking_levels,
        num_runs=args.num_runs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        output_file=args.output_file
    )
    
    logger.info("=== Testing Completed ===")
    logger.info(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()

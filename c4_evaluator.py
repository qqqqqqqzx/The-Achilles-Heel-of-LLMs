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
import gzip
import transformers

def setup_logging(log_file="c4_perplexity.log", log_level=logging.INFO):
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

class C4PerplexityTester:
    def __init__(self, model_path, neuron_file=None, device="auto", seed=42, logger=None):
        """
        Initialize C4 evaluator
        
        Args:
            model_path: Model path
            neuron_file: Path to neuron importance file (JSON), optional
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
        
        self.neurons = []
        if neuron_file:
            self.logger.info(f"Loading neuron importance data from: {neuron_file}")
            with open(neuron_file, 'r') as f:
                neuron_data = json.load(f)
            
            self.neurons = neuron_data.get('top_neurons', neuron_data.get('neurons', []))
            self.logger.info(f"Loaded {len(self.neurons)} neurons")
        else:
            self.logger.info("No neuron file provided, will only test baseline perplexity")
        
        self.masking_hooks = []
        self.current_masked_neurons = []
        
    def _load_model_multi_gpu(self, model_path):
        """Load model"""
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
    
    def load_c4_dataset(self, dataset_path, max_samples=None, min_text_length=50):
        """
        Load C4 dataset from JSON file
        
        Args:
            dataset_path: Path to C4 JSON file (can be .json or .json.gz)
            max_samples: Maximum number of samples to load (None for all)
            min_text_length: Minimum text length to include
            
        Returns:
            List of text samples
        """
        self.logger.info(f"Loading C4 dataset from: {dataset_path}")
        
        texts = []
        sample_count = 0
        
        try:
            if dataset_path.endswith('.gz'):
                file_opener = gzip.open
                mode = 'rt'
                self.logger.info("Loading compressed C4 file (.gz)")
            else:
                file_opener = open
                mode = 'r'
                self.logger.info("Loading uncompressed C4 file")
            
            with file_opener(dataset_path, mode, encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc="Loading C4 data")):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                            
                        data = json.loads(line)
                        text = data.get('text', '').strip()
                        
                        if text and len(text) >= min_text_length:
                            texts.append(text)
                            sample_count += 1
                            
                            if max_samples and sample_count >= max_samples:
                                break
                                
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON at line {line_num + 1}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing line {line_num + 1}: {e}")
                        continue
            
            self.logger.info(f"Loaded {len(texts)} text samples from C4 dataset")
            self.logger.info(f"Average text length: {np.mean([len(text) for text in texts]):.1f} characters")
            
            return texts
            
        except FileNotFoundError:
            self.logger.error(f"C4 dataset file not found: {dataset_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load C4 dataset: {e}")
            raise
    
    def _setup_masking_hooks(self, masked_neurons):
        """
        Set up hooks to mask specific neurons during forward pass
        
        Args:
            masked_neurons: List of neurons to mask
            
        Returns:
            Number of successfully registered hooks
        """
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
        """
        Apply neuron masking by setting up forward hooks
        
        Args:
            masked_neurons: List of neurons to mask
            
        Returns:
            Number of successfully masked neurons
        """
        if not masked_neurons:
            self._clear_masking_hooks()
            return 0
        
        hook_count = self._setup_masking_hooks(masked_neurons)
        self.logger.info(f"Applied masking to {len(masked_neurons)} neurons using {hook_count} hooks")
        return len(masked_neurons)
    
    def compute_perplexity_on_dataset(self, texts, max_length=512, batch_size=1, num_neurons_to_mask=0):
        """
        Compute perplexity on a dataset of texts
        
        Args:
            texts: List of text samples
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            num_neurons_to_mask: Number of top neurons to mask (0 for no masking)
            
        Returns:
            Dictionary with perplexity results
        """
        self.logger.info(f"Computing perplexity on {len(texts)} texts")
        self.logger.info(f"Max length: {max_length}, Batch size: {batch_size}")
        
        if num_neurons_to_mask > 0:
            self.logger.info(f"Masking top {num_neurons_to_mask} neurons")
            neurons_to_mask = self.neurons[:num_neurons_to_mask]
            self.apply_neuron_masking(neurons_to_mask)
        else:
            self.logger.info("No neuron masking applied")
            self._clear_masking_hooks()
        
        first_param = next(self.model.parameters())
        input_device = first_param.device
        
        total_loss = 0.0
        total_tokens = 0
        valid_samples = 0
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
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
            return {"error": "No valid tokens processed"}
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        results = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "valid_samples": valid_samples,
            "neurons_masked": num_neurons_to_mask
        }
        
        self.logger.info(f"Results: Perplexity = {perplexity:.4f}, Avg Loss = {avg_loss:.4f}")
        self.logger.info(f"Processed {valid_samples} samples, {total_tokens} tokens")
        
        return results
    
    def test_multiple_masking_levels(self, texts, masking_levels, max_length=512, batch_size=1, output_file="c4_results.json"):
        """
        Test perplexity with multiple neuron masking levels
        
        Args:
            texts: List of text samples
            masking_levels: List of neuron counts to mask
            max_length: Maximum sequence length
            batch_size: Batch size
            output_file: Output file name
            
        Returns:
            Dictionary with all results
        """
        self.logger.info("Testing multiple neuron masking levels")
        self.logger.info(f"Masking levels: {masking_levels}")
        
        all_results = {
            "model_info": {
                "model_path": getattr(self, 'model_path', 'unknown'),
                "num_gpus": self.num_gpus,
                "total_neurons": len(self.neurons)
            },
            "dataset_info": {
                "num_texts": len(texts),
                "max_length": max_length,
                "batch_size": batch_size,
                "dataset_type": "C4"
            },
            "results": []
        }
        
        for num_neurons in masking_levels:
            self.logger.info(f"\n=== Testing with {num_neurons} neurons masked ===")
            
            self._clear_masking_hooks()
            
            result = self.compute_perplexity_on_dataset(
                texts=texts,
                max_length=max_length,
                batch_size=batch_size,
                num_neurons_to_mask=num_neurons
            )
            
            result["masking_level"] = num_neurons
            all_results["results"].append(result)
            
            self.logger.info(f"Masking {num_neurons} neurons: Perplexity = {result.get('perplexity', 'N/A'):.4f}")
        
        self.logger.info(f"Saving all results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.info("\n=== Results Summary ===")
        for result in all_results["results"]:
            masked = result.get("masking_level", 0)
            ppl = result.get("perplexity", 0)
            self.logger.info(f"Masked {masked:5d} neurons: Perplexity = {ppl:8.4f}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="C4 Dataset Perplexity Tester with Neuron Masking")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--neuron_file", type=str, help="Neuron importance file (JSON)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to C4 dataset file (.json or .json.gz)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--min_text_length", type=int, default=50, help="Minimum text length to include")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_neurons_to_mask", type=int, default=0, help="Number of top neurons to mask")
    parser.add_argument("--masking_levels", type=str, help="Comma-separated list of neuron counts to test (e.g., '0,10,50,100')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="c4_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="c4_perplexity.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== C4 Dataset Perplexity Tester ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"C4 dataset: {args.dataset_path}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Min text length: {args.min_text_length}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Batch size: {args.batch_size}")
    
    tester = C4PerplexityTester(
        model_path=args.model_path,
        neuron_file=args.neuron_file,
        device=args.device,
        seed=args.seed,
        logger=logger
    )
    
    texts = tester.load_c4_dataset(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        min_text_length=args.min_text_length
    )
    
    if args.masking_levels:
        masking_levels = [int(x.strip()) for x in args.masking_levels.split(',')]
        results = tester.test_multiple_masking_levels(
            texts=texts,
            masking_levels=masking_levels,
            max_length=args.max_length,
            batch_size=args.batch_size,
            output_file=args.output_file
        )
    else:
        result = tester.compute_perplexity_on_dataset(
            texts=texts,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_neurons_to_mask=args.num_neurons_to_mask
        )
        
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    logger.info("=== Testing Completed ===")
    logger.info(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
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
import transformers

def setup_logging(log_file="neuron_masking.log", log_level=logging.INFO):
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

class MultiGPUNeuronMaskingAnalyzer:
    def __init__(self, model_path, neuron_file, device="auto", seed=42, logger=None):
        """
        Initialize neuron masking analyzer
        
        Args:
            model_path: Model path
            neuron_file: Path to neuron importance file (JSON)
            device: Computing device ("auto")
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
        
        self.logger.info(f"Loading neuron importance data from: {neuron_file}")
        with open(neuron_file, 'r') as f:
            neuron_data = json.load(f)
        
        self.neurons = neuron_data.get('top_neurons', neuron_data.get('neurons', []))
        self.logger.info(f"Loaded {len(self.neurons)} neurons")
        
        self.masking_hooks = []
        self.current_masked_neurons = []
        
    def _load_model_multi_gpu(self, model_path):
        """Load model"""
        self.logger.info(f"Loading model with multi-GPU: {model_path}")
        
        import transformers
        self.logger.info(f"Transformers version: {transformers.__version__}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto", 
            low_cpu_mem_usage=True,
            max_memory={i: "80GB" for i in range(self.num_gpus)} 
        )
        
        self.model.eval()
        self.logger.info("Multi-GPU model loading completed")
        
        self._print_device_map()
        
    def _print_device_map(self):
        """Print model device allocation"""
        self.logger.info("Model device allocation:")
        if hasattr(self.model, 'hf_device_map'):
            for layer_name, device in self.model.hf_device_map.items():
                self.logger.info(f"  {layer_name}: {device}")
        else:
            self.logger.info("  Device map not available")
    
    def _setup_masking_hooks(self, masked_neurons):
        """
        Set up hooks to mask critical neurons during forward pass
        
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
    
    def compute_perplexity(self, text):
        """
        Compute perplexity for given text
        
        Args:
            text: Input text
            
        Returns:
            Perplexity value
        """
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        first_param = next(self.model.parameters())
        input_device = first_param.device
        input_ids = encodings.input_ids.to(input_device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def progressive_masking_analysis(self, test_text, step_size=10, max_magnitude_increase=5, 
                                   output_file="masking_results.json"):
        """
        Progressive neuron masking analysis
        
        Args:
            test_text: Text to test perplexity on
            step_size: Number of neurons to mask in each step
            max_magnitude_increase: Stop when perplexity increases by this many orders of magnitude
            output_file: Output file name
            
        Returns:
            Results dictionary
        """
        self.logger.info("Starting multi-GPU progressive neuron masking analysis")
        self.logger.info(f"Test text: {test_text[:100]}...")
        self.logger.info(f"Step size: {step_size}")
        self.logger.info(f"Max magnitude increase: {max_magnitude_increase}")
        self.logger.info(f"Using {self.num_gpus} GPUs")
        
        self.logger.info("Computing baseline perplexity")
        baseline_perplexity = self.compute_perplexity(test_text)
        self.logger.info(f"Baseline perplexity: {baseline_perplexity:.4f}")
        
        results = {
            "baseline_perplexity": baseline_perplexity,
            "test_text": test_text,
            "step_size": step_size,
            "max_magnitude_increase": max_magnitude_increase,
            "num_gpus_used": self.num_gpus,
            "masking_steps": [],
            "final_masked_neurons": [],
            "stopping_reason": None
        }
        
        step = 1
        total_masked = 0
        
        first_param = next(self.model.parameters())
        input_device = first_param.device
        input_ids = self.tokenizer.encode(test_text, return_tensors="pt").to(input_device)
        
        while total_masked < len(self.neurons):
            neurons_to_mask = min(step * step_size, len(self.neurons))
            
            self.logger.info(f"Step {step}: masking {neurons_to_mask} neurons (total: {neurons_to_mask})")
            
            masked_neurons = self.neurons[:neurons_to_mask]
            
            masked_count = self.apply_neuron_masking(masked_neurons)
            
            masked_perplexity = self.compute_perplexity(test_text)
            
            magnitude_increase = math.log10(masked_perplexity / baseline_perplexity)
            
            self.logger.info(f"Masked perplexity: {masked_perplexity:.4f}")
            self.logger.info(f"Magnitude increase: {magnitude_increase:.4f}")
            
            step_result = {
                "step": step,
                "neurons_masked": neurons_to_mask,
                "masked_perplexity": masked_perplexity,
                "magnitude_increase": magnitude_increase,
                "perplexity_ratio": masked_perplexity / baseline_perplexity
            }
            results["masking_steps"].append(step_result)
            
            if magnitude_increase >= max_magnitude_increase:
                self.logger.info(f"Stopping: magnitude increase ({magnitude_increase:.4f}) >= threshold ({max_magnitude_increase})")
                results["stopping_reason"] = f"magnitude_increase_exceeded"
                results["final_masked_neurons"] = masked_neurons
                break
            
            total_masked = neurons_to_mask
            step += 1
            
            for gpu_id in range(self.num_gpus):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            
            if step > 1000:
                self.logger.warning("Stopping: maximum steps reached")
                results["stopping_reason"] = "max_steps_reached"
                results["final_masked_neurons"] = masked_neurons
                break
        
        if total_masked >= len(self.neurons):
            self.logger.info("Stopping: all neurons masked")
            results["stopping_reason"] = "all_neurons_masked"
            results["final_masked_neurons"] = self.neurons
        
        self._clear_masking_hooks()
        
        self.logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Multi-GPU progressive masking analysis completed")
        self.logger.info(f"Total steps: {len(results['masking_steps'])}")
        self.logger.info(f"Final masked neurons: {len(results['final_masked_neurons'])}")
        self.logger.info(f"Used {self.num_gpus} GPUs for computation")
        
        self.logger.info("\n=== Multi-GPU Masking Summary ===")
        for i, step_result in enumerate(results["masking_steps"]):
            self.logger.info(f"Step {step_result['step']}: {step_result['neurons_masked']} neurons, "
                           f"perplexity: {step_result['masked_perplexity']:.4f}, "
                           f"magnitude increase: {step_result['magnitude_increase']:.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Progressive neuron masking analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--neuron_file", type=str, required=True, help="Neuron importance file (JSON)")
    parser.add_argument("--test_text", type=str, required=True, help="Text to test perplexity on")
    parser.add_argument("--step_size", type=int, default=1, help="Number of neurons to mask per step (default: 1)")
    parser.add_argument("--max_magnitude", type=float, default=1.0, help="Maximum magnitude increase to allow (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_file", type=str, default="masking_results.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="neuron_masking.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device ('auto' for multi-GPU)")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== Multi-GPU Progressive Neuron Masking Analyzer ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Neuron file: {args.neuron_file}")
    logger.info(f"Test text: {args.test_text[:100]}...")
    logger.info(f"Step size: {args.step_size}")
    logger.info(f"Max magnitude increase: {args.max_magnitude}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    analyzer = MultiGPUNeuronMaskingAnalyzer(
        model_path=args.model_path,
        neuron_file=args.neuron_file,
        device=args.device,
        seed=args.seed,
        logger=logger
    )
    
    results = analyzer.progressive_masking_analysis(
        test_text=args.test_text,
        step_size=args.step_size,
        max_magnitude_increase=args.max_magnitude,
        output_file=args.output_file
    )
    
    logger.info("=== Multi-GPU Analysis Completed ===")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info(f"Log saved to: {args.log_file}")

if __name__ == "__main__":
    main()


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import argparse
import logging
from tqdm import tqdm
import random
from collections import defaultdict
import transformers

def setup_logging(log_file="neuron_analysis.log", log_level=logging.INFO):
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

class MultiGPUNeuronAnalyzer:
    def __init__(self, model_path, device="auto", seed=42, logger=None):
        """
        Initialize neuron activation analyzer
        
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
        
        self.layer_activations = {}
        self.hooks = []
        self.hooks_setup = False
        
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
    
    def _setup_hooks(self):
        """Set up hooks to collect neuron activations"""
        if self.hooks_setup:
            return
            
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.layer_activations[name] = {
                        'activation': output.detach(),
                        'device': output.device
                    }
            return hook
        
        module_count = 0
        
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in ['mlp', 'ffn', 'feed_forward', 'attention', 'self_attn']):
                if not list(module.children()) or isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                    self.hooks.append(module.register_forward_hook(get_activation(name)))
                    module_count += 1
        
        self.logger.info(f"Set up hooks for {module_count} key modules")
        self.hooks_setup = True
    
    def _sync_activations_across_gpus(self, activations_dict):
        synced_activations = {}
        
        for name, act_info in activations_dict.items():
            activation = act_info['activation']
            device = act_info['device']
            
            synced_activations[name] = activation.cpu()
            
        return synced_activations
    
    def analyze_activation_difference(self, input_text, noise_scale=0.1, num_samples=5, output_file="neuron_activation_diff.json"):
        """
        Analyze neuron activation differences
        
        Args:
            input_text: Input text
            noise_scale: Noise scale
            num_samples: Number of noise samples
            output_file: Output file name
            
        Returns:
            List of neurons sorted by activation difference
        """
        self.logger.info("Starting multi-GPU neuron activation difference analysis")
        
        self._setup_hooks()
        
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        first_param = next(self.model.parameters())
        input_device = first_param.device
        input_ids = input_ids.to(input_device)
        
        self.logger.info(f"Input text length: {len(input_ids[0])} tokens")
        self.logger.info(f"Input device: {input_device}")
        
        self.logger.info("Collecting baseline activations across GPUs")
        self.layer_activations.clear()
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        clean_activations = self._sync_activations_across_gpus(self.layer_activations)
        self.logger.info(f"Collected baseline activations from {len(clean_activations)} modules")
        
        aggregate_diffs = defaultdict(list)
        
        for i in range(num_samples):
            self.logger.info(f"Running noise experiment {i+1}/{num_samples}")
            
            with torch.no_grad():
                embedding_layer = self.model.get_input_embeddings()
                embeddings = embedding_layer(input_ids)
                
                noise = torch.randn_like(embeddings) * noise_scale
                noisy_embeddings = embeddings + noise
            
            self.layer_activations.clear()
            
            with torch.no_grad():
                _ = self.model(inputs_embeds=noisy_embeddings)
            
            noisy_activations = self._sync_activations_across_gpus(self.layer_activations)
            diff_count = 0
            
            for name, clean_act in clean_activations.items():
                if name in noisy_activations:
                    noisy_act = noisy_activations[name]
                    
                    if clean_act.shape == noisy_act.shape:
                        diff = (clean_act - noisy_act).abs()
                        aggregate_diffs[name].append(diff)
                        diff_count += 1
            
            self.logger.debug(f"Noise run {i+1} computed activation differences for {diff_count} modules")
            
            for gpu_id in range(self.num_gpus):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        
        self.logger.info(f"Successfully collected activation differences from {len(aggregate_diffs)} modules")
        
        neuron_importance = []
        
        self.logger.info("Computing neuron importance")
        for layer_name, diffs_list in tqdm(aggregate_diffs.items(), desc="Processing layers"):
            if diffs_list:
                avg_diff = torch.stack(diffs_list).mean(dim=0)
                
                if avg_diff.dim() >= 3: 
                    batch_size, seq_len, hidden_size = avg_diff.shape[0], avg_diff.shape[1], avg_diff.shape[-1]
                    
                    for batch_idx in range(batch_size):
                        for seq_idx in range(seq_len):
                            for hidden_idx in range(hidden_size):
                                importance = avg_diff[batch_idx, seq_idx, hidden_idx].item()
                                
                                if importance > 1e-6:
                                    neuron_importance.append({
                                        "layer_name": layer_name,
                                        "batch_idx": int(batch_idx),
                                        "seq_idx": int(seq_idx),
                                        "hidden_idx": int(hidden_idx),
                                        "activation_diff": float(importance)
                                    })
                
                elif avg_diff.dim() == 2: 
                    dim1_size, dim2_size = avg_diff.shape
                    
                    for i in range(dim1_size):
                        for j in range(dim2_size):
                            importance = avg_diff[i, j].item()
                            
                            if importance > 1e-6:
                                neuron_importance.append({
                                    "layer_name": layer_name,
                                    "dim1_idx": int(i),
                                    "dim2_idx": int(j),
                                    "activation_diff": float(importance)
                                })
                
                else: 
                    flat_diff = avg_diff.view(-1)
                    
                    for idx in range(flat_diff.size(0)):
                        importance = flat_diff[idx].item()
                        
                        if importance > 1e-6:
                            neuron_importance.append({
                                "layer_name": layer_name,
                                "flat_idx": int(idx),
                                "activation_diff": float(importance)
                            })
        
        self.logger.info(f"Found {len(neuron_importance)} meaningful neurons")
        
        self.logger.info("Sorting neurons by activation difference")
        sorted_neurons = sorted(neuron_importance, key=lambda x: x["activation_diff"], reverse=True)
        
        self.logger.info(f"Saving results to {output_file}")
        result_data = {
            "method": "multi_gpu_activation_difference_analysis",
            "input_text": input_text,
            "noise_scale": noise_scale,
            "num_samples": num_samples,
            "num_gpus_used": self.num_gpus,
            "total_neurons": len(sorted_neurons),
            "top_neurons": sorted_neurons[:10000] 
        }
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        self.logger.info("Multi-GPU analysis completed")
        self.logger.info(f"Total discovered neurons: {len(sorted_neurons)}")
        self.logger.info(f"Used {self.num_gpus} GPUs for computation")
        
        self.logger.info("Top 10 neurons with highest activation difference:")
        for i, neuron in enumerate(sorted_neurons[:10]):
            neuron_info = f"Rank {i+1}: Layer {neuron['layer_name']}, Activation diff: {neuron['activation_diff']:.6f}"
            for k, v in neuron.items():
                if k not in ['layer_name', 'activation_diff']:
                    neuron_info += f", {k}: {v}"
            self.logger.info(neuron_info)
        
        return sorted_neurons

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Neuron Activation Analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--input_text", type=str, required=True, help="Input text")
    parser.add_argument("--noise_scale", type=float, default=5, help="Noise scale (default: 5)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of noise samples (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_file", type=str, default="neuron_activation_diff.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="neuron_analysis.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device (use 'auto' for multi-GPU)")
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== Multi-GPU Neuron Activation Difference Analyzer ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Input text: {args.input_text[:100]}...")
    logger.info(f"Noise scale: {args.noise_scale}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    analyzer = MultiGPUNeuronAnalyzer(
        model_path=args.model_path, 
        device=args.device, 
        seed=args.seed, 
        logger=logger
    )
    
    neurons = analyzer.analyze_activation_difference(
        input_text=args.input_text,
        noise_scale=args.noise_scale,
        num_samples=args.num_samples,
        output_file=args.output_file
    )
    
    logger.info("=== Multi-GPU Analysis Completed ===")
    logger.info(f"Discovered {len(neurons)} neurons")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info(f"Log saved to: {args.log_file}")

if __name__ == "__main__":
    main()
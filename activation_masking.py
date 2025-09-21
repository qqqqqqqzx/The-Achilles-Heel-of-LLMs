import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json
import argparse
import logging
from tqdm import tqdm
import time
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

class MultiGPUActivationValueAnalyzer:
    def __init__(self, model_path, device="auto", seed=42, logger=None):
        """
        Initialize multi-GPU neuron activation value analyzer
        
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
        # 存储钩子
        self.hooks = []
        self.hooks_setup = False
        
    def _load_model_multi_gpu(self, model_path):
        """Load model with multi-GPU support"""
        self.logger.info(f"Loading model with multi-GPU: {model_path}")
      
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
    
    def analyze_activation_values(self, input_text, output_file="neuron_activation_values.json"):
        """
        Analyze neuron activation values with multi-GPU support
        
        Args:
            input_text: Input text
            output_file: Output file name
            
        Returns:
            List of neurons sorted by activation value (descending)
        """
        self.logger.info("Starting multi-GPU neuron activation value analysis")
        self._setup_hooks()

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        first_param = next(self.model.parameters())
        input_device = first_param.device
        input_ids = input_ids.to(input_device)
        
        self.logger.info(f"Input text length: {len(input_ids[0])} tokens")
        self.logger.info(f"Input device: {input_device}")
        self.logger.info("Collecting activations across GPUs")
        self.layer_activations.clear()
        
        with torch.no_grad():
            _ = self.model(input_ids)
        activations = self._sync_activations_across_gpus(self.layer_activations)
        self.logger.info(f"Collected activations from {len(activations)} modules")
        
        neuron_importance = []
        
        self.logger.info("Processing neuron activation values")
        for layer_name, activation in tqdm(activations.items(), desc="Processing layers"):
            if activation.dim() >= 3: 
                batch_size, seq_len, hidden_size = activation.shape[0], activation.shape[1], activation.shape[-1]
                
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        for hidden_idx in range(hidden_size):
                            value = activation[batch_idx, seq_idx, hidden_idx].item()
                            if abs(value) > 1e-6:
                                neuron_importance.append({
                                    "layer_name": layer_name,
                                    "batch_idx": int(batch_idx),
                                    "seq_idx": int(seq_idx),
                                    "hidden_idx": int(hidden_idx),
                                    "activation_diff": float(value)
                                })
            
            elif activation.dim() == 2:  
                dim1_size, dim2_size = activation.shape
                
                for i in range(dim1_size):
                    for j in range(dim2_size):
                        value = activation[i, j].item()
                        
                        if abs(value) > 1e-6:
                            neuron_importance.append({
                                "layer_name": layer_name,
                                "dim1_idx": int(i),
                                "dim2_idx": int(j),
                                "activation_diff": float(value)
                            })
            
            else:  
                flat_activation = activation.view(-1)
                
                for idx in range(flat_activation.size(0)):
                    value = flat_activation[idx].item()
                    
                    if abs(value) > 1e-6:
                        neuron_importance.append({
                            "layer_name": layer_name,
                            "flat_idx": int(idx),
                            "activation_diff": float(value)
                        })
        
        self.logger.info(f"Found {len(neuron_importance)} neurons with meaningful activation values")

        self.logger.info("Sorting neurons by activation value (descending)")
        sorted_neurons = sorted(neuron_importance, key=lambda x: x["activation_diff"], reverse=True)

        self.logger.info(f"Saving results to {output_file}")
        result_data = {
            "method": "multi_gpu_activation_value_analysis",
            "input_text": input_text,
            "noise_scale": None,
            "num_samples": 1,
            "num_gpus_used": self.num_gpus,
            "total_neurons": len(sorted_neurons),
            "top_neurons": sorted_neurons[:10000]
        }
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        self.logger.info("Multi-GPU activation value analysis completed")
        self.logger.info(f"Total discovered neurons: {len(sorted_neurons)}")
        self.logger.info(f"Used {self.num_gpus} GPUs for computation")

        self.logger.info("Top 10 neurons with highest activation values:")
        for i, neuron in enumerate(sorted_neurons[:10]):
            neuron_info = f"Rank {i+1}: Layer {neuron['layer_name']}, Activation value: {neuron['activation_diff']:.6f}"
            for k, v in neuron.items():
                if k not in ['layer_name', 'activation_diff']:
                    neuron_info += f", {k}: {v}"
            self.logger.info(neuron_info)
        
        return sorted_neurons

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Neuron Activation Value Analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--input_text", type=str, required=True, help="Input text")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_file", type=str, default="neuron_activation_values.json", help="Output file name")
    parser.add_argument("--log_file", type=str, default="neuron_value_analysis.log", help="Log file name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--device", type=str, default="auto", help="Computing device (use 'auto' for multi-GPU)")
    
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(args.log_file, log_level)
    
    logger.info("=== Multi-GPU Neuron Activation Value Analyzer ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Input text: {args.input_text[:100]}...")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    analyzer = MultiGPUActivationValueAnalyzer(
        model_path=args.model_path, 
        device=args.device, 
        seed=args.seed, 
        logger=logger
    )

    neurons = analyzer.analyze_activation_values(
        input_text=args.input_text,
        output_file=args.output_file
    )
    
    logger.info("=== Multi-GPU Activation Value Analysis Completed ===")
    logger.info(f"Discovered {len(neurons)} neurons")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info(f"Log saved to: {args.log_file}")

if __name__ == "__main__":
    main()
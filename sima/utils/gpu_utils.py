"""
GPU Utilities for optimizing performance on EC2 G5 instances
"""
import os
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def setup_gpu_environment(config: Dict[str, Any]) -> bool:
    """
    Configure the environment for optimal GPU usage on EC2 G5 instances
    
    Args:
        config: Configuration dictionary containing GPU settings
        
    Returns:
        True if GPU is available and configured, False otherwise
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Running on CPU.")
        return False
    
    # Get GPU device to use
    device_id = config.get("gpu_device", 0)
    if device_id >= torch.cuda.device_count():
        logger.warning(f"Requested GPU {device_id} not available. Using GPU 0 instead.")
        device_id = 0
        
    # Set device
    torch.cuda.set_device(device_id)
    
    # Log GPU info
    device_name = torch.cuda.get_device_name(device_id)
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
    logger.info(f"Using GPU {device_id}: {device_name} with {total_memory:.2f} GB memory")
    
    # Set precision based on config
    precision = config.get("sima", {}).get("model_precision", "float32")
    if precision == "float16" and torch.cuda.is_available():
        logger.info("Using mixed precision (float16) for faster processing")
        
    # Enable TF32 for NVIDIA A10G and newer GPUs (like in G5 instances)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True
    
    # Set environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    logger.info("GPU environment successfully configured")
    return True

def to_gpu(data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Move data to GPU memory with appropriate type conversion
    
    Args:
        data: Input data as torch.Tensor or numpy array
        
    Returns:
        Data as torch.Tensor on GPU
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    
    if torch.cuda.is_available():
        return data.cuda()
    return data

def batch_process(process_func, items, batch_size=4):
    """
    Process items in batches to maximize GPU utilization
    
    Args:
        process_func: Function to process each batch
        items: List of items to process
        batch_size: Size of each batch
        
    Returns:
        List of processed results
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    return results

def optimize_memory_usage():
    """
    Optimize GPU memory usage by clearing cache and garbage collection
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Force garbage collection
    import gc
    gc.collect()

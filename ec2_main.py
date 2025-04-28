"""
EC2 G5-optimized entry point for the bug detection system
This version includes GPU optimizations and cloud-specific configurations
"""
import os
import json
import logging
import time
import argparse
import torch
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the main pipeline
from main import GameTestingPipeline

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available! Running on CPU only.")
        return False

    # Log GPU information
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} CUDA-compatible GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        logger.info(f"GPU {i}: {gpu_name} with {memory_total:.2f} GB memory")
    
    # Set CUDA device
    torch.cuda.set_device(0)
    
    # Optimize CUDA operations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Configure PyTorch default tensor type based on GPU availability
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    logger.info("GPU configuration completed")
    return True

def setup_unity_integration(config: Dict[str, Any]):
    """Set up integration with Unity API as per memory recommendations"""
    if not config.get("use_unity_api", False):
        logger.info("Unity API integration is disabled in config")
        return
    
    logger.info("Setting up Unity API integration")
    # This is a placeholder for the actual Unity API integration code
    # In a real implementation, this would initialize connections to Unity's ML-Agents
    # or a custom C# script that exposes game objects and states
    
    # Example placeholder:
    unity_config = config.get("unity", {})
    host = unity_config.get("api_host", "localhost")
    port = unity_config.get("api_port", 8080)
    
    logger.info(f"Unity API integration configured at {host}:{port}")
    # Actual implementation would be added here

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="EC2 G5-optimized Bug Detection System")
    parser.add_argument("--config", default="config_ec2_g5.json", help="Path to configuration file")
    parser.add_argument("--game-desc", default="3D platformer game where players navigate obstacles", 
                        help="Description of the game being tested")
    parser.add_argument("--feature-desc", default="Character movement and collision detection", 
                        help="Description of the feature being tested")
    parser.add_argument("--num-tasks", type=int, default=10, 
                        help="Number of test tasks to generate")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set the OpenAI API key from environment if not in config
    if not config.get("openai_api_key"):
        config["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
        if not config["openai_api_key"]:
            logger.error("OpenAI API key not found in config or environment variables")
            return
    
    # Configure GPU
    has_gpu = configure_gpu()
    if has_gpu:
        logger.info("Using GPU acceleration for processing")
        config["use_gpu"] = True
    else:
        logger.info("GPU acceleration not available, using CPU")
        config["use_gpu"] = False
    
    # Setup Unity API integration
    setup_unity_integration(config)
    
    # Initialize the pipeline with our config
    logger.info("Initializing Game Testing Pipeline with EC2 G5 optimizations")
    pipeline = GameTestingPipeline(config)
    
    # Run the pipeline
    logger.info(f"Starting pipeline with {args.num_tasks} test tasks")
    try:
        pipeline.run_pipeline(
            game_description=args.game_desc,
            feature_description=args.feature_desc,
            num_tasks=args.num_tasks
        )
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()

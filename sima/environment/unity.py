# sima/environment/unity.py
import os
import time
import logging
import subprocess
from typing import Dict, Any, Optional

class UnityEnvironment:
    """
    Unity integration for the SIMA agent that manages running Unity games.
    This component provides game environment management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Unity environment with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.executable_path = config.get("executable_path", "")
        self.arguments = config.get("arguments", [])
        self.process = None
        
        if not self.executable_path:
            self.logger.warning("No Unity executable path provided")
        elif not os.path.exists(self.executable_path):
            self.logger.error(f"Unity executable not found at {self.executable_path}")
        else:
            self.logger.info(f"Unity environment configured with executable: {self.executable_path}")
    
    def start(self) -> bool:
        """
        Start the Unity game
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.executable_path or not os.path.exists(self.executable_path):
            self.logger.error("Cannot start Unity game: Invalid executable path")
            return False
        
        if self.process is not None and self.process.poll() is None:
            self.logger.warning("Unity game is already running")
            return True
        
        try:
            self.logger.info(f"Starting Unity game: {self.executable_path}")
            
            # Start the process
            self.process = subprocess.Popen(
                [self.executable_path] + self.arguments,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for the game to start
            time.sleep(self.config.get("start_delay", 5))
            
            # Check if the process is still running
            if self.process.poll() is not None:
                self.logger.error("Unity game failed to start")
                return False
            
            self.logger.info("Unity game started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting Unity game: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the Unity game
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self.process is None:
            self.logger.info("No Unity game process to stop")
            return True
        
        try:
            self.logger.info("Stopping Unity game")
            
            # Try graceful termination
            self.process.terminate()
            
            # Wait for the process to terminate
            time.sleep(2)
            
            # Force kill if still running
            if self.process.poll() is None:
                self.process.kill()
                time.sleep(1)
            
            self.process = None
            self.logger.info("Unity game stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping Unity game: {str(e)}")
            return False
    
    def is_running(self) -> bool:
        """Check if the Unity game is running"""
        return self.process is not None and self.process.poll() is None

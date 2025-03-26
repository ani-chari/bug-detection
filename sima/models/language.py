# sima/models/language.py
import torch
import torch.nn as nn
from typing import Dict, Any
import logging
from transformers import AutoModel, AutoTokenizer

class LanguageModel(nn.Module):
    """
    Language understanding model for SIMA that processes natural language instructions.
    Uses pre-trained transformers to encode instructions into embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize language model with configuration"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.model_name = config.get("model_name", "sentence-transformers/all-mpnet-base-v2")
        
        try:
            # Load pre-trained language model
            self.logger.info(f"Loading language model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            
            # Add projection layer
            self.projection = nn.Sequential(
                nn.Linear(self.embedding_dim, 512),
                nn.GELU(),
                nn.Linear(512, 512)
            )
            
            self.logger.info("Language model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading language model: {str(e)}")
            # Create fallback embedding model
            self.tokenizer = None
            self.model = self._create_fallback_model()
            self.embedding_dim = 512
            self.projection = nn.Identity()
    
    def _create_fallback_model(self) -> nn.Module:
        """Create a fallback embedding model if transformer loading fails"""
        self.logger.info("Creating fallback language model")
        vocab_size = 30000
        embedding_dim = 512
        return nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, 512)
        )
    
    def _tokenize_fallback(self, text: str) -> torch.Tensor:
        """Simple tokenization for fallback model"""
        # Basic character-level encoding
        chars = list(text.lower())
        ids = [ord(c) % 30000 for c in chars]
        return torch.tensor([ids])
    
    def forward(self, instruction: str) -> torch.Tensor:
        """
        Process instruction and return embedding
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            Instruction embedding tensor
        """
        with torch.no_grad():
            if self.tokenizer is not None:
                # Process with transformer
                inputs = self.tokenizer(
                    instruction, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=128
                )
                
                # Move to correct device
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                # Use [CLS] token or mean pooling
                if hasattr(outputs, "pooler_output"):
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            else:
                # Process with fallback
                tokens = self._tokenize_fallback(instruction)
                device = next(self.model.parameters()).device
                tokens = tokens.to(device)
                embeddings = self.model(tokens)
        
        # Project to final dimension
        embedding = self.projection(embeddings)
        return embedding

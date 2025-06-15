"""Base interface for all AI models."""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseModel(ABC):
    """Abstract base class for all AI models."""
    
    @abstractmethod
    def load(self, model_name: str, device: str = None) -> None:
        """Load the model with the given name."""
        pass
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process input data using the loaded model."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Return a dictionary of available models and their metadata."""
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Return information about the currently loaded model."""
        pass

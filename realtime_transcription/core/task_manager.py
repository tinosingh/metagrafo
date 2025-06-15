"""Manages different AI tasks and their respective models."""
from typing import Dict, Any, Optional, Type
from realtime_transcription.models.base_model import BaseModel
from realtime_transcription.models.whisper_model import WhisperModel

class TaskManager:
    """Manages different AI tasks and their respective models."""
    
    TASK_MODELS = {
        "transcription": WhisperModel,
        # Will add more task types (spellcheck, summarization) later
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.tasks: Dict[str, BaseModel] = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize models for all supported tasks."""
        for task_name, model_class in self.TASK_MODELS.items():
            self.tasks[task_name] = model_class()
    
    def load_model(self, task_name: str, model_name: str, **kwargs) -> bool:
        """Load a specific model for a task."""
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
        return self.tasks[task_name].load(model_name, **kwargs)
    
    def process(self, task_name: str, input_data: Any, **kwargs) -> Any:
        """Process input using the appropriate model for the task."""
        if task_name not in self.tasks:
            raise ValueError(f"No model loaded for task: {task_name}")
        return self.tasks[task_name].process(input_data, **kwargs)
    
    def get_available_models(self, task_name: str = None) -> Dict[str, Any]:
        """Get available models for a task or all tasks."""
        if task_name:
            if task_name not in self.tasks:
                raise ValueError(f"Unknown task: {task_name}")
            return {
                task_name: self.tasks[task_name].get_available_models()
            }
        return {
            task: model.get_available_models() 
            for task, model in self.tasks.items()
        }
    
    def get_model_info(self, task_name: str) -> Dict[str, Any]:
        """Get information about the currently loaded model for a task."""
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
        return self.tasks[task_name].model_info
    
    def add_task(self, task_name: str, model_class: Type[BaseModel]) -> None:
        """Add support for a new task type."""
        if not issubclass(model_class, BaseModel):
            raise TypeError("Model class must inherit from BaseModel")
        self.TASK_MODELS[task_name] = model_class
        
    def register_model(self, model_type: str, task_name: str, model_instance: BaseModel) -> None:
        """Register a pre-initialized model instance for a task.
        
        Args:
            model_type: Type of the model (e.g., 'whisper')
            task_name: Name of the task this model handles
            model_instance: Pre-initialized model instance
            
        Note:
            This allows using custom model instances instead of the default ones.
        """
        if not isinstance(model_instance, BaseModel):
            raise TypeError("Model instance must inherit from BaseModel")
            
        if task_name not in self.TASK_MODELS:
            self.TASK_MODELS[task_name] = type(model_instance)
            
        self.tasks[task_name] = model_instance

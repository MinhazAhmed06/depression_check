from pydantic import BaseModel, Field
from typing import Dict, Any

class LinearModelNameConfig(BaseModel):
    selected_model: str = 'LinearRegression'
    params: Dict[str, Any] = Field(default_factory=dict)
    # Use a dictionary to hold model-specific parameters that will be passed as kwargs
    
class ModelNameConfig(BaseModel):
    selected_model: str = 'LogisticRegression'
    params: Dict[str, Any] = Field(default_factory=dict)
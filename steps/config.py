from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    selected_model: str = 'LinearRegression'
    random_state: int = 42
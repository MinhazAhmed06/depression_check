import json
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig

from pipelines.utils import get_data_for_test 

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    min_f1s: float = 0.8
    
@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    f1s: float,
    config: DeploymentTriggerConfig,
):
    return f1s > config.min_f1s

class MLFlowDeploymentLoaderStepParameters(BaseModel):
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = 'model',
) -> MLFlowDeploymentService:
    
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        model_name = model_name,
        running = running,
    )
    
    if not existing_services:
        raise RuntimeError(
            f'No MLflow deployment service found for pipeline {pipeline_name}'
            f'step {pipeline_step_name} and model {model_name}.'
            f'Pipeline for the {model_name} model is currently running.'
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    service.start(timeout=10)
    data = json.loads(data)
    data.pop('columns')
    data.pop('index')
    columns_for_df = ['Gender','Age','Academic Pressure','CGPA','Study Satisfaction','Sleep Duration','Dietary Habits',
                      'Have you ever had suicidal thoughts ?','Study Hours','Financial Stress',
                      'Family History of Mental Illness']
    df = pd.DataFrame(data['data'], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    print(f"df type: {type(df)}")
    print(f"df.values type: {type(df.values)}")
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings = {"docker":docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_f1s: float = 0.8,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model_config = ModelNameConfig()
    model = train_model(X_train, y_train, model_config)
    precision, recall, f1score = evaluate_model(model, X_test, y_test)
    dtconfig = DeploymentTriggerConfig(min_f1s = min_f1s)
    deploy_decision = deployment_trigger(f1score, dtconfig)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deploy_decision,
        workers = workers,
        timeout = timeout,
    )

@pipeline(enable_cache=False, settings={'docker': docker_settings})
def inference_pipeline(pipeline_name:str, pipeline_step_name:str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        running = False
    )
    prediction = predictor(service=service, data=data)
    return prediction
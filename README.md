# depression_check
Hosting a machine learning model on the web to be used by minimum amount of data given of a student. A model in the backend would try to roughly predict if the student is depressed or not.


zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set

mlflow ui --backend-store-uri "[given_uri]"

python run_deployment.py -config deploy
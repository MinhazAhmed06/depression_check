from zenml.client import Client
from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    client = Client()
    experiment_tracker = client.active_stack.experiment_tracker
    tracking_uri = experiment_tracker.get_tracking_uri()
    print(tracking_uri)
    train_pipeline(data_path='data/student_depression_dataset.csv')
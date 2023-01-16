import json
import os


def load_model_task_data(task_name, model_name):
    data_dir = os.environ["HELM_DATA_DIR"]
    with open(os.path.join(data_dir, task_name, f"{model_name}.json")) as f:
        return json.load(f)


def load_tasks_data():
    data_dir = os.environ["HELM_DATA_DIR"]
    with open(os.path.join(data_dir, "tasks.json")) as f:
        return json.load(f)

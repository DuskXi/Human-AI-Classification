import json
import os


class Config(dict):
    def __init__(self, **kwargs):
        config_data = {
            "human_path": kwargs.get("human_path", "data/human"),
            "ai_path": kwargs.get("ai_path", "data/ai"),
            "train_data_percentage": kwargs.get("train_data_percentage", 0.8),
            "epochs": kwargs.get("epochs", 10),
            "output_path": kwargs.get("output_path", "output"),
            "random_data": kwargs.get("random_data", False),
            "batch_size": kwargs.get("batch_size", 32),
            "shuffle": kwargs.get("shuffle", True),
            "num_workers": kwargs.get("num_workers", 4),
            "learning_rate": kwargs.get("learning_rate", 0.001),
            "test_model_per_steps": kwargs.get("test_model_per_steps", 250),
            "show_test_results": kwargs.get("show_test_results", True),
            "model_name": kwargs.get("model_name", "vgg16"),
            "optimizer": kwargs.get("optimizer", "adam"),
            "dynamic_lr": kwargs.get("dynamic_lr", True),
            "dynamic_lr_gamma": kwargs.get("dynamic_lr_gamma", 0.05),
            "dynamic_lr_step_size": kwargs.get("dynamic_lr_step_size", 2),
        }
        for key, value in config_data.items():
            self.__setattr__(key, value)
        super().__init__(**config_data)

    @staticmethod
    def from_file(path):
        if not os.path.exists(path):
            with open(path, "w", encoding="UTF-8") as f:
                json.dump(Config(), f, indent=4)
        with open(path, "r", encoding="UTF-8") as f:
            config_data = json.load(f)
            return Config(**config_data).save(path)

    def save(self, path):
        with open(path, "w", encoding="UTF-8") as f:
            json.dump(self, f, indent=4)
        return self

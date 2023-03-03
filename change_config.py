from config import Config
import os


def main():
    path = "./task"
    config_names = [x for x in os.listdir(path) if x.endswith(".json")]
    for config_name in config_names:
        config = Config.from_file(os.path.join(path, config_name))
        # change config
        config["dynamic_lr_step_size"] = 3
        config.save(os.path.join(path, config_name))


if __name__ == "__main__":
    main()

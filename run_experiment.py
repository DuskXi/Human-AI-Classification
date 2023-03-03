import os

import argparse
from train import Trainer
from config import Config
from loguru import logger
from rich.pretty import pprint


def main(args):
    configs_dir = args.task
    logger.info(f"Load configs from {configs_dir} folder...")
    configs_name = [x for x in os.listdir(configs_dir) if x.endswith(".json")]
    configs_name.sort()
    configs = []
    for config in configs_name:
        configs.append(Config.from_file(os.path.join(configs_dir, config)))
    logger.info(f"Configs loaded: ({len(configs)})")
    for i, config in enumerate(configs):
        logger.info(f"Use config: {os.path.join(configs_dir, configs_name[i])}")
        pprint(config)
        trainer = Trainer(config)
        logger.info("Start training...")
        try:
            if not trainer.current_epoch >= config["epochs"]:
                trainer.train(config["epochs"])
        except Exception as e:
            logger.error("Training failed")
            logger.error(e)
        logger.info("Training finished")
        trainer.cleanup()
    logger.info("All training finished")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="./task", dest="task", help="Folder that have task config files")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

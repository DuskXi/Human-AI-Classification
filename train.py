import json
import math
import os

import torch
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchsummary import summary
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import nvidia_smi
from config import Config
from HADataset import HADataset
from rich.progress import Progress

from loguru import logger
from rich.logging import RichHandler

from model import ModelLoader
from torch.utils.tensorboard import SummaryWriter

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class Trainer:
    def __init__(self, config: Config, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        model_loader = ModelLoader()
        self.config = config
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.num_workers = config["num_workers"]
        self.learning_rate = config["learning_rate"]
        self.output_path = config["output_path"]
        self.test_model_per_steps = config["test_model_per_steps"]
        self.show_test_results = config["show_test_results"]
        self.model_name = config["model_name"]
        self.dynamic_lr = config["dynamic_lr"]
        self.dynamic_lr_gamma = config["dynamic_lr_gamma"]
        self.dynamic_lr_step_size = config["dynamic_lr_step_size"]
        self.model = model_loader.create_model(self.model_name)
        self.model.to(device)
        self.device = device
        self.dataset = HADataset(config, device, test_data=False)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_dataset = HADataset(config, device, test_data=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.criterion = nn.CrossEntropyLoss()
        if config["optimizer"].lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif config["optimizer"].lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError("Optimizer not supported")
        self.loss_list = []
        self.epoch_loss = []
        self.test_loss = []
        self.current_epoch = 0
        self.current_step = 0
        self.last_save_model_name = None
        output_folder_template = f"{self.model_name}-lr={self.learning_rate}-optimizer={config['optimizer']}" + (f"-dynamic_lr_gamma={self.dynamic_lr_gamma}" if self.dynamic_lr else "")
        if not self.output_path.endswith(output_folder_template) and not self.output_path.endswith(output_folder_template + "/"):
            self.output_path = os.path.join(self.output_path, output_folder_template)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.training_stats = TrainingStats.load(os.path.join(self.output_path, "training_stats.json"))
        if self.training_stats is not None:
            logger.info("Load training stats from previous training")
            self.loss_list = self.training_stats.loss_list
            self.epoch_loss = self.training_stats.epoch_loss
            self.test_loss = self.training_stats.test_loss
            self.current_epoch = self.training_stats.current_epoch + 1
            self.current_step = self.training_stats.current_step
            self.last_save_model_name = self.training_stats.last_model
            # load model dict
            if self.last_save_model_name is not None:
                logger.info(f"Load model weight from {os.path.join(self.output_path, self.last_save_model_name)}")
                self.model.load_state_dict(torch.load(os.path.join(self.output_path, self.last_save_model_name)))
            if config["optimizer"] == "adam" and os.path.exists(os.path.join(self.output_path, "adam_stats.pt")):
                logger.info(f"Load optimizer weight from {os.path.join(self.output_path, 'adam_stats.pt')}")
                self.optimizer.load_state_dict(torch.load(os.path.join(self.output_path, "adam_stats.pt")))

        self.writer = SummaryWriter(os.path.join(self.output_path, "tensorboard"))
        if self.dynamic_lr:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.dynamic_lr_step_size, gamma=self.dynamic_lr_gamma)
            for _ in range(self.current_epoch):
                self.scheduler.step()
            # self.scheduler.last_epoch = self.current_step

    def epoch(self, epoch_index, start_steps, progress):
        self.model.train()
        self.model.to(self.device)
        logger.info(f"Training for epoch {epoch_index}")
        task = progress.add_task("[red]Training...", total=len(self.dataloader))
        for image, label in self.dataloader:
            batch_size = image.shape[0]
            # prediction
            pred = self.model(image)

            # loss
            loss = self.criterion(pred, label)
            self.writer.add_scalar("Step/Loss/train", loss, self.current_step)

            # backward
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer.add_scalar("Step/Learning rate", self.optimizer.param_groups[0]["lr"], self.current_step)
            self.writer.add_scalar("Step/Gpu Power", float(nvidia_smi.get_gpu_power()[0]), self.current_step)
            self.writer.add_scalar("Step/Gpu Use", float(nvidia_smi.get_gpu_usage()), self.current_step)
            # update progress and show loss
            progress.update(task, advance=1, description=f"[red]Training in epoch {epoch_index}... Loss: {loss.item():.4f}, step: {self.current_step}")
            self.loss_list[epoch_index].append(loss.item())
            step_prev = self.current_step
            self.current_step += image.shape[0]
            if self.current_step // self.test_model_per_steps > step_prev // self.test_model_per_steps:
                avg_loss, test_results, accuracy = self.test(epoch_index)
                if self.show_test_results:
                    self.show_test_results_plt(test_results)
                self.test_loss.append((self.current_step, avg_loss))
        logger.info(f"Finish training for epoch {epoch_index}")

    def train(self, epochs):
        if self.current_epoch != 0:
            logger.info(f"Continue training at {self.current_epoch} epochs / {self.current_step} steps")
        else:
            logger.info(f"Start training for total {self.current_epoch} epochs")
        with Progress() as progress:
            task = progress.add_task("[green]Training...", total=epochs)
            self.loss_list.append([])
            if self.current_epoch > 0:
                progress.update(task, advance=self.current_epoch, description=f"[green]Training... Epoch: {self.current_epoch}")
            for epoch_index in range(self.current_epoch, epochs):
                self.epoch(epoch_index, epoch_index * len(self.dataloader), progress)
                # calculate avg loss
                avg_loss = sum(self.loss_list[epoch_index]) / len(self.loss_list[epoch_index])
                self.epoch_loss.append(avg_loss)
                self.writer.add_scalar("Epoch/Loss/train", avg_loss, self.current_step)
                test_avg_loss, test_results, test_accuracy = self.test(epoch_index)
                self.writer.add_scalar("Epoch/Loss/test", test_avg_loss, epoch_index)
                self.writer.add_scalar("Epoch/Accuracy/test", test_accuracy, epoch_index)
                progress.update(task, advance=1, description=f"[green]Training... Epoch avg loss: {avg_loss:.4f}")
                self.save_weights()
                self.save_training_stats()
                if self.dynamic_lr:
                    self.scheduler.step()
                if self.config["optimizer"] == "adam" and self.dynamic_lr:
                    self.save_adam_stats()
                logger.info(f"Epoch {epoch_index} avg loss: {avg_loss:.4f}")
                logger.info(f"Epoch {epoch_index} lr: {self.optimizer.param_groups[0]['lr']}")
                self.current_epoch += 1
                self.loss_list.append([])
        self.summary_training()

    def save_training_stats(self):
        self.training_stats = TrainingStats(current_epoch=self.current_epoch, current_step=self.current_step, loss_list=self.loss_list, epoch_loss=self.epoch_loss, test_loss=self.test_loss,
                                            last_model=self.last_save_model_name)
        self.training_stats.save(os.path.join(self.output_path, "training_stats.json"))

    def save_adam_stats(self):
        stats_dict = self.optimizer.state_dict()
        torch.save(stats_dict, os.path.join(self.output_path, "adam_stats.pt"))

    def test(self, epoch):
        self.model.eval()
        loss_list = []
        test_results = []
        with torch.no_grad():
            for image, label in self.test_dataloader:
                pred = self.model(image)
                loss = self.criterion(pred, label)
                loss_list.append(loss.item())
                for i in range(image.shape[0]):
                    test_results.append((image[i], pred[i], label[i]))
        accuracy = (sum([1 if (pred[0] > pred[1]) == (label[0] > label[1]) else 0 for image, pred, label in test_results]) / len(test_results)) * 100
        avg_loss = sum(loss_list) / len(loss_list)
        self.writer.add_scalar("Step/Loss/test", avg_loss, self.current_step)
        self.writer.add_scalar("Step/Accuracy/test", accuracy, self.current_step)
        logger.info(f"Test avg loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}%")
        return avg_loss, test_results, accuracy

    @staticmethod
    def show_test_results_plt(test_results):
        transform = transforms.ToPILImage()
        inv_normalize = transforms.Normalize(
            mean=[0.5 / 0.5, 0.5 / 0.5, 0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )
        for image, pred, label in test_results:
            image = transform(inv_normalize(image))
            # show image by plt
            pred_result = "human" if pred[0] > pred[1] else "ai"
            label_result = "human" if label[0] > label[1] else "ai"
            is_success = "success" if pred_result == label_result else "fail"
            plt.title(f"Prediction: {pred_result}, Label: {label_result}, {is_success}")
            plt.imshow(image)
            plt.show()

    def save_weights(self, path=None):
        if path is None:
            path = self.output_path
        model_name = f"model_{self.current_epoch}_{self.current_step}.pth"
        logger.info(f"Saving model to {os.path.join(path, model_name)}....")
        torch.save(self.model.to("cpu").state_dict(), os.path.join(path, model_name))
        self.last_save_model_name = model_name
        logger.info(f"Model saved")

    def summary_training(self):
        # show loss graph
        plt.title("Step-Loss")
        plt.plot(self.epoch_loss)
        plt.show()
        plt.title("Test step-loss")
        plt.plot([x[0] for x in self.test_loss], [x[1] for x in self.test_loss])
        plt.show()
        # show loss epoch graph
        plt.title("Epoch-Loss")
        plt.plot(self.epoch_loss)
        plt.show()

    def summary(self):
        summary(self.model, (3, 224, 224))

    def cleanup(self):
        self.writer.close()
        del self.model
        del self.criterion
        del self.optimizer
        del self.scheduler


class TrainingStats:
    def __init__(self, current_epoch, current_step, last_model, loss_list, epoch_loss, test_loss):
        self.current_epoch = current_epoch
        self.current_step = current_step
        self.last_model = last_model
        self.loss_list = loss_list
        self.epoch_loss = epoch_loss
        self.test_loss = test_loss

    def save(self, path):
        data_body = {
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "last_model": self.last_model,
            "loss_list": self.loss_list,
            "epoch_loss": self.epoch_loss,
            "test_loss": self.test_loss
        }
        with open(path, "w", encoding="UTF-8") as f:
            json.dump(data_body, f, indent=4)

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="UTF-8") as f:
            data_body = json.load(f)
        return TrainingStats(
            data_body["current_epoch"],
            data_body["current_step"],
            data_body["last_model"],
            data_body["loss_list"],
            data_body["epoch_loss"],
            data_body["test_loss"]
        )

    def __str__(self):
        return f"TrainingState(current_epoch={self.current_epoch}, current_step={self.current_step}, last_model={self.last_model}, loss_list={self.loss_list}, epoch_loss={self.epoch_loss}, test_loss={self.test_loss})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.current_epoch == other.current_epoch and self.current_step == other.current_step and self.last_model == other.last_model and self.loss_list == other.loss_list and self.epoch_loss == other.epoch_loss and self.test_loss == other.test_loss

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.current_epoch, self.current_step, self.last_model, self.loss_list, self.epoch_loss, self.test_loss))


def run_training():
    config = Config.from_file("config.json")
    logger.info(f"Config loaded: {config}")
    trainer = Trainer(config)
    logger.info("Start training...")
    trainer.train(config["epochs"])


if __name__ == "__main__":
    run_training()

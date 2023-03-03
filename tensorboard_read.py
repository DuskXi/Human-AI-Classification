import math
import os
from contextlib import suppress

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.stats import linregress
import sys
from time import gmtime, strftime

import pandas as pd

from tensorboard.backend.event_processing.event_file_loader import (
    EventFileLoader,
)
from config import Config
import matplotlib.pyplot as plt
import argparse

line_colors = ["red", "blue", "green", "y", "orange", "pink", "purple", "cyan", "brown", "gray", "olive", "black"]


def detect_outliers(df, flag=False):
    outlier_indices = []

    # 1st quartile (25%)
    Q1 = np.percentile(df, 25)
    # 3rd quartile (85%)
    Q3 = np.percentile(df, 90)
    # Interquartile range (IQR)
    IQR = Q3 - Q1
    outlier_index = []
    # outlier step
    outlier_step = 1.5 * IQR
    for i, nu in enumerate(df):
        # if (nu < Q1 - outlier_step) | (nu > Q3 + outlier_step):
        if flag:
            if (nu < Q1 - outlier_step) | (nu > Q3 + outlier_step):
                outlier_index.append(i)
        else:
            if (nu > Q3 + outlier_step):
                outlier_index.append(i)
    return outlier_index


def read_tensorboard_file(file_path):
    info = {}
    speed = {}
    for event in EventFileLoader(file_path).Load():
        step = event.step
        for value in event.summary.value:
            tag = value.tag
            value = value.tensor.float_val[0]
            if tag not in info:
                info[tag] = {}
                speed[tag] = {}
            info[tag][str(step)] = value
            wall_time = event.wall_time
            speed[tag][wall_time] = int(step)
    for tag in info.keys():
        if tag not in ['Epoch/Loss/train', 'Epoch/Loss/test']:
            outlier_index = detect_outliers(list(info[tag].values()))
            outlier_keys = [list(info[tag].keys())[i] for i in outlier_index]
            for key in outlier_keys:
                del info[tag][key]
    return info, speed


def draw_plot_by_info(info, speed=None, title="", output_folder=None):
    if speed is None:
        speed = {}
    figures = []
    for tag, data in info.items():
        x = []
        y = []
        for step, value in data.items():
            x.append(int(step))
            y.append(value)
        x_type = tag.split("/")[0]
        y_type = "%" if tag.split("/")[1] == "Accuracy" else ""
        figure = plt.figure()
        plot = figure.add_subplot(111)
        plot.set_title(f"{title}\n {tag}")
        plot.set_xlabel(f"{x_type}")
        plot.set_ylabel(f"{y_type}")
        plot.grid()
        plot.plot(x, y)
        if output_folder is not None:
            with suppress(Exception):
                os.makedirs(output_folder)
            plt.savefig(os.path.join(output_folder, f"{title.replace(':', '=')}-{tag.replace('/', '_')}.png"))
        else:
            plt.show()
        plt.clf()
        figures.append({
            "title": f"{title}\n {tag}",
            "xlabel": f"{x_type}",
            "ylabel": f"{y_type}",
            "data": (x, y)
        })
    time_list = []
    step_list = []
    max_key = max(list(speed.keys()), key=lambda x: len(list(speed[x].keys())))
    start_time = min(list(speed[max_key].keys()))
    for wall_time, step in speed[max_key].items():
        time_list.append((wall_time - start_time))
        step_list.append(step)

    slope, intercept, r_value, p_value, std_err = linregress(time_list, step_list)

    figures.append({
        "title": f"Speed",
        "xlabel": f"time(s)",
        "ylabel": f"steps",
        "data": (time_list, step_list),
        "label": f"speed: {slope / 1000.0:.2f} steps/s"
    })
    total_figure_width = math.ceil(math.sqrt(len(figures)))
    total_figure_height = math.ceil(len(figures) / total_figure_width)
    plt.rcParams.update({'font.size': 15})
    font_backup = plt.rcParams['font.size']
    # total_figure, axes = plt.subplots(total_figure_width, total_figure_height)
    total_figure = plt.figure(figsize=(25, 20))
    for i in range(total_figure_width):
        for j in range(total_figure_height):
            if i * total_figure_width + j >= len(figures):
                break
            plot = total_figure.add_subplot(total_figure_width, total_figure_height, i * total_figure_width + j + 1)
            plot.set_title(figures[i * total_figure_width + j]["title"])
            plot.set_xlabel(figures[i * total_figure_width + j]["xlabel"])
            plot.set_ylabel(figures[i * total_figure_width + j]["ylabel"])
            plot.plot(*figures[i * total_figure_width + j]["data"], label=figures[i * total_figure_width + j].get("label", None))
            plot.grid()

    total_figure.suptitle(title, fontsize=20)
    plt.tight_layout(pad=4, w_pad=2, h_pad=2)
    # plt.tight_layout()
    plt.legend()
    if output_folder is not None:
        with suppress(Exception):
            os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, f"{title.replace(':', '=')} total.png"))
    plt.show()
    plt.rcParams.update({'font.size': font_backup})


def draw_horizontal_comparison(infos, speeds=None, output_folder=None):
    colors = []
    index = 0
    for i in range(len(infos)):
        colors.append(line_colors[index])
        index += 1
        if index >= len(line_colors):
            index = 0
    total_figure = plt.figure(figsize=(40, 20))
    figure_args = {}
    speed_figure_args = {
        "x": [],
        "y": [],
        "color": [],
        "label": [],
        "ylabel": "steps",
    }
    for i in range(len(infos)):
        info = infos[i]
        for tag, data in info['info'].items():
            if tag not in figure_args:
                figure_args[tag] = {
                    "x": [],
                    "y": [],
                    "color": [],
                    "label": []
                }
            x = []
            y = []
            for step, value in data.items():
                x.append(int(step))
                y.append(value)
            figure_args[tag]["x"].append(x)
            figure_args[tag]["y"].append(y)
            figure_args[tag]["color"].append(colors[i])
            figure_args[tag]["label"].append(infos[i]["model_name"] + "-" + infos[i]["optimizer"])

    for i, speed in enumerate(speeds):
        time_list = []
        step_list = []
        max_key = max(list(speed.keys()), key=lambda x: len(list(speed[x].keys())))
        start_time = min(list(speed[max_key].keys()))
        for wall_time, step in speed[max_key].items():
            time_list.append(wall_time - start_time)
            step_list.append(step)
        slope, intercept, r_value, p_value, std_err = linregress(time_list, step_list)
        speed_figure_args["x"].append(time_list)
        speed_figure_args["y"].append(step_list)
        speed_figure_args["color"].append(colors[i])
        speed_figure_args["label"].append(f"speed: {slope:.2f} steps/s")

    figure_args["Speed(s)"] = speed_figure_args

    subplot_width = math.ceil(math.sqrt(len(figure_args)))
    subplot_height = math.ceil(len(figure_args) / subplot_width)
    subplot_width = 3
    subplot_height = 4
    plt.rcParams.update({'font.size': 15})
    for index, (tag, args) in enumerate(figure_args.items()):
        x_type = tag.split("/")[0]
        y_type = ("%" if tag.split("/")[1] == "Accuracy" else "") if len(tag.split("/")) > 1 else (args.get("ylabel", ""))
        plot = total_figure.add_subplot(subplot_width, subplot_height, index + 1)
        plot.set_title(f"{tag}")
        plot.set_xlabel(f"{x_type}")
        plot.set_ylabel(f"{y_type}")
        for i in range(len(args["x"])):
            plot.plot(args["x"][i], args["y"][i], color=args["color"][i], label=args["label"][i])
        plot.grid()
        plot.legend()

    # draw batch size in legend
    legend_line = []
    for i, info in enumerate(infos):
        batch_size = info["batch_size"]
        model_name = info["model_name"]
        optimizer = info["optimizer"]
        legend_line.append(Line2D([0], [0], color=colors[i], label=f"{model_name}-{optimizer} batch size: {batch_size}"))
    plt.tight_layout(pad=4, w_pad=2, h_pad=2)
    total_figure.suptitle(f"Comparison in {len(infos)} training logs", fontsize=20)
    total_figure.legend(handles=legend_line, loc='lower right', fancybox=True, shadow=True)

    if output_folder is not None:
        with suppress(Exception):
            os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, f"Comparison in {len(infos)} training logs.png"))
    plt.show()


def draw_ranking(target_tag, start_steps, infos, speeds, subplot):
    reverse = False if "Loss" not in target_tag else True
    colors = []
    index = 0
    for i in range(len(infos)):
        colors.append(line_colors[index])
        index += 1
        if index >= len(line_colors):
            index = 0
    results = {}
    labels = []
    for i, info in enumerate(infos):
        if target_tag not in info['info'] or info['optimizer'] == "SDG":
            continue
        model_name = info["model_name"]
        temp = []
        for step, data in info['info'][target_tag].items():
            if int(step) >= start_steps:
                temp.append(data)
        if len(temp) == 0:
            results[model_name] = 0
            continue
        indexs = detect_outliers(temp, True)
        temp = list(np.delete(temp, indexs))
        results[model_name] = np.sum(temp) / len(temp) if len(temp) > 0 else 0
    colors = [x[0] for x in sorted(zip(colors, list(results.values())), key=lambda x: x[1], reverse=reverse)]
    results = sorted(results.items(), key=lambda x: x[1], reverse=reverse)
    for model_name, value in results:
        if "Accuracy" in target_tag:
            labels.append(f"{value:.3f}%")
        elif not target_tag.endswith("rate"):
            labels.append(f"{value * 100:.3f}%")
        else:
            labels.append(f"{value:.3f}")
    bars = subplot.barh([x[0] for x in results], [x[1] for x in results], color=colors)
    subplot.bar_label(bars, labels=labels, padding=3)
    subplot.set_title(f"Ranking of {target_tag}")


def draw_ranking_for_all_tag(infos, speeds, output_folder=None):
    colors = []
    index = 0
    for i in range(len(infos)):
        colors.append(line_colors[index])
        index += 1
        if index >= len(line_colors):
            index = 0
    plt.rcParams.update({'font.size': 15})
    total_figure = plt.figure(figsize=(25, 20))
    tags = []
    for info in infos:
        tags.extend(info['info'].keys())
    tags = list(filter(lambda x: x not in ["Step/Gpu Power", "Step/Gpu Use", "Step/Learning rate"], tags))
    tags = list(set(tags))
    subplot_width = math.ceil(math.sqrt(len(tags)))
    subplot_height = math.ceil(len(tags) / subplot_width)
    subplot_width = 4
    subplot_height = 2
    for i, tag in enumerate(tags):
        subplot = total_figure.add_subplot(subplot_width, subplot_height, i + 1)
        if tag.startswith("Epoch"):
            draw_ranking(tag, 6, infos, speeds, subplot)
        else:
            draw_ranking(tag, 5000, infos, speeds, subplot)
    legend_line = []
    for i, info in enumerate(infos):
        batch_size = info["batch_size"]
        model_name = info["model_name"]
        optimizer = info["optimizer"]
        # legend_line.append(Line2D([0], [0], color=colors[i], label=f"{model_name}-{optimizer} batch size: {batch_size}"))
        legend_line.append(mpatches.Patch(color=colors[i], label=f"{model_name}-{optimizer} batch size: {batch_size}"))

    plt.tight_layout(pad=4, w_pad=2, h_pad=2)
    total_figure.suptitle(f"Ranking in {len(infos)} training logs", fontsize=20)
    total_figure.legend(handles=legend_line, loc='lower right', fancybox=True, shadow=True)
    if output_folder is not None:
        with suppress(Exception):
            os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, f"Ranking in {len(infos)} training logs.png"))
    plt.show()


def main(args):
    file_path = args.file_path
    task = args.task
    output_folder = args.output
    config_names = [x for x in os.listdir(task) if x.endswith(".json")]
    folder_names = [x for x in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, x))]
    target = []
    for config_name in config_names:
        config = Config.from_file(os.path.join(task, config_name))
        output_folder_template = f"{config['model_name']}-lr={config['learning_rate']}-optimizer={config['optimizer']}" + (
            f"-dynamic_lr_gamma={config['dynamic_lr_gamma']}" if config['dynamic_lr'] else "")
        if output_folder_template in folder_names:
            target.append({"path": os.path.join(file_path, output_folder_template), "config": config})
    target.sort(key=lambda x: x['path'])
    infos = []
    speeds = []
    for t in target:
        tensorboard_files = [x for x in os.listdir(os.path.join(t['path'], 'tensorboard')) if x.startswith("events.out.tfevents")]
        for tensorboard_file in tensorboard_files:
            info, speed = read_tensorboard_file(os.path.join(os.path.join(t['path'], 'tensorboard'), tensorboard_file))
            if len(list(info.keys())) == 0:
                continue
            title = f"Model: {t['config']['model_name']}, Optimizer: {t['config']['optimizer']}"
            infos.append({"info": info, "title": title, "model_name": t['config']['model_name'], "optimizer": t['config']['optimizer'], "batch_size": t['config']['batch_size']})
            speeds.append(speed)
            #draw_plot_by_info(info, speed=speed, title=title, output_folder=output_folder)
    #draw_horizontal_comparison(infos, speeds=speeds, output_folder=output_folder)
    draw_ranking_for_all_tag(infos, speeds, output_folder=output_folder)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", type=str, dest="file_path", default="./output", help="file path")
    parser.add_argument("--task", "-t", type=str, dest="task", default="./task", help="title")
    parser.add_argument("--output", "-o", type=str, dest="output", default="./saving", help="image output path")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

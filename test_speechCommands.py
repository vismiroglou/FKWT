from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from src.dataset import SpeechCommands
from utils.config_parser import parse_config
from utils.trainer import LightningKWT



def get_model(ckpt, config, useFNet=False):
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = LightningKWT.load_from_checkpoint(ckpt, config=config, useFnet=useFNet)
    model.to(device)
    print(device)

    return model, device


def get_data(config):
    eval_set = SpeechCommands(
        root=config["dataset_root"],
        audio_config=config["audio_config"],
        labels_map=config["labels_map"],
        subset="testing",
        augment=False,
    )

    eval_loader = DataLoader(
        eval_set, batch_size=config["hparams"]["batch_size"], num_workers=5
    )

    print("test loader size:", len(eval_loader))

    classes = {
        "backward": 0,
        "bed": 1,
        "bird": 2,
        "cat": 3,
        "dog": 4,
        "down": 5,
        "eight": 6,
        "five": 7,
        "follow": 8,
        "forward": 9,
        "four": 10,
        "go": 11,
        "happy": 12,
        "house": 13,
        "learn": 14,
        "left": 15,
        "marvin": 16,
        "nine": 17,
        "no": 18,
        "off": 19,
        "on": 20,
        "one": 21,
        "right": 22,
        "seven": 23,
        "sheila": 24,
        "six": 25,
        "stop": 26,
        "three": 27,
        "tree": 28,
        "two": 29,
        "up": 30,
        "visual": 31,
        "wow": 32,
        "yes": 33,
        "zero": 34,
    }
    return eval_loader, classes


def get_predictions(device, model, test_loader):
    predicted_labels = torch.tensor([]).to(device)
    true_labels = torch.tensor([]).to(device)

    for specs, labels in tqdm(test_loader):
        specs = specs.to(device)  # sending spectograms to device
        labels = labels.to(device)  # sending labels to device
        output = model(specs)  # feeding data to network

        predicted_labels = torch.cat(
            (predicted_labels, output.argmax(1))
        )  # argmax to find "hot one" in one hot encoding
        true_labels = torch.cat((true_labels, labels))

    predicted_labels = torch.flatten(predicted_labels).cpu()  # flattening dimensions
    true_labels = torch.flatten(true_labels).cpu()  # flattening dimensions
    return predicted_labels, true_labels


def test_pipeline(model, test_loader, device, classes):
    model.eval()
    print("Initiating testing...")
    predicted_labels, true_labels = get_predictions(device, model, test_loader)

    # Calculate accuracy
    accuracy = torch.sum(predicted_labels == true_labels).item() / len(true_labels)

    print("Accuracy:", accuracy)
    print("Shape of predicted_labels:", predicted_labels.shape)
    print("Shape of true_labels:", true_labels.shape)

    # Creating confusion matrix using sklearn function
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plotting confusion matrix
    df_cm = pd.DataFrame(
        cm / np.sum(cm, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    # confusion_matrix_fig = plt.figure(figsize=(12, 7))
    plt.title("Confusion Matrix")
    sns.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    # plt.savefig("confusion_matrix.png")
    plt.show()


def main(args):
    config = parse_config(args.conf)
    if args.dataset_root:
        config["dataset_root"] = args.dataset_root
    if args.labels_map:
        config["labels_map"] = args.labels_map

    model, device = get_model(args.ckpt, config, args.useFNet)
    test_loader, classes = get_data(config)

    # Print the shape of the first spectrogram in the training set
    spectrogram, _ = next(iter(test_loader))
    print("Shape of the first spectrogram in the training set:", spectrogram.shape)

    # Logging setup
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id

    if config["exp"]["wandb"]:
        wandb.login()
        with wandb.init(
            project=config["exp"]["proj_name"],
            name=config["exp"]["exp_name"],
            config=config["hparams"],
            entity=config["exp"]["entity"],
        ):
            test_pipeline(model, test_loader, device, classes)

    else:
        test_pipeline(model, test_loader, device, classes)


if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to config.yaml file."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint file.", default=None
    )
    parser.add_argument(
        "--id",
        type=str,
        required=False,
        help="Optional experiment identifier.",
        default=None,
    )
    parser.add_argument("--useFNet", type=bool, default=False, help="Use the FKWT")
    parser.add_argument("--dataset_root", type=str, help="Dataset root directory")
    parser.add_argument("--labels_map", type=str, help="Path to lbl_map.json")
    args = parser.parse_args()

    main(args)

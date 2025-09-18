import json
import pandas as pd
import matplotlib.pyplot as plt

models = {
    "resnet_base": lambda: resnet.ResidualNetwork(INPUT_DIM),
    "fully_connected_base": lambda: fully_connected.FullyConnectedNetwork(INPUT_DIM),
    "resnet_larger": lambda: resnet.ResidualNetwork_Larger(INPUT_DIM),
    "fully_connected_larger": lambda: fully_connected.FullyConnectedNetwork_Larger(INPUT_DIM),
    "resnet_deeper": lambda: resnet.ResidualNetwork_Deeper(INPUT_DIM),
    "fully_connected_deeper": lambda: fully_connected.FullyConnectedNetwork_Deeper(INPUT_DIM),
}

"""plot para os test cases """
def plot_test_cases(nn_type):
    file_path = f"{nn_type}/test_cases.json"   # depends on nn type
    with open(file_path, "r") as f:
        records = [json.loads(line) for line in f]
    df = pd.DataFrame(records)
    plt.figure(figsize=(10, 6))
    for sample, group in df.groupby("sample"):
        plt.plot(group["epoch"], group["prediction"], label=f"Sample {sample}", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Prediction")
    plt.title("Prediction per Epoch by Sample")
    #plt.ylim(0.1, 0)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{nn_type}/test_cases_plot.png") #depends on nn type

"""plot para a loss"""
def plot_loss(nn_type):
    file = f"{nn_type}/training.json" #depends on nn type
    with open(file, "r") as f:
        data = json.load(f)   # carrega um único objeto JSON válido

    loss = data["loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss)), loss, marker='o', linestyle='-', label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training Loss by Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{nn_type}/loss_plot.png") #depends on nn type

"""plot da loss detalhado """
def plot_loss_detail(nn_type):
    file = f"{nn_type}/training.json" #depends on nn type
    with open(file, "r") as f:
        data = json.load(f)   # carrega um único objeto JSON válido

    loss = data["loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss)), loss, marker='o', linestyle='-', label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 0.1)
    plt.title("Training Loss by Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{nn_type}/loss_plot_detailed.png") #depends on nn type

def run_all_plots(nn_type):
    plot_test_cases(nn_type)
    plot_loss(nn_type)
    plot_loss_detail(nn_type)

if __name__ == "__main__":
    for key in models.keys():
        plot_test_cases(key)
        plot_loss(key)
        plot_loss_detail(key)
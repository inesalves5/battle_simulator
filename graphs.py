<<<<<<< HEAD
import matplotlib.pyplot as plt
import csv

with open("log_path.csv", "r") as csvfile:
    log_reader = csv.reader(csvfile, delimiter=',')
    next(log_reader)  # Skip header
    epochs = []
    losses = []
    #jap= []
    for row in log_reader:
        epochs.append(int(row[0]))
        losses.append(float(row[1]))
        #jap.append(float(row[2]))
"""
plt.figure(figsize=(10, 5))
plt.plot(epochs, jap, label="Japanese")
plt.xlabel("Epoch")
plt.title("Value of test case por Epoch")
plt.legend()
plt.grid(True)
plt.savefig("value_per_epoch.png")
plt.show()
"""
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label="Losses")
plt.xlabel("Epoch")
plt.yscale("log")  # <- Escala logarítmica aqui
plt.title("Loss por Epoch")
plt.legend()
plt.grid(True)
plt.savefig("loss_per_epoch_v1.png")
plt.show()
=======
import json
import pandas as pd
import matplotlib.pyplot as plt

"""plot para os test cases """

file_path = "resnet/test_cases.json"   # depends on nn type
with open(file_path, "r") as f:
    records = [json.loads(line) for line in f]
df = pd.DataFrame(records)
plt.figure(figsize=(10, 6))
for sample, group in df.groupby("sample"):
    plt.plot(group["epoch"], group["prediction"], label=f"Sample {sample}", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Prediction")
plt.title("Prediction per Epoch by Sample")
plt.legend()
plt.grid(True)
plt.savefig("resnet/test_cases_plot.png") #depends on nn type
plt.show()

"""plot para a loss"""
file = "resnet/training.json" #depends on nn type
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
plt.savefig("resnet/loss_plot.png") #depends on nn type
>>>>>>> e11e087 (graficos e mlp e resnet feitos)

import json
import pandas as pd
import matplotlib.pyplot as plt

"""plot para os test cases """

file_path = "fully_connected/test_cases_1.json"   # depends on nn type
with open(file_path, "r") as f:
    records = [json.loads(line) for line in f]
df = pd.DataFrame(records)

#df = df[df["sample"] % 2 == 0] #so as pares que sao de dia pq ainda n treinou de noite

plt.figure(figsize=(10, 6))
for sample, group in df.groupby("sample"):
    plt.plot(group["epoch"], group["prediction"], label=f"Sample {sample}", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Prediction")
plt.title("Prediction per Epoch by Sample")
#plt.ylim(0.1, 0)
plt.legend()
plt.grid(True)
plt.savefig("resnet/test_cases_plot_100.png") #depends on nn type
plt.show()

"""plot para a loss"""
file = "fully_connected/training_1.json" #depends on nn type
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
plt.savefig("fully_connected/loss_plot_100.png") #depends on nn type

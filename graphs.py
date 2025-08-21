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

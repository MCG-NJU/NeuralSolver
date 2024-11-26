
data_dict = {
    "Adam2-Solver": [0.53, 0.55, 0.56, 0.56, 0.57, 0.58],
    "Adam4-Solver": [0.53, 0.55, 0.56, 0.57, 0.58, 0.58],
    "Searched-Solver": [0.58, 0.60, 0.59, 0.59, 0.60, 0.60],
}

x_axis = [5, 6, 7, 8, 9, 10]
color_table = {
    'Adam2-Solver':"#FF595E",
    'Adam4-Solver':"#FFCA3A",
    "Searched-Solver":"#8AC926",
}
import matplotlib.pyplot as plt
import numpy as np
for data_name, data in data_dict.items():
    plt.plot(x_axis, data, label=data_name, color=color_table[data_name])
    plt.scatter(x_axis, data, marker="x", color=color_table[data_name])
    if data_name == "Searched-Solver":
        for i in range(len(data)):
            plt.text(x_axis[i]-0.05, data[i]+0.002, str(data[i]), color=color_table[data_name])
plt.plot(x_axis, [0.59, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 0.58, "Eular(50steps) - 0.59", color="#6A4C93")
plt.title("Performance of solvers on SiT-XL/2")
plt.ylabel("Recall")
plt.ylim(0.5, 0.65)
plt.xlabel("Number of steps")
plt.legend()
# plt.show()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/sit-recall.png", bbox_inches='tight')
plt.close()
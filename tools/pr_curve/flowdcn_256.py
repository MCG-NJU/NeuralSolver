
data_dict = {
    "Adam2-Solver": [0.73, 0.76, 0.78, 0.79, 0.79, 0.79],
    "Adam4-Solver": [0.73, 0.76, 0.78, 0.79, 0.79, 0.80],
    "Searched-Solver": [0.74, 0.78, 0.79, 0.79, 0.80, 0.80],
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
            plt.text(x_axis[i] - 0.15, data[i] + 0.002, str(data[i]), color=color_table[data_name])
plt.plot(x_axis, [0.81, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 0.80, "Eular(50steps) - 0.81", color="#6A4C93")
plt.title("Performance of solvers on FlowDCN-XL/2(256x256)")
plt.ylabel("Precision")
plt.ylim(0.7, 0.85)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/flowdcn256-pr.png", bbox_inches='tight')
plt.close()
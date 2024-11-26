
data_dict = {
    "Adam2-Solver": [0.708, 0.76, 0.79, 0.81, 0.82, 0.82],
    "Adam4-Solver": [0.71, 0.76, 0.79, 0.81, 0.82, 0.825],
    "Searched-Solver": [0.80, 0.819, 0.829, 0.830, 0.834, 0.833],
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
            plt.text(x_axis[i]-0.15, data[i] + 0.002, str(data[i]), color=color_table[data_name])
plt.plot(x_axis, [0.837, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 0.83, "Eular(50steps) - 0.837", color="#6A4C93")
plt.title("Performance of solvers on FlowDCN-XL/2(512x512)")
plt.ylabel("Precision")
plt.ylim(0.65, 0.9)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/flowdcn512-pr.png", bbox_inches='tight')
plt.close()

data_dict = {
    "Adam2-Solver": [8.48, 7.32, 6.59, 6.06, 5.72, 5.43],
    "Adam4-Solver": [12.6, 10.1, 8.41, 7.24, 6.52, 5.97],
    "Searched-Solver": [5.50, 5.12, 5.2, 5.33, 5.32, 5.07],
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
            plt.text(x_axis[i] - 0.15, data[i] + 0.2, str(data[i]), color=color_table[data_name])
plt.plot(x_axis, [4.32, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 4.5, "Eular(50steps) - 4.32", color="#6A4C93")
plt.title("Performance of solvers on FlowDCN-XL/2(256x256)")
plt.ylabel("sFID")
plt.ylim(4, 15)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/flowdcn256-sfid.png", bbox_inches='tight')
plt.close()
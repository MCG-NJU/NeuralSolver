
data_dict = {
    "Adam2-Solver": [8.47, 5.96, 4.66, 3.87, 3.4, 3.05],
    "Adam4-Solver": [8.87, 6.21, 4.65, 3.70, 3.17, 2.81],
    "Searched-Solver": [5.10, 3.41, 2.73, 2.60, 2.46, 2.35],
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
plt.plot(x_axis, [2.17, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 2.5, "Eular(50steps) - 2.17", color="#6A4C93")
plt.title("Performance of solvers on FlowDCN-XL/2(256x256)")
plt.ylabel("FID")
plt.ylim(2, 10)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/flowdcn256-fid.png", bbox_inches='tight')
plt.close()
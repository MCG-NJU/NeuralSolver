
data_dict = {
    "Adam2-Solver": [122, 155, 177, 191, 202, 209],
    "Adam4-Solver": [123, 156, 178, 194, 205, 215],
    "Searched-Solver": [178, 203, 222, 226, 232, 238],
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
plt.plot(x_axis, [243, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 240, "Eular(50steps) - 243", color="#6A4C93")
plt.title("Performance of solvers on FlowDCN-XL/2(512x512)")
plt.ylabel("IS")
plt.ylim(100, 250)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/flowdcn512-is.png", bbox_inches='tight')
plt.close()
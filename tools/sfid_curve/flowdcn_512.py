
data_dict = {
    "Adam2-Solver": [19.12, 15.4, 13.0, 11.3, 10.2, 9.37],
    "Adam4-Solver": [24.4, 19.71, 15.9, 13.3, 11.65, 10.39],
    "Searched-Solver": [6.07, 5.10,	4.69, 4.7, 4.61, 4.68],
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
            plt.text(x_axis[i]-0.15, data[i] + 0.5, str(data[i]), color=color_table[data_name])
plt.plot(x_axis, [5.44, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 5.40, "Eular(50steps) - 5.44", color="#6A4C93")
plt.title("Performance of solvers on FlowDCN-XL/2(512x512)")
plt.ylabel("sFID")
plt.ylim(3, 20)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/flowdcn512-sfid.png", bbox_inches='tight')
plt.close()
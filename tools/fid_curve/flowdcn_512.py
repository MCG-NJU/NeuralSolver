
data_dict = {
    "Adam2-Solver": [19.21, 12.4, 9.21, 7.38, 6.26, 5.49],
    "Adam4-Solver": [19.9, 13.19, 9.55, 7.41, 6.16, 5.33],
    "Searched-Solver": [7.24, 4.7,	3.45, 3.18, 3.00, 2.77],
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
plt.plot(x_axis, [2.81, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 3.0, "Eular(50steps) - 2.81", color="#6A4C93")
plt.title("Performance of solvers on FlowDCN-XL/2(512x512)")
plt.ylabel("FID")
plt.ylim(2, 20)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/flowdcn512-fid.png", bbox_inches='tight')
plt.close()
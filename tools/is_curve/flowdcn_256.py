
data_dict = {
    "Adam2-Solver": [174, 195, 207, 216, 222, 226],
    "Adam4-Solver": [175, 194, 209, 221, 228, 233],
    "Searched-Solver": [192, 218, 228, 232, 237, 239],
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
            plt.text(x_axis[i]-0.1, data[i] + 2, str(data[i]), color=color_table[data_name])
plt.plot(x_axis, [247, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 249, "Eular(50steps) - 247", color="#6A4C93")
plt.title("Performance of solvers on FlowDCN-XL/2(256x256)")
plt.ylabel("IS")
plt.ylim(160, 260)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/flowdcn256-is.png", bbox_inches='tight')
plt.close()

data_dict = {
    "Adam2-Solver": [8.96, 6.35, 4.96, 4.16, 3.6, 3.26],
    "Adam4-Solver": [9.64, 6.92, 5.21, 4.15, 3.51, 3.11],
    "Searched-Solver": [4.92, 3.57, 2.78, 2.65, 2.47, 2.40],
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
            plt.text(x_axis[i]-0.05, data[i]+0.15, str(data[i]), color=color_table[data_name])
plt.plot(x_axis, [2.23, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 2.5, "Eular(50steps) - 2.23", color="#6A4C93")
plt.title("Performance of solvers on SiT-XL/2")
plt.ylabel("FID")
plt.ylim(2, 10)
plt.xlabel("Number of steps")
plt.legend()
# plt.show()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/sit-fid.png", bbox_inches='tight')
plt.close()
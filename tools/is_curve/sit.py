
data_dict = {
    "Adam2-Solver": [166, 190, 203, 212, 217, 222],
    "Adam4-Solver": [166, 187, 203, 214, 221, 227],
    "Searched-Solver": [192, 214, 226, 230, 231, 234],
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
            plt.text(x_axis[i]-0.05, data[i]+3, str(data[i]), color=color_table[data_name])
plt.plot(x_axis, [244, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 240, "Eular(50steps) - 244", color="#6A4C93")
plt.title("Performance of solvers on SiT-XL/2")
plt.ylabel("IS")
plt.ylim(160, 260)
plt.xlabel("Number of steps")
plt.legend()
# plt.show()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/sit-is.png", bbox_inches='tight')
plt.close()
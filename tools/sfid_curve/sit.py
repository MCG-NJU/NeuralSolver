
data_dict = {
    "Adam2-Solver": [12.6, 7.83, 7.08, 6.56, 6.25, 5.96],
    "Adam4-Solver": [13.6, 11.1, 9.29, 8.07, 7.33, 6.73],
    "Searched-Solver": [4.85, 4.83, 4.79, 4.89, 4.91, 4.96],
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
plt.plot(x_axis, [4.60, ] * len(x_axis), linestyle="--", color="#6A4C93")
plt.text(5, 4.0, "Eular(50steps) - 4.60", color="#6A4C93")
plt.title("Performance of solvers on SiT-XL/2")
plt.ylabel("sFID")
plt.ylim(3.5, 15)
plt.xlabel("Number of steps")
plt.legend()
# plt.show()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/sit-sfid.png", bbox_inches='tight')
plt.close()
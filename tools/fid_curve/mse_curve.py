data_dict = {
    "FlowDCN-S/2": [0.0336, 0.0204, 0.0150, 0.0110, 0.0087, 0.0081],
    "FlowDCN-B/2": [0.0342, 0.0211, 0.0126, 0.0105, 0.0076, 0.0064],
    "SiT-XL/2": [0.0324, 0.0192, 0.0117, 0.0079, 0.0072, 0.0051],
}

x_axis = [5, 6, 7, 8, 9, 10]
color_table = {
    'SiT-XL/2':"#FF595E",
    'FlowDCN-S/2':"#FFCA3A",
    "FlowDCN-B/2":"#8AC926",
}
import matplotlib.pyplot as plt
import numpy as np
for data_name, data in data_dict.items():
    plt.plot(x_axis, data, label=data_name, color=color_table[data_name])
    plt.scatter(x_axis, data, marker="x", color=color_table[data_name])
plt.title("RecError of Solver with Different Search Model on SiT-XL/2")
plt.ylabel("MSE")
# plt.ylim(2, 10)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/models-mse.png", bbox_inches='tight')
plt.close()
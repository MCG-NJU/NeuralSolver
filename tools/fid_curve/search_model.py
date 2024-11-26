
data_dict = {
    "FlowDCN-S/2": [5.35, 3.47, 3.05, 2.78, 2.54, 2.48],
    "FlowDCN-B/2": [4.92, 3.57, 2.78, 2.65, 2.47, 2.40],
    "SiT-XL/2": [5.65, 3.46, 2.86, 2.68, 2.58, 2.43],
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
plt.title("Solver with Different Search Model on SiT-XL/2")
plt.ylabel("FID")
plt.ylim(2, 6)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/models-fid.png", bbox_inches='tight')
plt.close()
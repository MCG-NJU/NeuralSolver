
data_dict = {
    "25-steps": [5.45,4.07,3.39,3.18,2.97,2.82],
    "50-steps": [4.97,3.48,3.13,2.77,2.60,2.55],
    "100-steps": [4.92,3.57,2.78,2.65,2.47,2.40],
}

x_axis = [5, 6, 7, 8, 9, 10]
color_table = {
    '25-steps':"#FF595E",
    '50-steps':"#FFCA3A",
    "100-steps":"#8AC926",
}
import matplotlib.pyplot as plt
import numpy as np
for data_name, data in data_dict.items():
    plt.plot(x_axis, data, label=data_name, color=color_table[data_name])
    plt.scatter(x_axis, data, marker="x", color=color_table[data_name])
plt.title("Solver with various steps of reference Euler on SiT-XL/2")
plt.ylabel("FID")
plt.ylim(2, 6)
plt.xlabel("Number of steps")
plt.legend()
# plt.margins(0, 0)
plt.savefig(f"tools/plot_figs/coeffs/steps-fid.png", bbox_inches='tight')
plt.close()
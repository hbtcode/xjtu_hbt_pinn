import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

h = 1000
h1=1
xc = torch.linspace(0, 1, h)
xm, ym = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)

u_real = (6 * xx * (1 - xx))/((h1+1-xx)*(h1+1-xx)* (1+2*h1))
u_real_fig = u_real.reshape(h, h)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_real_fig.detach().numpy())
ax.text2D(0.5, 0.9, "real solve", transform=ax.transAxes)
plt.show()
fig.savefig("real solve2.png")




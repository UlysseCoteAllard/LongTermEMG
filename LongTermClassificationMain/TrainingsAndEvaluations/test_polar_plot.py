import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


sns.set()

# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute areas and colors
fig = plt.figure()
ax = Axes3D(fig)

n = 12
m = 6
rad = np.linspace(-70, 60, m)
a = np.linspace((-70.*np.pi)/180., (70.*np.pi)/180., n)
r, th = np.meshgrid(rad, a)

z = np.random.rand(6, 12)
print(np.shape(r), "   ", np.shape(th), "   ", np.shape(z))

print(np.max(z), "   ", np.min(z))

new_cmap = truncate_colormap(cmap=plt.get_cmap("magma"), minval=np.min(z), maxval=np.max(z))

ax = plt.subplot(111, polar=True)


ax.pcolormesh(th, r, np.swapaxes(z, 1, 0), cmap=new_cmap)

ax.plot(a, r, ls='none', color='k')
#ax.grid()
ax.set_rorigin(-5.)
ax.set_theta_zero_location('W', offset=270)
ax.set_thetamin(-70)
ax.set_thetamax(70)

plt.show()

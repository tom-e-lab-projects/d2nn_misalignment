import torch
from model import system 
import matplotlib.pyplot as plt
import numpy as np

c = 3e8*1e3             # speed of light
f = 400e9               # 400GHz
lambda0 = c/f           # wavelength
L = 80   
M = 256
               # DOE size
z = [30.0,30.0,30.0,30.0] 
tilt = torch.tensor([[0.3,-0.1],[0.2,0.1]])
d = torch.tensor([[0,0],[0,0.0]])
d2 = torch.tensor([[0,0],[0,0.0]])
tilt_copy = torch.tensor([[0.22, 0.002],[0.3,-0.05]])

I1 = torch.zeros(((1,M,M)))

split1 = M//3
#I1[0,150:150,150:150] = -1
I1[0,split1:2*split1,split1:2*split1] = 1
I2 = I1.clone()


true_system = system(M, L, lambda0, z, d2, tilt)
#copy_system = system(M,L, lambda0, z, d_copy, tilt_copy)

out = true_system(I1, I2)

output = (out*out.conj()).real
output = output.squeeze(0).detach().numpy()

system_2 = system(M, L, lambda0, z, d, tilt_copy)
out2 = system_2(I1, I2)

output2 = (out2*out2.conj()).real
output2 = output2.squeeze(0).detach().numpy()

fig, ax = plt.subplots(1,2)
im = ax[0].imshow(output )
im2 = ax[1].imshow(output2)

plt.colorbar(im)
plt.colorbar(im2)
#im2 =ax.imshow(output2)
plt.show()


# for next time lets plot c.o.m against tilt angle and and displacement
# and also mean intensity
# since what we are learining is sime sort of correlation maximiser can we solve more elegantly


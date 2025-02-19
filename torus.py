import numpy as np
import model as model
import torch
import utils

import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.decomposition import PCA 
from sklearn.manifold import SpectralEmbedding

# draw patterns

# ckpt = torch.load('../logs/04_rnn_isometry/20220901-110251-rnn_step=10-adaptive_dr=True-reg_decay_until=15000-batch_size=10000-005-1-num_steps_train=100000-gpu=0/ckpt/checkpoint-step100000.pth')
# ckpt = torch.load('../logs/04_rnn_isometry/20220920-003731-rnn_step=10-block_size=12-trans_type=lstm-005-1-15-adaptive_dr=True-positive_v=False-reg_decay_until=15000-batch_size=8000-num_steps_train=100000-gpu=0/ckpt/checkpoint-step100000.pth')
# ckpt = torch.load('../logs/04_rnn_isometry/20220904-165820-rnn_step=5-block_size=24-01-15-09-adaptive_dr=True-reg_decay_until=15000-batch_size=10000-num_steps_train=20000-gpu=0/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/04_rnn_isometry/20220906-160854-rnn_step=5-block_size=36-02-2-1-adaptive_dr=True-reg_decay_until=15000-batch_size=10000-num_steps_train=20000-gpu=0/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/02_linear/20230305-194314-normalization=True-module_size=24-batch_size=50000-num_steps_train=20000--2-gpu=2/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/09_nonlinear_12mod_polar/20230805-105856-9-95-4-gpu=2/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/09_nonlinear_12mod_polar/20230805-133436-9-95-4-gpu=2/ckpt/checkpoint-step20000.pth')

# ckpt = torch.load('../logs/09_nonlinear_12mod_polar/20230918-091031-003-8-w_fixed=1-8-gpu=0/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/05_linear_polar_norm/20230918-213524-8-4-003-num_steps_train=20000-num_theta=24-gpu=2/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/09_nonlinear_24mod_polar_tanh/20230928-001206-0-norm_v=False-gpu=0/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/09_nonlinear_24mod_polar/20231026-213556-005-batch_size=4000-learnable_s=False-num_theta=24-0-4-5-positive_u=False-gpu=1/ckpt/checkpoint-step20000.pth')

ckpt = torch.load('../logs/21_linear_polar_norm_single/20240504-150027-0-0-0-gpu=0/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/21_linear_polar_norm_single/20240520-121950-s_fixed=8-positive_v=True-25-w_isometry=4-gpu=0/ckpt/checkpoint-step20000.pth')
# ckpt = torch.load('../logs/21_linear_polar_norm_single/20240504-162208-s_fixed=10-max_dr_isometry=5-5-gpu=0/ckpt/checkpoint-step20000.pth')

config = ckpt['config']
device = utils.get_device(0)
print(device)

model_config = model.GridCellConfig(**config.model)
model = model.GridCell(model_config)
model.load_state_dict(ckpt['state_dict'])
model.to(device)

weights = model.encoder.v.data.cpu().detach().numpy()
# weights = weights.reshape((-1, 12, 40, 40))[:8, :12]
# weights = weights.reshape((-1, 24, 40, 40))[:8, :24]
weights = weights.reshape((-1, 24, 40*40))[0] #[24, 1600]

pca = PCA(n_components=6)
reduced_vectors = pca.fit_transform(weights.transpose(1, 0))
print(reduced_vectors.shape)

embedding = SpectralEmbedding(n_components=3)
reduced_vectors = embedding.fit_transform(reduced_vectors)
print(reduced_vectors.shape)

# Plot
# cdict = []
# h,w,c = np.shape(reduced_vectors)
# for i in range(h):
#     for j in range(w):

# draw torus
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(*reduced_vectors.transpose(1, 0), c=reduced_vectors.transpose(1, 0)[0], s=30)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()

# draw rings
rate_map = weights

# Perform SVD
import scipy
X = rate_map - rate_map.mean(-1,keepdims=True)
X -= X.mean(-1, keepdims=True)
Ua, S, V = scipy.linalg.svd(X)

rm_embed = Ua.T@rate_map

res = 40
# Construct torus
# k1 = [1,0]
# k2 = [0.5, 0.5*np.sqrt(3)]
# k3 = np.array([-0.5, 0.5*np.sqrt(3)]) 

# k1 = [3,0]
# k2 = [2,2.5]
# k3 = [-1,2.3]

# k1 = [1,-1]
# k2 = [1.3,0.3]
# k3 = [0.3,1.3]

# k1 = [1.1,-1]
# k2 = [1.7,1]
# k3 = [0.2,1.5]

k1 = [1.7, 3]
k2 = [3.5, 0]
k3 = [1.7, -3]

# k1 = [1,-2]
# k2 = [2.2,0]
# k3 = [1,2]

# k1 = [2,0]
# k2 = [1,1]
# k3 = [1,-1]

freq = 1
x = np.mgrid[:res,:res] * 2*np.pi/ res
x = x.reshape(2, -1)
k = freq*np.stack([k1,k2,k3])
X = np.concatenate([np.cos(k.dot(x)), np.sin(k.dot(x))], axis=0)
cmaps = ['Blues', 'Oranges', 'Greens']

crop = 0
idxs1, idxs2 = np.mgrid[crop:res-crop, crop:res-crop]
idxs = np.ravel_multi_index((idxs1,idxs2), (res,res)).ravel()

# Find rotation
RM = rm_embed[:10, idxs]
X_crop = X[:,idxs]
R = np.linalg.inv(RM.dot(RM.T)).dot(RM).dot(X_crop.T).T

# Lowdin symmetric orthogonalization 
U,s,V = np.linalg.svd(R)
S2 = U.dot(np.diag(1./s)).dot(U.T)
R = S2.dot(R)

# Plot rings
plt.figure(figsize=(12,4))
for i in range(R.shape[0]//2):
    plt.subplot(1,3,i+1)
    plt.scatter(R.dot(RM)[i], R.dot(RM)[i+3], c=X[i][idxs], cmap=cmaps[i], s=20)
    plt.axis('off')

plt.show()

# Fourier transform 
Ng = 24
rm_fft_real = np.zeros([Ng,res,res])
rm_fft_imag = np.zeros([Ng,res,res])

for i in range(Ng):
    rm_fft_real[i] = np.real(np.fft.fft2(rate_map[i].reshape([res,res])))
    rm_fft_imag[i] = np.imag(np.fft.fft2(rate_map[i].reshape([res,res])))
    
rm_fft = rm_fft_real + 1j * rm_fft_imag

width = 6
idxs = np.arange(-width+1, width)
x2, y2 = np.meshgrid(np.arange(2*width-1), np.arange(2*width-1))
im = (np.real(rm_fft)**2).mean(0)
# im = (np.abs(rm_fft)).mean(0)
im[0,0] = 0
plt.scatter(x2,y2,c=im[idxs][:,idxs], s=500, cmap='Oranges', marker='s')
plt.axis('equal')
plt.axis('off')
plt.title('Mean power')
plt.show()

# freq = 1
# crop = 0
# res = 50
# k1 = [3,0]
# k2 = [2,3]
# k3 = [-1,3]
# phases = [np.angle(mode) for mode in modes]
# cmaps = ['Blues', 'Oranges', 'Greens']
# x = np.mgrid[:res,:res] * 2*np.pi/ res
# x = x.reshape(2, -1)
# k = freq*np.stack([k1,k2,k3])
# X = np.concatenate([np.cos(k.dot(x)), np.sin(k.dot(x))], axis=0)
# idxs1, idxs2 = np.mgrid[crop:res-crop, crop:res-crop]
# idxs = np.ravel_multi_index((idxs1,idxs2), (res,res)).ravel()

# plt.figure(figsize=(12,4))
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     B = np.stack([np.cos(phases[i]), np.sin(phases[i])])
#     test = B@rate_map
#     plt.scatter(test[0], test[1], c=X[i][idxs], cmap=cmaps[i], s=20)
#     plt.axis('off')

# print(weights[2,0])
# print(weights[3,0])
# utils.draw_heatmap(weights)
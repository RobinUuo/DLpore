import os

from resnetModel import resnet50
import numpy as np
import openpnm as op
import porespy as ps
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import r2_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = resnet50()
model.to(device)
model.load_state_dict(torch.load('./total_resnet50.pth', map_location=device))

# 读取images文件夹下所有文件的名字
rootdir = 'D:\dev\pythonProject\DL\sliceim_seed1110/'
imagelist = os.listdir(rootdir)
start = 20
# start = 70
end = 50
ii = 0
img = Image.open(rootdir + imagelist[0])
img = np.array(img, dtype=bool)
dim1 = img.shape[0]
dim2 = img.shape[1]
cutted = 360
# cutted = 180
raw = np.empty((cutted, cutted, end - start), dtype='bool')
# for i in range(len(imagelist)):
for i in range(start, end):
    img = Image.open(rootdir + imagelist[i])
    # convert to np.ndarray
    img = np.array(img, dtype=bool)
    raw[:, :, ii] = img[0:cutted, 0:cutted]  # seed = 0
    # raw[:, :, ii] = img[cutted:, 0:cutted] #seed = 1
    # raw[:, :, ii] = img[0:cutted, cutted:] #seed = 2
    ii = ii + 1
dir = 'D:\\tmp'
# savetorawtotal(dir, raw)
porosity = np.count_nonzero(raw == True) / (end - start) / cutted / cutted
print('generated raw file. porosity = ', porosity)

fig, ax = plt.subplots(1, 1, figsize=[4, 4])
ax.imshow(raw[:, :, 20], origin='lower', interpolation='none')
plt.show()

prev_diff = []
prev_adve = []
snow = ps.networks.snow2(raw, boundary_width=[3, 0, 0], parallelization=None)
regions = snow.regions
net = snow.network
throat_conns = net['throat.conns']
throat_global_center = net['throat.global_peak']
pore_centers = net['pore.coords']
data, zm_ratios = ps.networks.create_inputs(regions, throat_conns)
with torch.no_grad():
    for inputs in data:
        inputs = inputs.to(device)
        output1, output2 = model(inputs)
        output1 = output1.cpu().numpy()
        output2 = output2.cpu().numpy()
        prev_diff.append(output1)
        prev_adve.append(output2)
diff_list = [item for sublist in prev_diff for item in sublist]
adve_list = [item for sublist in prev_adve for item in sublist]
size_factors_dif = diff_list * (1 / np.array(zm_ratios))
size_factors_adv = adve_list * ((1 / np.array(zm_ratios)) ** 3)
net['throat.diffusive_size_factor_AI'] = size_factors_dif * 0.5
k = np.where(size_factors_adv < 0)
size_factors_adv[k] = 1e-100
net['throat.advection_size_factor_AI'] = size_factors_adv / (0.001 / 3)
pn = op.io.network_from_porespy(net)

Tsin = pn.find_neighbor_throats(pores=pn.pores('pore.xmin'))
Tsout = pn.find_neighbor_throats(pores=pn.pores('pore.xmax'))

Tsnoinout = []
inouts = np.concatenate((Tsin, Tsout))
for idx, ele in enumerate(net['throat.advection_size_factor_AI']):
    if idx not in inouts:
        Tsnoinout.append(idx)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
para = 1
net['throat.advection_size_factor_AI'][inouts] = net['throat.advection_size_factor_AI'][inouts] * para
net['throat.diffusive_size_factor_AI'][inouts] = net['throat.diffusive_size_factor_AI'][inouts] * para

fig, ax = plt.subplots(1, 1, figsize=[5, 5])
ax = op.visualization.plot_connections(network=pn, alpha=0.8, color='grey', ax=ax)
ax = op.visualization.plot_coordinates(network=pn, ax=ax, color='b', markersize=50)
plt.show()

pn['pore.diameter'] = pn['pore.extended_diameter']
pn['throat.diameter'] = pn['throat.inscribed_diameter']
pn['throat.length'] = pn['throat.direct_length']
air = op.phase.Phase(network=pn)

air['pore.diffusivity'] = 1.0 / 8
air['throat.diffusivity'] = 1.0 / 8
air['throat.diffusive_conductance'] = net['throat.diffusive_size_factor_AI']

cn = pn.conns
L1, L2 = pn['pore.diameter'][cn[:, 0]] / 2, pn['pore.diameter'][cn[:, 1]] / 2
Lt = pn['throat.length'] - pn['pore.inscribed_diameter'][cn[:, 0]] / 2 \
     - pn['pore.inscribed_diameter'][cn[:, 1]] / 2
for i in range(len(Lt)):
    if Lt[i] < 0:
        Lt[i] = 1e-8

D1, Dt, D2 = pn['pore.diameter'][
    cn[:, 0]], pn['throat.diameter'], pn['pore.diameter'][cn[:, 1]]
A1, At, A2 = np.pi * D1 ** 2 / 4, np.pi * Dt ** 2 / 4, np.pi * D2 ** 2 / 4
g_Geo = 1 / (L1 / A1 + L2 / A2 + Lt / At)

porexmin = np.where(pn['pore.xmin'])[0]
porexmax = np.where(pn['pore.xmax'])[0]
li = np.concatenate((porexmin, porexmax))
for i in range(len(cn)):
    for j in cn[i, :]:
        if j in li:
            g_Geo[i] = 1 / (pn['throat.length'][i] / (np.pi * pn['pore.extended_diameter'][j] ** 2 / 4))
            continue

# air['throat.diffusive_conductance'] = g_Geo*air['throat.diffusivity'][0]

# diffusion
Deff = op.algorithms.FickianDiffusion(network=pn, phase=air)
C_in = 1.0
C_out = 0.0
Deff.set_value_BC(pores=pn['pore.xmin'], values=C_in)
Deff.set_value_BC(pores=pn['pore.xmax'], values=C_out)
Deff.run()
# Ts = pn.find_neighbor_throats(pores=outlet, flatten=False, mode="union")

# flux0 = Deff.rate(throats=Ts)[0]
flux = Deff.rate(pores=pn['pore.xmax'])
fluxoutletD = -flux

Dvalue = fluxoutletD * cutted / (cutted * (end - start)) / (1 - 0)
reeff = Dvalue / air['pore.diffusivity'][0]
print('relative diffusivity: ', reeff)

c = Deff['pore.concentration']
r = Deff.rate(throats=pn.Ts, mode='single')
d = pn['pore.diameter']
d2 = pn['throat.diameter']
fig, ax = plt.subplots(figsize=[10, 10])
op.visualization.plot_coordinates(network=pn, color_by=c, size_by=d, markersize=1000, ax=ax)
op.visualization.plot_connections(network=pn, color_by=r, size_by=d2, linewidth=5, ax=ax)
plt.show()

# adv according to LBM parameters
air['pore.viscosity'] = 1.0 / 6
air['throat.viscosity'] = 1.0 / 6
air['throat.hydraulic_conductance'] = net['throat.advection_size_factor_AI']
# calculate directly
cond = np.pi / 128 / air['throat.viscosity'] \
       * pn['throat.equivalent_diameter'] ** 4 \
       / pn['throat.direct_length']

delpres = 0.001 / 3
flow = op.algorithms.StokesFlow(network=pn, phase=air)
flow.set_value_BC(pores=pn['pore.xmin'], values=delpres)
flow.set_value_BC(pores=pn['pore.xmax'], values=0)
flow.run()

fig, ax = plt.subplots(figsize=[10, 10])
d = pn['pore.diameter']
p = flow['pore.pressure']
ax = op.visualization.plot_connections(pn)
ax = op.visualization.plot_coordinates(pn, ax=ax, color_by=p, size_by=d, markersize=50)
plt.show()

Q = flow.rate(pores=pn['pore.xmax'], mode='group')[0]
Q1 = flow.rate(pores=pn['pore.xmin'], mode='group')[0]
A = op.topotools.get_domain_area(pn, inlets=pn['pore.xmin'], outlets=pn['pore.xmax'])
L = op.topotools.get_domain_length(pn, inlets=pn['pore.xmin'], outlets=pn['pore.xmax'])
mu = air['pore.viscosity'][0]

K = Q * cutted * mu / (cutted * (end - start)) / delpres
print('The value of K is: ', K)
pn['pore.concentration'] = flow['pore.pressure']
op.io.project_to_iMorph2(network=pn, filename='fibernet')

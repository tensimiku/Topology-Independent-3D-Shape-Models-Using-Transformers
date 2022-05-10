from time import time
import torch
import net.networks as networks
from net import dataset
import openmesh as om
import os
import numpy as np


if __name__ == "__main__":
    mesh =  om.read_trimesh("./net/coma_template.obj")
    template = (mesh.points(), mesh.face_vertex_indices())

    net = networks.shapetransformernet(0, 0, 0., False)
    net.load_model()

    tmp_dir = "./test"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    dset = dataset.coma_dataset("./coma_dataset/", is_train=False)

    with torch.no_grad():
        mean = torch.Tensor(net.mean).cuda()
        std = torch.Tensor(net.std).cuda()
        sel_idx = np.random.choice(len(dset), 8)
        print('test idx', sel_idx)
        template_neutral_shape = torch.Tensor(dset.nv).cuda()
        ns_norm = (template_neutral_shape  - mean ) / std

        x = torch.stack([dset[idx] for idx in sel_idx], dim=0)
        x = x.cuda(non_blocking=True).type(torch.float32) / std
        st = time()
        y = net.model(ns_norm, x) # for transformer
        ed = time()
        y = (y * std + template_neutral_shape).cpu().numpy()
    fin = ed-st
    print('done:', fin, 'sec')
    print('avg per frame per sec:', len(x)/fin)
    ox = (x * std+template_neutral_shape).cpu().numpy()
    for i, (ix, iy) in enumerate(zip(ox, y)):
        print(ix.shape)
        mesh = om.TriMesh(ix, template[1])
        om.write_mesh(os.path.join(tmp_dir, f'orig{i}.obj'), mesh)
        mesh = om.TriMesh(iy, template[1])
        om.write_mesh(os.path.join(tmp_dir, f'pred{i}.obj'), mesh)


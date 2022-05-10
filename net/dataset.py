from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import os 
import numpy as np
import openmesh as om
import json


coma_validation_shape = ['FaceTalk_170811_03274_TA', 'FaceTalk_170908_03277_TA']

class AbsDataset(Dataset):
    """mesh data wrapper"""
    def __init__(self):
        super(AbsDataset, self).__init__()

    @property
    @abstractmethod
    def x(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass

    # @abstractmethod
    def template(self):
        pass

    def normalize(self, x: np.ndarray):
        pass

    def denormalize(self, x: np.ndarray):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

class coma_dataset(Dataset):
    def __init__(self, dirpath: str, template_path='./net/coma_template.obj', is_train=True, normalize=False):
        self.need_normalize = normalize
        if is_train:
            shapedirs = list(filter(lambda x: x not in coma_validation_shape, os.listdir(dirpath)))
        else:
            shapedirs = coma_validation_shape # is validation
        datas = []
        if not os.path.exists(template_path):
            print('template path not exists.')
            print('please run: python net/dataset.py or provide valid path.')
            exit()
        
        for shape in shapedirs:
            shapepath = os.path.join(dirpath, shape)
            for expr in os.listdir(shapepath):
                exprpath = os.path.join(shapepath, expr)
                for ply in os.listdir(exprpath):
                    if ply.endswith(".ply"):
                        mesh = om.read_trimesh(os.path.join(exprpath, ply))
                        datas.append(mesh.points())

        
        self.nv = om.read_trimesh(template_path).points()
        self.shapes = np.array(datas)
        with torch.no_grad():
            self.std = torch.Tensor(np.std(self.shapes.reshape(-1, 3), axis=0))
            self.mean = torch.Tensor(np.mean(self.shapes.reshape(-1, 3), axis=0))
            minstd = 0.001 # adjust std value
            self.std[self.std<minstd] = minstd
            self.offsets = torch.Tensor(self.shapes - self.nv)
        # self.data = torch.Tensor(self.normalize(self.data))

    def save_std_mean(self, path):
        """save std, mean of current dataset"""
        np.savez_compressed(path, std=self.std, mean=self.mean)
    
    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        if self.need_normalize:
            return self.offsets[i] / self.std
        return self.offsets[i]
    
    def normalize(self, x: np.ndarray):
        return (x - self.mean) / self.std
    
    def denormalize(self, x: np.ndarray):
        return x * self.std + self.mean


def make_coma_template(coma_dir_path, save_path='./net/coma_template.obj'):
    """make template(mean shape) of coma shapes"""
    shapedirs = os.listdir(coma_dir_path)
    vs = []
    for shape in shapedirs:
        shapepath = os.path.join(coma_dir_path, shape)
        for expr in os.listdir(shapepath):
            plypath = os.path.join(shapepath, expr, expr+'.000001.ply')
            if os.path.exists(plypath):
                mesh = om.read_trimesh(plypath)
                f = mesh.face_vertex_indices()
                vs.append(mesh.points())
    mesh = om.TriMesh(np.mean(vs, axis=0), f)
    om.write_mesh(save_path, mesh)



def project_err(err):
    # face obj is 1000 centimeter = 1 so value 0.01 mean 10cm
    maxerr = 0.01 # max err to 10 cm 
    return np.clip(err, 0, maxerr) * 100 # make err [0, 1]

def save_dataset(results, dataset, template, savedir, stdmean=None):
    import time

    def denorm(x, std, mean):
        return x * std + mean

    if results is None:
        return
    
    if stdmean is not None:
        std = stdmean['std']
        mean = stdmean['mean']

    savedir = os.path.join(savedir, time.strftime('%Y%m%d-%H%M%S', time.localtime()))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if type(results) == list: # list type is shape completion.
        i = 0
        for result in results: # batches
            for _, vs in enumerate(result): # batch, iter, vertex, 3
                subdir = os.path.join(savedir, str(i))
                os.makedirs(subdir)
                if stdmean is not None:
                    vs = denorm(vs, std, mean)
                errs = []
                for ii, v in enumerate(vs): # iter, vertex, 3
                    # igl.write_obj(os.path.join(subdir, f'{ii}.obj'), v, template[1]) # save obj
                    mesh = om.TriMesh(v, template[1])
                    om.write_mesh(os.path.join(subdir, f'{ii}.obj'), mesh)
                    orig_err = np.linalg.norm(dataset.gt[i] - v, axis=-1) # calc per-vertex err
                    err = project_err(orig_err) # clip [0, 1] (0~10cm)
                    with open(os.path.join(subdir, f'{ii}.json'), 'w') as f:
                        json.dump(err.tolist(), f)
                    errs.append(orig_err)
                np.savez_compressed(os.path.join(subdir, 'err.npz'), err=np.array(errs)) # save raw error
                
                i += 1
    else:
        if type(results) == torch.Tensor:
            results = results.detach().cpu().numpy()
        if stdmean is not None:
            results = denorm(results, std, mean)
        for i, res in enumerate(results):
            # igl.write_obj(os.path.join(savedir, f'{i}.obj'), res, template[1])
            mesh = om.TriMesh(res, template[1])
            om.write_mesh(os.path.join(savedir, f'{i}.obj'), mesh)


if __name__ == "__main__":
    make_coma_template("./coma_dataset")


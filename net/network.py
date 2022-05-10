import os
import time
import numpy as np
import json
from net import dataset
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter



class net:
    def __init__(self, model_path, trainable, rank=None, w_decay=0.0001, lr_rate=0.0001, lr_decay=0.995):
        self.model_path = model_path
        self.trainable = trainable
        # self.mean = torch.Tensor(np.zeros(3)).cuda() # TODO fixit
        # self.std = torch.Tensor(np.ones(3)).cuda() # TODO fixit
        self.model = torch.nn.Module()
        self.rank = rank
        self.std = None
        self.mean = None
        self.loaded = False

        self.save_at = 10
        self.w_decay = w_decay 
        self.lr_rate = lr_rate 
        self.lr_decay = lr_decay

    def lossfn(self, x, y):
        self.loss = torch.functional.F.mse_loss(x, y)
        return self.loss

    def find_last_epoch(self):
        if not os.path.exists(self.model_path):
            return None
        fs = os.listdir(self.model_path)
        fs = list(filter(lambda x: not os.path.isdir(os.path.join(self.model_path,x)), fs))
        if len(fs) == 0:
            return None
        kf = lambda x: int(x.split('.')[0])
        sfs = sorted(fs, key=kf)
        return os.path.join(self.model_path, sfs[-1])
    
    def load_ckpt(self):
        self.std = self.ckpt['std'].cpu().numpy()
        self.mean = self.ckpt['mean'].cpu().numpy()
        self.loaded = True

    def load_model(self):
        lp = self.find_last_epoch()
        if lp and not self.loaded:
            self.ckpt = torch.load(lp)
            if self.rank is None:
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(self.ckpt['model'],prefix='module.')
            self.model.load_state_dict(self.ckpt['model'])
            print('loading from.. {}'.format(lp))
            self.load_ckpt()

    def load_model_at_epoch(self, epoch):
        lp = os.path.join(self.model_path, f'{epoch}.pt')
        if os.path.exists(lp) and not self.loaded:
            self.ckpt = torch.load(lp)
            self.model.load_state_dict(self.ckpt['model'])
            print('loading from.. {}'.format(lp))
            self.load_ckpt()
            
    def get_result_dir(self):
        return os.path.join(self.model_path, 'results')

    def save_results(self, x, y, std, mean, ep):
        with torch.no_grad():
            x = self.denormalize_input(x, std, mean)
            y = self.denormalize_input(y, std, mean)
            self.save_pos(x, y, ep)

    def save_pos(self, x, y, ep):
        gresdir = self.get_result_dir()
        if not os.path.exists(gresdir):
            os.makedirs(gresdir)
        torch.save({
            'x': x,
            'y': y,
        }, os.path.join(gresdir, str(ep)+'.pt'))
    
    def prepare_train_data(self):
        x = dataset.coma_dataset("./coma_dataset/", is_train=True, normalize=True)
        return x, len(x)
    
    def normalize_input(self, x, std, mean):
        return (x - mean) / std

    def denormalize_input(self, x, std, mean):
        return x * std + mean
    
    def calc_loss(self, x, y):
        loss = torch.nn.functional.mse_loss(x, y)
        return loss

    def train_setup(self, epoch, batch_size):
        self.dset, self.lenx = self.prepare_train_data()
        dset = self.dset
        lenx = self.lenx

        if self.rank is not None: # is distributed
            sampler = torch.utils.data.DistributedSampler(self.dset) if self.rank is not None else None
            traindata = torch.utils.data.DataLoader(self.dset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=sampler, drop_last=True, persistent_workers=True)
        else:
            traindata = torch.utils.data.DataLoader(self.dset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

        # traindata = torch.utils.data.DataLoader(self.dset, batch_size=batch_size, shuffle=True, num_workers=12, persistent_workers=True)

        is_main_process = self.rank is None or self.rank == 0


        save_at = self.save_at 
        lr_rate = self.lr_rate 
        lr_decay = self.lr_decay


        # find no weight decay params.
        no_weight_decay_params = []
        if hasattr(self.model, "no_weight_decay"):
            no_weight_decay_params.extend(self.model.no_weight_decay()) # for model it self

        def recurse_children(prefix, module: torch.nn.Module):
            if isinstance(module, torch.nn.Module):
                for n, m in module.named_children():
                    current = prefix+n+'.'
                    recurse_children(current, m)
                    if hasattr(m, "no_weight_decay"):
                        for p in m.no_weight_decay():
                            no_weight_decay_params.append(current+p)
 
        recurse_children('', self.model) # find another modules

        print(no_weight_decay_params)

        decay_params = []
        no_decay_params = []
        for n, p in self.model.named_parameters():
            if n in no_weight_decay_params:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        
        print(no_decay_params)

        optim_params = [
            {"params": decay_params, "weight_decay": self.w_decay},
            {"params": no_decay_params, "weight_decay": 0.},
        ]

        self.opti = torch.optim.RAdam(optim_params, lr=lr_rate)
        self.lr_sch = torch.optim.lr_scheduler.ExponentialLR(self.opti, lr_decay)

        if is_main_process:
            writer = SummaryWriter(os.path.join(self.model_path, 'logs'))
        else:
            writer = None


        print('prepare training . . . :')
        for pt in self.model.state_dict():
            print(pt, "\t", self.model.state_dict()[pt].size())
        
        ep = 0
        total_step = 0
        lp = self.find_last_epoch()
        if lp:
            ckpt = torch.load(lp)
            ep = ckpt['ep']
            total_step = ckpt['total_step']
            self.model.load_state_dict(ckpt['model'])
            self.opti.load_state_dict(ckpt['opti'])
            self.lr_sch.load_state_dict(ckpt['lr_scheduler'])
            print('loading from.. {}'.format(lp))
        elif not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        std = torch.Tensor(dset.std).cuda()
        mean = torch.Tensor(dset.mean).cuda()
        template_neutral_shape = torch.Tensor(dset.nv).cuda()
        # fesible_idxs = marker_idxs.get_all_feasible_idxs()
        # template_vtxs = template_neutral_shape[template_marker_idxs]

        return is_main_process, traindata, lenx, template_neutral_shape, std, mean, ep, total_step, writer, save_at


    def train(self, epoch, batch_size):
        if self.rank is not None:
            import torch.distributed as dist
        is_main_process, traindata, lenx, template_neutral_shape, std, mean, ep, total_step, writer, save_at = self.train_setup(epoch, batch_size)


        ns_norm = (template_neutral_shape - mean) / std

        if self.rank is not None:
            dist.barrier()

        istep = ep
        start_time = time.time()
        for ep in range(istep, epoch):
            if is_main_process:
                ep_losses = []
            for i, x in enumerate(traindata):
                step_time = time.time()
                with torch.no_grad():
                    x = x.cuda(non_blocking=True).type(torch.float32)
                # print(m.shape)
                self.model.zero_grad()
                y = self.model(ns_norm, x)
                loss = self.calc_loss(x, y)
                loss.backward()
                self.opti.step()
                step_time = time.time() - step_time

                if is_main_process:
                    total_step += batch_size
                    ep_losses.append(loss)
                    writer.add_scalar('step/total_loss', loss, total_step)
                
                if i%5 == 0 and is_main_process:
                    print(ep,': [' ,i*batch_size, "/", lenx, '] loss:', loss.item(), 'elapsed step time', step_time, end='\r')
            
            self.lr_sch.step()

            if is_main_process:
                writer.add_scalar('epoch/total_loss', sum(ep_losses)/len(ep_losses), ep)
                writer.add_scalar('epoch/learning_rate', self.lr_sch.get_last_lr()[0], ep)

            if ep % save_at == 0:
                if is_main_process:
                    torch.save({
                        'ep': ep,
                        'total_step': total_step,
                        'std': std,
                        'mean': mean,
                        'model': self.model.state_dict(),
                        'opti': self.opti.state_dict(),
                        'lr_scheduler': self.lr_sch.state_dict(),
                    }, os.path.join(self.model_path, str(ep)+'.pt'))
                    # if self.save_results:
                    #     self.save_results(x, y, std, mean, ep)
                
                    print(ep,': [' ,lenx, "/", lenx, '] loss:', loss.item())
                    print(ep,'th loss:', loss.item())
                    print("--- training %s seconds ---" % (time.time() - start_time))


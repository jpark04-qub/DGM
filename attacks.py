import torch
import numpy as np
import torchvision
import algorithm.attack as attack
from utilities.loss_function import list as lossf_list

import matplotlib.pyplot as plt
import time
import torch.nn.functional as F


def make_grid(sample):
    img = torchvision.utils.make_grid(sample)
    return img.detach().numpy()


class Configuration:
    def __init__(self, on=False, name="none", eps=0, eta=0, alp=0, beta=0, c=0, lr=0, norm='l2', iter=0, sigma=0, stop=False):
        self.on = on
        self.name = name
        self.eps = eps
        self.eta = eta
        self.c = c
        self.alp = alp
        self.beta = beta
        self.lr = lr
        self.norm = norm
        self.iter = iter  # steps
        self.sigma = sigma
        self.stop = stop
        return


class Statistics:
    def __init__(self):
        self.adve = []
        self.pred = []
        self.prob = []
        self.quer = []
        self.dist = []
        self.method = []
        return


def prediction(model, x):
    output = model(x)
    _, hx = output.data.max(1)
    return output, hx


def test(device, classes, target_model, test_loader):

    targeted_attack = False

    fgsm_cfg = Configuration(False, "fgsm", eps=0.3)
    #sgm_cfg  = Configuration(False, "sgm_l2", eps=0.01, norm='l2', iter=300)
    sgm_cfg = Configuration(True, "sgm_inf", eps=0.001, norm='inf', iter=300)
    #dgm_cfg  = Configuration(True, "dgm_l2", eps=0.01, lr=0.001, norm='l2', iter=300)
    dgm_cfg  = Configuration(True, "dgm_inf", eps=0.001, lr=0.001, norm='inf', iter=300)
    bim_cfg  = Configuration(True, "bim", eps=0.09, alp=0.001, iter=300)

    count = 0

    succ_sum = np.zeros(10)
    dist_sum = np.zeros(10)
    prob_sum = np.zeros(10)
    quer_sum = np.zeros(10)

    for data, true_class in test_loader:
        x = data.clone()
        y = true_class.clone()

        pix_min = torch.min(x.flatten()).to(device)
        pix_max = torch.max(x.flatten()).to(device)

        x = x.to(device)
        y = y.to(device)

        p, h = prediction(target_model, x)
        if h != y:
            continue

        if targeted_attack:
            # l.l. attack
            ll = p.min(1, keepdim=True)[1]
            y = ll.flatten()
            #y = torch.tensor([403]).cuda()

        count += 1

        stat = Statistics()

        def evaluation(idx, alg, name):
            stat.method.append(name)
            if targeted_attack:
                adv, query, _, _ = alg.target(x, y)
            else:
                adv, query, _, _ = alg.untarget(x, y)

            stat.adve.append(adv)

            adv_output = target_model(adv)
            #adv_output.detached()
            adv_pred = adv_output.max(1, keepdim=True)[1]

            stat.pred.append(adv_pred.item())
            stat.prob.append(adv_output[0, adv_pred].item())
            stat.quer.append(query)
            stat.dist.append(torch.norm(torch.abs(adv - x)) / torch.norm(x))

            successful_attack = False
            if targeted_attack and adv_pred.item() == y:
                successful_attack = True
            if not targeted_attack and adv_pred.item() != y:
                successful_attack = True
            if successful_attack:
                succ_sum[idx] += 1
                prob_sum[idx] += adv_output[0, adv_pred].flatten()
                dist_sum[idx] += torch.norm(torch.abs(adv - x)) / torch.norm(x)
                quer_sum[idx] += query

        idx = 0
        if fgsm_cfg.on:
            cfg = fgsm_cfg
            lossF = lossf_list(target_model.loss)
            alg = attack.FGSM(device, target_model, lossF, eps=cfg.eps, min=pix_min, max=pix_max)
            evaluation(idx, alg, cfg.name)
            idx += 1

        if sgm_cfg.on:
            cfg = sgm_cfg
            alg = attack.SGM(device, target_model, eps=cfg.eps, iter=cfg.iter, norm=cfg.norm, min=pix_min, max=pix_max)
            evaluation(idx, alg, cfg.name)
            idx += 1

        if dgm_cfg.on:
            cfg = dgm_cfg
            alg = attack.DGM(device, target_model, eps=cfg.eps, lr=cfg.lr, iter=cfg.iter, norm=cfg.norm, min=pix_min, max=pix_max)
            evaluation(idx, alg, cfg.name)
            idx += 1

        if bim_cfg.on:
            cfg = bim_cfg
            lossF = lossf_list(target_model.loss)
            alg = attack.BIM(device, target_model, lossF, eps=cfg.eps, alp=cfg.alp, min=pix_min, max=pix_max)
            evaluation(idx, alg, cfg.name)
            idx += 1

        if count == 1:
            for i in range(idx):
                print("{}[{} - {}] ".format(i, stat.method[i], target_model.name), end='')
            print("")

        print("{}[{}]- ".format(count, classes[true_class.item()]), end='')
        for i in range(idx):
            print("{}[{}, {:.3f}, {:.3f}, {}] ".format(int(succ_sum[i]),
                                            classes[stat.pred[i]], stat.dist[i], stat.prob[i], stat.quer[i]), end='')
        print("")

        if count % 100 == 0:
            for i in range(idx):
                print("[{:.3f}, {:.3f}, {}] ".format(dist_sum[i]/succ_sum[i], prob_sum[i]/succ_sum[i],
                                                             int(quer_sum[i]/succ_sum[i])), end='')
            print("")

        del stat


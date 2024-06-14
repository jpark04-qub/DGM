import torch
import numpy as np
import matplotlib.pyplot as plt
from ...base import Base
from ...utility import Utility as util

"""
Single Gradient Method 
A variant of GDM which use g- gradient only
"""

class SGM(Base):
    def __init__(self, device, model, eps=0.3, norm='l2', iter=40, min=0, max=1):
        super().__init__()
        self.device = device
        self.model = model
        self.eps = eps
        self.norm = norm
        self.min = min
        self.max = max
        self.iter = iter
        self.converged = False
        self.targeted = False

    def f(self, x):
        p = self.model(x)
        _, l = torch.sort(p, descending=True)
        l = l.flatten()
        y_predic = l[0]
        if self.targeted:
            y_target = self.label
        else:
            y_target = l[1]
        return y_predic, y_target, p

    def perturb(self, gn):
        if self.norm == 'l2':
            g = (gn / (torch.max(torch.abs(gn))+1e-10))
        else:
            g = gn.sign()
        eta = self.eps * g
        return eta

    def check_converge(self, t, y_predic, y_target):
        successful = False
        if self.targeted and y_predic == self.label:
            successful = True
        if not self.targeted and y_predic != self.label:
            successful = True
        if successful:
            # successful attack, run to the target class little further
            y_target = y_predic
            if not self.converged:
                self.converged = True
                t = self.iter
        return t, y_target

    def core(self, image, label):
        x_t = image.detach().clone()
        x_t.requires_grad = True
        y_predic, y_target, p = self.f(x_t)

        iter0 = 0
        t = 0
        while t < self.iter:
            # negative directional gradient
            gn = (-1) * torch.autograd.grad(p[0, y_predic], x_t, retain_graph=True, create_graph=True)[0]

            # perturbation
            eta_t = self.perturb(gn)

            # update adversarial example
            x_t = x_t.detach() + eta_t
            x_t = torch.clamp(x_t, self.min, self.max)

            # check convergence
            y_predic, y_target, p = self.f(x_t)
            t += 1
            t, y_target = self.check_converge(t, y_predic, y_target)

            iter0 += 1

        # post process
        adv = x_t.detach().clone()
        adv.requires_grad = False
        del x_t

        if self.converged is not True:
            print("SGM is not converged")
        else:
            adv = torch.clamp(adv, self.min, self.max)

        return adv, iter0, 0

    def untarget(self, image, label):
        self.targeted = False
        self.label = label

        img = image.detach().clone()
        adv, iter0, iter1 = self.core(img, label[0])

        return adv, iter0+iter1, iter0, iter1

    def target(self, image, label):
        raise RuntimeError('SGM not support targeted attack yet!')

        return adv, iter0+iter1, iter0, iter1


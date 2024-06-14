import torch
import torch.optim as optim
import torch.nn as nn
from ...base import Base

"""
Jeonghwan Park, Niall McLaughlin, Paul Miller
Hard-label based small query black-box adversarial attack (Supplements)
IEEE/CVF Winter Conference on Applications of Computer Vision, 2024
"""

class DGM(Base):
    def __init__(self, device, model, eps=0.01, lr=0.005, iter=300, c=0.3, norm='l2', min=0, max=1):
        super().__init__()
        self.device = device
        self.model = model
        self.eps = eps
        self.lr = lr
        self.iter = iter
        self.c = c
        self.norm = norm
        self.min = min
        self.max = max
        self.label = torch.tensor([0])
        self.targeted = False
        self.converged = False

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

    def penalty(self, p, y_predic, y_target):
        if self.converged:
            # target class is decided, run to the target class with gp only
            c = 0
        else:
            temp = (p[0, y_target] / (p[0, y_predic] + p[0, y_target]))
            c = 1 / torch.exp(4 * temp)
            c = min([c, self.c])
            if self.targeted:
                c = c
            if c < 0 or c > 1:
                print("Error wrong c value \n")
        return c

    def perturb(self, c, gn, gp):
        #gn = gn.clone().detach()
        #gp = gp.clone().detach()
        if self.norm == 'l2':
            g = (c * (gn / (torch.max(torch.abs(gn))+1e-10)) + (1-c) * (gp / (torch.max(torch.abs(gp))+1e-10)))
        else:
            g = (c * gn.sign() + (1 - c) * gp.sign())
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
                t = max(self.iter - round(t / 2.25), 0)
                #t = self.iter
        return t, y_target

    def tune_adv(self, image, label, adv_init):
        if self.lr == 0:
            return adv_init, 0

        adv_t = adv_init.detach().clone()

        optimizer = optim.Adam([adv_t], lr=self.lr)
        t = 0
        while t < 100:
            adv = adv_t.detach().clone()
            # dist = torch.dist(adv, image, 2)
            adv_t.requires_grad = True
            cost = nn.MSELoss(reduction='sum')(adv_t, image)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            y_predic, _, _ = self.f(torch.clamp(adv_t, self.min, self.max).detach())
            if y_predic != label:
                break
            t = t + 1
        return adv, t

    def core(self, image, label):
        x_t = image.detach().clone()
        x_t.requires_grad = True
        y_predic, y_target, p = self.f(x_t)

        iter0 = 0
        t = 0
        while t < self.iter:
            # penalty
            c = self.penalty(p, y_predic, y_target)

            # negative directional gradient
            gn = (-1) * torch.autograd.grad(p[0, y_predic], x_t, retain_graph=True, create_graph=True)[0]
            # positive directional gradient
            gp = (+1) * torch.autograd.grad(p[0, y_target], x_t, retain_graph=True, create_graph=True)[0]

            # perturbation
            eta_t = self.perturb(c, gn, gp)

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
            print("DGM is not converged")
        else:
            # adversarial tuning
            adv, iter1 = self.tune_adv(image, y_predic, adv)
            adv = torch.clamp(adv, self.min, self.max)

        return adv, iter0, 0

    def untarget(self, image, label):
        self.targeted = False
        self.label = label

        img = image.detach().clone()
        adv, iter0, iter1 = self.core(img, label[0])

        return adv, iter0+iter1, iter0, iter1

    def target(self, image, label):
        self.targeted = True
        self.label = label

        img = image.detach().clone()
        adv, iter0, iter1 = self.core(img, label[0])

        return adv, iter0+iter1, iter0, iter1



import torch
import numpy as np
import matplotlib.pyplot as plt
from ...base import Base
from ...utility import Utility as util

"""
Alexey Kurakin, Ian J. Goodfellow, Samy Bengio
Adversarial Examples in the Physical World
International Conference on Learning Representations, 2017
"""

class BIM(Base):
    def __init__(self, device, model, lossF, eps=4/255, alp=1/255, iter=300, min=0, max=1):
        super().__init__()
        self.device = device
        self.model = model
        self.lossF = lossF
        self.eps = eps
        self.alp = alp
        self.min = min
        self.max = max
        self.iter = iter
        self.targeted = False

    def core(self, image, label):
        adv = image.detach().clone().to(self.device)

        #if self.iter == 0:
        #    steps = int(min(self.eps * 255 + 4, 1.25 * self.eps * 255))
        #else:
        #    steps = self.iter
        steps = self.iter

        iter0 = 0
        for i in range(steps):
            adv.requires_grad = True
            outputs = self.model(adv)
            cost = self.lossF(outputs, label)

            if self.targeted:
                cost *= -1

            grad = torch.autograd.grad(cost, adv, retain_graph=False, create_graph=False)[0]

            adv = adv + self.alp * grad.sign()

            a = torch.clamp(image - self.eps, min=self.min)
            b = (adv >= a).float() * adv + (a > adv).float() * a
            c = (b > image + self.eps).float() * (image + self.eps) + (image + self.eps >= b).float() * b
            adv = torch.clamp(c, max=self.max).detach()

            output = self.model(adv).to(self.device)
            _, pre = torch.max(output, 1)
            if self.targeted:
                if label == pre:
                    break
            else:
                if label != pre:
                    break
            iter0 += 1
        else:
            print("BIM attack is not converged")

        image.requires_grad = False

        #adv = torch.clamp(adv, self.pix_min, self.pix_max)
        return adv, iter0

    def untarget(self, image, label):
        self.targeted = False
        adv, iter0 = self.core(image, label)
        return adv, iter0, 0, 0

    def target(self, image, label):
        self.targeted = True
        adv, iter0 = self.core(image, label)
        return adv, iter0, 0, 0
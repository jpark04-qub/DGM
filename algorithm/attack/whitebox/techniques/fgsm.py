import torch
import numpy as np
from ...base import Base

"""
Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy
Explaining and Harnessing Adversarial Examples
International Conference on Learning Representations, 2015
"""

class FGSM(Base):
    def __init__(self, device, model, lossF, eps=0.1, min=0, max=1):
        super().__init__()
        self.device = device
        self.model = model
        self.lossF = lossF
        self.eps = eps
        self.min = min
        self.max = max

    def test1(self, image, label): #FGSM

        image = image.to(self.device)

        image.requires_grad = True

        output = self.model(image).to(self.device)
        loss = self.lossF(output, label).to(self.device)
        gradient = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]

        image.requires_grad = False
        # Collect the element-wise sign of the data gradient
        perturb = self.eps * gradient.sign()
        # adversarial example
        adv = image + perturb
        adv = torch.clamp(adv, self.min, self.max)

        return adv, 0, 0, 0

    def untarget(self, image, label): #FGSM

        image = image.to(self.device)

        image.requires_grad = True

        output = self.model(image).to(self.device)
        loss = self.lossF(output, label).to(self.device)
        gradient = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]

        image.requires_grad = False
        # Collect the element-wise sign of the data gradient
        perturb = self.eps * gradient.sign()
        # adversarial example
        adv = image + perturb
        adv = torch.clamp(adv, self.min, self.max)

        return adv, 1, 1, 0

    def target(self, image, label): #FGSM
        image.requires_grad = True
        output = self.model(image).to(self.device)

        loss = -self.lossF(output, torch.tensor([label])).to(self.device)
        gradient = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]

        image.requires_grad = False
        # Collect the element-wise sign of the data gradient
        perturb = self.eps * gradient.sign()
        # adversarial example
        adv = image + perturb
        adv = torch.clamp(adv, self.min, self.max)

        return adv, perturb, 1, 0

    def train(self, image, label): #FGSM
        adv, perturb, iter0, iter1 = self.untarget(image, label)

        return adv, perturb, iter0, iter1
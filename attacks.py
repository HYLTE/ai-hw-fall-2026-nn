"""White-box, untargeted L-inf attacks: FGSM, I-FGSM / PGD and MI-FGSM.

Each attack takes a model on [0, 1] pixels, clean images x and true labels y, and returns
adversarial images x_adv with ||x_adv - x||_inf <= eps and pixels in [0, 1].
"""
import torch
from torch.nn import functional as F


def input_gradient(model, x, y):
    """nabla_x L(f(x), y): gradient of the cross-entropy loss with respect to the input."""
    x = x.detach().requires_grad_(True)
    # Sum, not mean, so each image's gradient does not depend on the batch size.
    loss = F.cross_entropy(model(x), y, reduction="sum")
    return torch.autograd.grad(loss, x)[0]


def project(x_adv, x, eps):
    """Clip x_adv into the L-inf ball of radius eps around x, then into valid pixels [0, 1]."""
    return (x + (x_adv - x).clamp(-eps, eps)).clamp(0, 1)


def fgsm(model, x, y, eps):
    """FGSM (Goodfellow et al., 2015): x' = clip(x + eps * sign(nabla_x L(x, y)))."""
    return (x + eps * input_gradient(model, x, y).sign()).clamp(0, 1)


def pgd(model, x, y, eps, alpha, steps, random_start=True):
    """PGD (Madry et al., 2018): x_{t+1} = Proj(x_t + alpha * sign(nabla_x L(x_t, y))),
    starting from a uniformly random point in the eps-ball around x.
    Without the random start (x_0 = x) this is I-FGSM / BIM (Kurakin et al., 2017)."""
    x_adv = project(x + torch.empty_like(x).uniform_(-eps, eps), x, eps) if random_start else x
    for _ in range(steps):
        x_adv = project(x_adv + alpha * input_gradient(model, x_adv, y).sign(), x, eps)
    return x_adv


def mifgsm(model, x, y, eps, alpha, steps, decay=1.0):
    """MI-FGSM (Dong et al., 2018): g_{t+1} = mu * g_t + nabla_x L(x_t, y) / ||nabla_x L(x_t, y)||_1,
    x_{t+1} = Proj(x_t + alpha * sign(g_{t+1})).  With decay mu = 0 it is I-FGSM."""
    x_adv, g = x, torch.zeros_like(x)
    for _ in range(steps):
        grad = input_gradient(model, x_adv, y)
        # Per-image L1 norm; the clamp only guards against an all-zero gradient.
        g = decay * g + grad / grad.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        x_adv = project(x_adv + alpha * g.sign(), x, eps)
    return x_adv

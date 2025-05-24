import torch


def saliency(m, x, y=None, createGraph=False):
    if not x.requires_grad:
        x.requires_grad_(True)
    if hasattr(x, 'zero_grad'):
        x.zero_grad()
    output = m(x)

    if y == None:
        y = output.argmax(dim=1, keepdim=True)

    # scores = output.gather(1, y.view(-1, 1)).squeeze()
    scores = output.gather(1, y.view(-1, 1))
    # m.zero_grad()
    saliency_map = torch.autograd.grad(scores, x, torch.ones_like(scores), create_graph=createGraph)[0]
    # saliency_map = saliency_map.abs()

    return saliency_map, output

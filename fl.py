import copy
import operator
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

import fuse
import model
from im2col import im2col
from namagiri import add, matmul_fl, to_f32, to_posit


def weight2posit(m, n=8, es=1):
    if isinstance(m, nn.Conv2d):
        m.weight_p = to_posit(m.weight.data.numpy(), n, es)
        m.weight_p = m.weight_p.reshape((m.weight.size(0), -1))

        if m.bias is not None:
            m.bias_p = to_posit(m.bias.data.numpy(), n, es)

    if isinstance(m, nn.Linear):
        m.weight_p = to_posit(m.weight.data.numpy(), n, es).T
        if m.bias is None:
            m.bias_p = to_posit(m.bias.data.numpy(), n, es)


def posit_cnn(m, i, n=8, es=1):

    batch, _, n_H_prev, n_W_prev = i.shape

    n_H = int((n_H_prev + 2 * m.padding[0] - m.kernel_size[0]) / m.stride[0]) + 1
    n_W = int((n_W_prev + 2 * m.padding[0] - m.kernel_size[0]) / m.stride[0]) + 1

    im = im2col(i, m.kernel_size, m.stride[0], m.padding[0])

    out = matmul_fl(m.weight_p, im, n, es)

    if m.bias is not None:
        out = add(out.T, m.bias_p, 8, 1).T

    out = np.array(np.hsplit(out, batch)).reshape((batch, m.out_channels, n_H, n_W))

    return out


def posit_linear(m, i, n=8, es=1):

    if m.bias is None:
        out = matmul_fl(i, m.weight_p, n, es)
    else:
        out = add(matmul_fl(i, m.weight_p, n, es), m.bias_p, n, es)

    return out


def parasitize(m, n=8, es=1):
    if isinstance(m, nn.Conv2d):
        m.forward = lambda x: posit_cnn(m, x)

    if isinstance(m, nn.Linear):
        m.forward = lambda x: posit_linear(m, x)

    if isinstance(m, nn.ReLU):
        m.forward = lambda x: np.where(x <= 128, x, 0)

    if isinstance(m, nn.MaxPool2d):
        m.orig = copy.deepcopy(m)
        m.forward = lambda x: to_posit(
            m.orig(torch.from_numpy(to_f32(x, n, es))).detach().numpy(), n, es
        )


def test(img, target) -> int:
    net = model.Net()
    net.load_state_dict(
        torch.load("res9.pth", map_location=torch.device("cpu")), strict=False
    )

    net.eval()

    net = fuse.fuse(net)

    net.apply(lambda m: weight2posit(m))
    net.apply(lambda m: parasitize(m))

    for node in net.graph.nodes:
        if node.op == "call_function":
            if node.target == torch.flatten:
                node.target = lambda x, i: np.reshape(x, (x.shape[0], -1))
            if node.target == operator.add:
                node.target = lambda x, y: add(x, y, 8, 1)

    net.recompile()

    qi = to_posit(img.numpy(), 8, 1)
    with torch.no_grad():
        outputs = net(qi)
        outputs = torch.from_numpy(to_f32(outputs, 8, 1))

    _, predicted = torch.max(outputs, dim=1)
    return (predicted == target.flatten()).sum().item()


if __name__ == "__main__":
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        "data/cifar", train=False, download=True, transform=test_transform
    )

    batch_size = len(test_dataset) // os.cpu_count() + 1
    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    from joblib import Parallel, delayed

    result = Parallel(n_jobs=os.cpu_count())(
        delayed(test)(i, t) for i, t in data_loader
    )
    print(sum(result))

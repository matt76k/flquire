# https://qiita.com/kuroitu/items/35d7b5a4bde470f69570
import numpy as np


def im2col(images, filters, stride=1, pad=0):

    if images.ndim == 2:
        images = images.reshape(1, 1, *images.shape)
    elif images.ndim == 3:
        B, I_h, I_w = images.shape
        images = images.reshape(B, 1, I_h, I_w)
    B, C, I_h, I_w = images.shape
    if isinstance(filters, tuple):
        if len(filters) == 2:
            filters = (1, 1, *filters)
        elif len(filters) == 3:
            M, F_h, F_w = filters
            filters = (M, 1, F_h, F_w)
        _, _, F_h, F_w = filters
    else:
        if filters.ndim == 2:
            filters = filters.reshape(1, 1, *filters.shape)
        elif filters.ndim == 3:
            M, F_h, F_w = filters.shape
            filters = filters.reshape(M, 1, F_h, F_w)
        _, _, F_h, F_w = filters.shape

    if isinstance(stride, tuple):
        stride_ud, stride_lr = stride
    else:
        stride_ud = stride
        stride_lr = stride
    if isinstance(pad, tuple):
        pad_ud, pad_lr = pad
    elif isinstance(pad, int):
        pad_ud = pad
        pad_lr = pad
    elif pad == "same":
        pad_ud = 0.5 * ((I_h - 1) * stride_ud - I_h + F_h)
        pad_lr = 0.5 * ((I_w - 1) * stride_lr - I_w + F_w)
    pad_zero = (0, 0)

    O_h = int((I_h - F_h + 2 * pad_ud) // stride_ud + 1)
    O_w = int((I_w - F_w + 2 * pad_lr) // stride_lr + 1)

    pad_ud = int(np.ceil(pad_ud))
    pad_lr = int(np.ceil(pad_lr))
    pad_ud = (pad_ud, pad_ud)
    pad_lr = (pad_lr, pad_lr)
    images = np.pad(images, [pad_zero, pad_zero, pad_ud, pad_lr], "constant")

    cols = np.empty((B, C, F_h, F_w, O_h, O_w)).astype(images.dtype)
    for h in range(F_h):
        h_lim = h + stride_ud * O_h
        for w in range(F_w):
            w_lim = w + stride_lr * O_w
            cols[:, :, h, w, :, :] = images[:, :, h:h_lim:stride_ud, w:w_lim:stride_lr]

    return cols.transpose(1, 2, 3, 0, 4, 5).reshape(C * F_h * F_w, B * O_h * O_w)

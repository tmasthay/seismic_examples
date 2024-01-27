import os
from typing import Callable, Iterable, Iterator, List, Union
import black
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from numpy.typing import ArrayLike
import numpy as np
from omegaconf import AnyNode, DictConfig, ListConfig

from PIL import Image
import torch


def format_with_black(
    code: str,
    line_length: int = 80,
    preview: bool = True,
    magic_trailing_comma: bool = False,
    string_normalization: bool = False,
) -> str:
    try:
        mode = black.FileMode(
            line_length=line_length,
            preview=preview,
            magic_trailing_comma=magic_trailing_comma,
            string_normalization=string_normalization,
        )
        formatted_code = black.format_str(code, mode=mode)
        return formatted_code
    except black.NothingChanged:
        return code


class PlotTypes:
    Index = Union[int, List[int]]
    Indices = Union[Iterator[Index], List[Index]]
    PlotHandler = Callable[[ArrayLike, Indices, Figure, List[Axes]], bool]


class DotDict:
    def __init__(self, d):
        if type(d) is DotDict:
            self.__dict__.update(d.__dict__)
        else:
            self.__dict__.update(d)

    def set(self, k, v):
        self.__dict__[k] = v

    def get(self, k):
        if type(k) != str:
            raise ValueError(
                f"Key must be a string. Got key={k} of type {type(k)}."
            )
        return getattr(self, k)

    def __setitem__(self, k, v):
        self.set(k, v)

    def __getitem__(self, k):
        return self.get(k)

    def getd(self, k, v):
        return self.__dict__.get(k, v)

    def setdefault(self, k, v):
        self.__dict__.setdefault(k, v)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def has(self, k):
        return hasattr(self, k)

    def has_all(self, *keys):
        return all([self.has(k) for k in keys])

    def has_all_type(self, *keys, lcl_type=None):
        return all(
            [self.has(k) and type(self.get(k)) is lcl_type for k in keys]
        )

    def update(self, d):
        self.__dict__.update(DotDict.get_dict(d))

    def str(self):
        return format_with_black('DotDict(\n' + str(self.__dict__) + '\n)')

    def dict(self):
        return self.__dict__

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    @staticmethod
    def get_dict(d):
        if isinstance(d, DotDict):
            return d.dict()
        else:
            return d


def save_frames(
    frames, *, path, movie_format='gif', duration=100, verbose=False, loop=0
):
    dir, name = os.path.split(os.path.abspath(path))
    os.makedirs(dir, exist_ok=True)
    name = name.replace(f".{movie_format}", "")
    plot_name = f'{os.path.join(dir, name)}.{movie_format}'
    if verbose:
        print(f"Creating GIF at {plot_name} ...", end="")
    frames[0].save(
        plot_name,
        format=movie_format.upper(),
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop,
    )


def get_frames_bool(
    *,
    data: ArrayLike,
    iter: Iterable,
    fig: Figure,
    axes: List[Axes],
    plotter: PlotTypes.PlotHandler,
    framer: PlotTypes.PlotHandler = None,
    **kw,
):
    frames = []

    from time import time

    if framer is None:

        def default_frame_handler(*, data, idx, fig, axes, **kw2):
            fig.canvas.draw()
            frame = Image.frombytes(
                'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            return frame

        frame_handler = default_frame_handler

    curr_kw = kw
    for idx, plot_frame in iter:
        iter_time = time()
        curr_kw = plotter(data=data, idx=idx, fig=fig, axes=axes, **curr_kw)
        if plot_frame:
            frames.append(
                frame_handler(data=data, idx=idx, fig=fig, axes=axes, **kw)
            )
        iter_time = time() - iter_time
        print(
            f'idx={idx} took {iter_time:.2f} seconds:'
            f' len(frames)=={len(frames)}',
            flush=True,
        )

    return frames


def convert_config_simplest(obj):
    if isinstance(obj, DictConfig):
        return convert_config_simplest(DotDict(obj.__dict__["_content"]))
    elif isinstance(obj, dict):
        return convert_config_simplest(DotDict(obj))
    elif isinstance(obj, AnyNode):
        return obj._value()
    elif isinstance(obj, ListConfig):
        return convert_config_simplest(list(obj))
    elif isinstance(obj, list):
        return [convert_config_simplest(e) for e in obj]
    elif isinstance(obj, DotDict):
        for k, v in obj.items():
            obj[k] = convert_config_simplest(v)
        return obj
    elif obj is None:
        return None
    else:
        return obj


def bool_slice(
    *args,
    permute=None,
    none_dims=(),
    ctrl=None,
    strides=None,
    start=None,
    cut=None,
):
    def default_ctrl(x, y):
        return True

    ctrl = ctrl or default_ctrl
    permute = list(permute or range(len(args)))
    permute.reverse()

    # Logic is not correct here for strides, start, cut, etc. TODO: Fix
    strides = strides or [1 for _ in range(len(args))]
    start = start or [0 for _ in range(len(args))]
    cut = cut or [0 for _ in range(len(args))]
    tmp = list(args)
    for i in range(len(strides)):
        if i not in none_dims:
            tmp[i] = (tmp[i] - start[i] - cut[i]) // strides[i]

    args = list(args)
    for i in range(len(args)):
        if i not in none_dims:
            args[i] = args[i] - cut[i]
    # Total number of combinations
    total_combinations = np.prod(
        [e for i, e in enumerate(tmp) if i not in none_dims]
    )

    # Initialize indices
    idx = [
        slice(None) if i in none_dims else start[i] for i in range(len(args))
    ]

    for combo in range(total_combinations):
        yield tuple([tuple(idx)]) + (ctrl(idx, args),)

        # Update indices
        for i in permute:
            if i in none_dims:
                continue
            idx[i] += strides[i]
            if idx[i] < args[i]:
                break
            idx[i] = start[i]


def clean_idx(idx, show_colons=True):
    res = [str(e) if e != slice(None) else ':' for e in idx]
    if not show_colons:
        res = [e for e in res if e != ':']
    return f'({", ".join(res)})'


def tensor_summary(t, num=5):
    d = {
        'shape': t.shape,
        'dtype': t.dtype,
        'min': t.min(),
        'max': t.max(),
        'mean': t.mean() if t.dtype == torch.float32 else None,
        'std': t.std() if t.dtype == torch.float32 else None,
        f'top {num} values': torch.topk(t.reshape(-1), num, largest=True)[0],
        f'bottom {num} values': torch.topk(t.reshape(-1), num, largest=False)[
            0
        ],
    }
    s = ''
    for k, v in d.items():
        s += f'{k}\n    {v}\n'
    return s

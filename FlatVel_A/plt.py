from typing import Any, List
import matplotlib.pyplot as plt
import hydra
from helpers import (
    DotDict,
    bool_slice,
    clean_idx,
    convert_config_simplest,
    get_frames_bool,
    save_frames,
    tensor_summary,
)
from omegaconf import ListConfig
import os
import torch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap


def place_tensors(cfg: DotDict) -> DotDict:
    tensors = [
        e.replace('.pt', '')
        for e in os.listdir(cfg.paths.iomt)
        if e.endswith('.pt')
    ]
    tensors = {
        e: torch.load(os.path.join(cfg.paths.iomt, e + '.pt')) for e in tensors
    }
    cfg.tensors = DotDict(tensors)
    with open(os.path.join(cfg.paths.iomt, 'metadata.pydict'), 'r') as f:
        cfg.set('meta', DotDict({}))
        cfg.meta.update(DotDict(eval(f.read())))
        cfg.meta.set('units', DotDict({}))
        cfg.meta.units.set('space', 'm')
        cfg.meta.units.set('time', 's')
    cfg.t = torch.arange(cfg.meta.nt) * cfg.meta.dt

    with open(cfg.cbar.path, 'r') as f:
        cmap_name = os.path.basename(cfg.cbar.path).replace('.csv', '')
        data = [
            [float(ee) for ee in e.split(',')]
            for e in f.read().split('\n')
            if e != ''
        ]
        cfg.cbar['cmap'] = (
            LinearSegmentedColormap.from_list(cmap_name, data, len(data)),
        )[0]
        register_cmap(name=cmap_name, cmap=cfg.cbar.cmap)

    return cfg


def handle_outside(local_cfg, cfg, idx):
    plt.title(f'{local_cfg.title} {clean_idx(idx)}')
    plt.xlabel(f'{local_cfg.xlabel} ({cfg.meta.units.space})')
    plt.ylabel(f'{local_cfg.ylabel} ({cfg.meta.units.space})')
    if local_cfg.cbar:
        plt.colorbar()


def show_debug(cfg, name, info, idx=None):
    if idx is not None and sum([e for e in idx if e != slice(None)]) != 0:
        return
    if cfg.debug:
        print(info)
        print(
            f'Showing {name} info above. Will continue after'
            f' {cfg.debug_sleep} seconds.'
        )
        os.system(f'sleep {cfg.debug_sleep}')


def plot_model(
    *, data: torch.Tensor, idx: List, fig: Figure, axes: Axes, cfg: DotDict
):
    plt.clf()
    plt.imshow(
        data[idx],
        **cfg.model.kw,
        extent=[0, cfg.meta.nx * cfg.meta.dx, 0, cfg.meta.ny * cfg.meta.dy],
    )
    handle_outside(cfg.model.configs[0], cfg, idx)
    return {'cfg': cfg}


def plot_obs(
    *, data: torch.Tensor, idx: List, fig: Figure, axes: Axes, cfg: DotDict
):
    plt.clf()
    curr = cfg.obs.configs[0]
    d = data[idx].T
    depth_scaling = (
        (1.0 + cfg.obs.gain.const * (cfg.t / cfg.t[-1]) ** cfg.obs.gain.pow)
        .unsqueeze(-1)
        .expand(*d.shape)
    )
    show_debug(cfg, 'depth_scaling', tensor_summary(depth_scaling), idx)
    plt.imshow(
        d * depth_scaling,
        **curr.kw,
        extent=[0, cfg.meta.nx * cfg.meta.dx, cfg.meta.nt * cfg.meta.dt, 0.0],
    )
    handle_outside(curr, cfg, idx)
    rec_loc = cfg.tensors.rec_loc_y[idx[0], ...]
    rec_loc[..., 0] *= cfg.meta.dx
    rec_loc[..., 1] *= cfg.meta.dt
    plt.scatter(rec_loc[..., 0], rec_loc[..., 1], **cfg.obs.configs[0].rec)

    src_loc = cfg.tensors.src_loc_y[idx[0], ...].squeeze().float()
    src_loc[0] *= cfg.meta.dx
    src_loc[1] *= cfg.meta.dt
    plt.scatter(src_loc[0], src_loc[1], **cfg.obs.configs[0].src)
    plt.legend(**cfg.obs.configs[0].legend)
    return {'cfg': cfg}


@hydra.main(config_path="conf", config_name="plots", version_base=None)
def main(cfg: ListConfig) -> None:
    cfg = convert_config_simplest(cfg)
    cfg = place_tensors(cfg)

    show_debug(
        cfg, 'shape', DotDict({k: v.shape for k, v in cfg.tensors.items()})
    )

    iter = bool_slice(*cfg.tensors.model1.shape, **cfg.model.slice)

    fig, axes = plt.subplots()
    frames = get_frames_bool(
        data=cfg.tensors.model1,
        iter=iter,
        fig=fig,
        axes=axes,
        plotter=plot_model,
        cfg=cfg,
    )
    save_frames(frames, **cfg.model.save)

    fig, axes = plt.subplots(*cfg.obs.subplot.shape, **cfg.obs.subplot.kw)
    iter = bool_slice(*cfg.tensors.obs_data.shape, **cfg.obs.slice)
    frames = get_frames_bool(
        data=cfg.tensors.obs_data,
        iter=iter,
        fig=fig,
        axes=axes,
        plotter=plot_obs,
        cfg=cfg,
    )
    save_frames(frames, **cfg.obs.save)


if __name__ == "__main__":
    main()

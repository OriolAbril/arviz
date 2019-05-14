import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from ..data import convert_to_dataset, convert_to_inference_data
from .plot_utils import _scale_fig_size, xarray_to_ndarray, get_coords
from ..utils import _var_names
from .posteriorplot import plot_posterior
from .pairplot import plot_pair


def plot_func_posterior(data, ax, np_fun=np.diff, **kwargs):
    data = convert_to_dataset(data)
    var_names = [name.replace("\n", "_") for name in data.data_vars]
    func_name = np_fun.__name__
    xlabel = "{}({},\n{})".format(func_name, *var_names)
    data = np_fun(data.to_array(), axis=0).squeeze()
    plot_posterior({xlabel: data}, ax=ax, **kwargs)


def plot_pair_extended(
    data,
    var_names,
    coords=None,
    combined=True,
    lower_fun=plot_pair,
    upper_fun=None,
    diag_fun=plot_posterior,
    lower_kwargs=None,
    upper_kwargs=None,
    diag_kwargs=None,
    figsize=None,
    labels='edges',
    ax=None,
):
    if coords is None:
        coords = {}

    if lower_kwargs is None:
        lower_kwargs = {}

    if upper_kwargs is None:
        upper_kwargs = {}

    if diag_kwargs is None:
        diag_kwargs = {}

    if labels not in ("edges", "all", "none"):
        raise ValueError("labels must be one of (edges, all, none)")

    # Get posterior draws and combine chains
    data = convert_to_inference_data(data)
    posterior_data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, posterior_data)
    flat_var_names, _posterior = xarray_to_ndarray(
        get_coords(posterior_data, coords), var_names=var_names, combined=combined
    )
    flat_var_names = np.array(flat_var_names)
    numvars = len(flat_var_names)

    (figsize, _, _, _, _, _) = _scale_fig_size(figsize, None, numvars, numvars)
    if ax is None:
        _, ax = plt.subplots(numvars, numvars, figsize=figsize, constrained_layout=True)

    for i in range(numvars):
        for j in range(numvars):
            index = np.array([i, j], dtype=int)
            if i > j:
                if lower_fun is not None:
                    lower_fun(
                        {flat_var_names[j]: _posterior[j], flat_var_names[i]: _posterior[i]},
                        ax=ax[i, j],
                        **lower_kwargs
                    )
                else:
                    ax[i, j].axis("off")
            elif i < j:
                if upper_fun is not None:
                    upper_fun(
                        {flat_var_names[j]: _posterior[j], flat_var_names[i]: _posterior[i]},
                        ax=ax[i, j],
                        **upper_kwargs
                    )
                else:
                    ax[i, j].axis("off")
            elif i == j:
                if diag_fun is not None:
                    diag_fun({flat_var_names[i]: _posterior[i]}, ax=ax[i, j], **diag_kwargs)
                else:
                    ax[i, j].axis("off")


            if (i + 1 != numvars and labels=="edges") or labels=="none":
                ax[i, j].axes.get_xaxis().set_major_formatter(NullFormatter())
                ax[i, j].set_xlabel("")
            if (j != 0 and labels=="edges") or labels=="none":
                ax[i, j].axes.get_yaxis().set_major_formatter(NullFormatter())
                ax[i, j].set_ylabel("")

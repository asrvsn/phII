from typing import List
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
from itertools import repeat

from lib import *
from compute_polygon_metrics import all_metrics, iterate_over_stage_metrics
from compute_projected_metrics import iterate_over_stage_metrics_projected
from compute_spheroid_metrics import all_spheroid_metrics, SpheroidMetricContext, iterate_with_spheroid_context
from asrvsn_mpl.rvs import plot_violin_rvs

def make_ap_fig(
        metrics: List[Tuple[str, str]], 
        ap_mode: str='raw', 
        err_mode: str='stddev', 
        ap_bins: int=8, 
        fmult: float=1.0, 
        xsize: float=5.5, 
        ysize: float=3.6, 
        orientation='horizontal', 
        no_hists: bool=False, 
        hist_bins: int=100, 
        means: bool=True,
        qualifier: str='',
    ):
    my_ticks_fs = ticks_fs * 1.
    y_title_fs = title_fs * 1.1
    N = len(metrics)
    assert all(metric in all_metrics for (metric, _) in metrics)
    layout = [
        list(range(N)),
        list(range(N, 2 * N)),
    ]
    if no_hists:
        layout = layout[:1]
    if orientation == 'horizontal':
        pass
    elif orientation == 'vertical':
        layout = np.array(layout).T.tolist()
    else:
        raise ValueError('orientation must be either "horizontal" or "vertical"')
    fig, axs = plt.subplot_mosaic(layout, figsize=(xsize * len(layout[0]), ysize * len(layout)))

    # Draw
    for i, (metric, ap_pos_metric) in enumerate(metrics):
        print(f'Processing metric {metric}')
        units = all_metrics[metric].units
        assert units != None
        label = f'{metric} ({units})' if units != 'unitless' else metric

        # Col 1: Mean AP
        j = 0 
        ax = axs[layout[j][i]] if orientation == 'horizontal' else axs[layout[i][j]]
        pt.left_title_ax(ax, metric, fontsize=int(fmult*y_title_fs), offset=-0.275)

        def fun_ap(stage: str, df: pd.DataFrame):
            stage_xvals = df[ap_pos_metric]
            stage_yvals = df[metric]
            if ap_mode == 'raw':
                ax.scatter(stage_xvals, stage_yvals, color=stage_colors[stage], s=0.5, alpha=0.5)
            elif ap_mode == 'binned':
                bin_means, bin_edges, _ = stats.binned_statistic(stage_xvals, stage_yvals, statistic='mean', bins=ap_bins)
                bin_std, _, _ = stats.binned_statistic(stage_xvals, stage_yvals, statistic='std', bins=ap_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.fill_between(bin_centers, bin_means+bin_std, bin_means-bin_std, color=stage_colors[stage], alpha=0.1)
                ax.plot(bin_centers, bin_means, linewidth=group_lw*2, color=stage_colors[stage], zorder=10, linestyle='--')
            else:
                raise ValueError(f'Invalid ap_mode {ap_mode}')
            
        datasource = iterate_over_stage_metrics(fun_ap, qualifier)
        
        # To the existing set of ticks, append "P" and "A" at group_xvals[0], group_xvals[-1]
        ticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int((t*100))) for t in ticks]) # Convert to range [0, 100%] from posterior to anterior
        ax.set_xlabel(r'\% PA axis', fontsize=int(1.1*title_fs))
        ax.tick_params(axis='x', labelsize=int(fmult*my_ticks_fs))
        ax.tick_params(axis='y', labelsize=int(fmult*my_ticks_fs))
        pt.label_ax(ax, f'{upperchars[i]}{j+1}', fontsize=int(fmult*title_fs))

        # Col 2: empirical distributions
        if not no_hists:
            j = 1
            ax = axs[layout[j][i]] if orientation == 'horizontal' else axs[layout[i][j]]
            metric_data = dict()

            def fun_hist(stage: str, df: pd.DataFrame):
                metric_data[stage] = df[metric].copy()
            
            iterate_over_stage_metrics(fun_hist, qualifier)
            
            plot_violin_rvs(ax, metric_data, rvs=all_metrics[metric].rvs, bins=hist_bins, means=means, colors=stage_colors, linewidth=2*means_lw)
            ax.tick_params(axis='x', labelsize=int(fmult*my_ticks_fs))
            ax.tick_params(axis='y', labelsize=int(fmult*my_ticks_fs))
            ax.set_xlabel(label, fontsize=int(1.1*title_fs))
            # Address some units-specific axes labeling here
            if units == 'count':
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.set_ylabel('Frequency', fontsize=int(fmult*title_fs))
            pt.label_ax(ax, i if no_hists else f'{upperchars[i]}{j+1}', fontsize=int(fmult*title_fs))

    return layout, fig, axs, datasource

def make_projected_ap_fig(
        metrics: List[Tuple[str, str]], 
        ap_bins: int=8, 
        fmult: float=1.0, 
        xsize: float=5.5, 
        ysize: float=3.6, 
        hist_bins: int=100, 
        means: bool=True,
    ):
    my_ticks_fs = ticks_fs * 1.
    y_title_fs = title_fs * 1.1
    N = len(metrics)
    assert all(metric in all_metrics for (metric, _) in metrics)
    layout = [
        list(range(4*n, 4*(n+1)))
        for n in range(N)
    ]
    fig, axs = plt.subplot_mosaic(layout, figsize=(xsize * len(layout[0]), ysize * len(layout)))

    # Draw
    for i, (metric, ap_pos_metric) in enumerate(metrics):
        print(f'Processing metric {metric}')
        units = all_metrics[metric].units
        assert units != None
        label = f'{metric} ({units})' if units != 'unitless' else metric

        # Cols 1 and 2: Anterior-posterior, original vs. projected
        for j, projected in zip([0, 1], [False, True]):
            ax = axs[layout[i][j]]
            if i == 0:
                title = 'Original' if not projected else 'Projected'
                pt.title_ax(ax, title, fontsize=int(fmult*title_fs))
            if j == 0:
                pt.left_title_ax(ax, metric, fontsize=int(fmult*y_title_fs), offset=-0.275)

            def fun_ap(stage: str, df_raw: pd.DataFrame, df_proj: pd.DataFrame):
                stage_xvals = df_raw[ap_pos_metric] # Use the AP values from df_raw
                stage_yvals = (df_proj if projected else df_raw)[metric]
                bin_means, bin_edges, _ = stats.binned_statistic(stage_xvals, stage_yvals, statistic='mean', bins=ap_bins)
                bin_std, _, _ = stats.binned_statistic(stage_xvals, stage_yvals, statistic='std', bins=ap_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.fill_between(bin_centers, bin_means+bin_std, bin_means-bin_std, color=stage_colors[stage], alpha=0.1)
                ax.plot(bin_centers, bin_means, linewidth=group_lw*2, color=stage_colors[stage], zorder=10, linestyle='--')
                
            datasource = iterate_over_stage_metrics_projected(fun_ap)
        
            # To the existing set of ticks, append "P" and "A" at group_xvals[0], group_xvals[-1]
            ticks = [0, 0.25, 0.5, 0.75, 1]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int((t*100))) for t in ticks]) # Convert to range [0, 100%] from posterior to anterior
            ax.set_xlabel(r'\% PA axis', fontsize=int(1.1*title_fs))
            ax.tick_params(axis='x', labelsize=int(fmult*my_ticks_fs))
            ax.tick_params(axis='y', labelsize=int(fmult*my_ticks_fs))
            pt.label_ax(ax, f'{upperchars[i]}{j+1}', fontsize=int(fmult*title_fs))
            # Match y-axis limits between original and projected
            if j == 1:
                ax.set_ylim(axs[layout[i][0]].get_ylim())

        # Cols 3 and 4: empirical distributions, original vs. projected
        for j, projected in zip([2, 3], [False, True]):
            ax = axs[layout[i][j]]
            metric_data = dict()

            if i == 0:
                title = 'Original' if not projected else 'Projected'
                pt.title_ax(ax, title, fontsize=int(fmult*title_fs))

            def fun_hist(stage: str, df_raw: pd.DataFrame, df_proj: pd.DataFrame):
                metric_data[stage] = (df_proj if projected else df_raw)[metric].copy()
            
            iterate_over_stage_metrics_projected(fun_hist)
            
            plot_violin_rvs(ax, metric_data, rvs=all_metrics[metric].rvs, bins=hist_bins, means=means, colors=stage_colors, linewidth=2*means_lw)
            ax.tick_params(axis='x', labelsize=int(fmult*my_ticks_fs))
            ax.tick_params(axis='y', labelsize=int(fmult*my_ticks_fs))
            ax.set_xlabel(label, fontsize=int(1.1*title_fs))
            # Address some units-specific axes labeling here
            if units == 'count':
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.set_ylabel('Frequency', fontsize=int(fmult*title_fs))
            pt.label_ax(ax, f'{upperchars[i]}{j+1}', fontsize=int(fmult*title_fs))
            # Match x-axis limits between original and projected
            if j == 3:
                ax.set_xlim(axs[layout[i][2]].get_xlim())

    return layout, fig, axs, datasource

def make_corr_fig(
        metric_pairs: List[List[Tuple[str, str, List[str]]]], # Metric on X axis, Metrix on Y axis, which stages to compute correlation 
        fmult: float=1, 
        apply_corr_transform: bool=True, 
        xlabel_all: bool=False, 
        show_corr: bool=True,
    ):
    layout = [[str(pair) for pair in sublist] for sublist in metric_pairs]
    fig, axs = plt.subplot_mosaic(layout, figsize=(5 * len(layout[0]), 4 * len(layout)))
    datasource = dict()
    seen = set()
    
    for i, sublist in enumerate(metric_pairs):
        for j, (metric_x, metric_y, corr_stages) in enumerate(sublist):
            n = i * len(sublist) + j
            assert metric_x != metric_y
            if apply_corr_transform:
                x_fn, y_fn = all_metrics[metric_x].corr_transform, all_metrics[metric_y].corr_transform
            else:
                x_fn, y_fn = lambda x: x, lambda x: x
            
            # Pair plot
            ax = axs[str((metric_x, metric_y, corr_stages))]
            mxlabel = all_metrics[metric_x].corr_transform_label if apply_corr_transform else all_metrics[metric_x].label
            if i == len(metric_pairs) - 1 or xlabel_all:
                ax.set_xlabel(mxlabel, fontsize=int(fmult*label_fs))
            else:
                ax.set_xticklabels([])
            mylabel = all_metrics[metric_y].corr_transform_label if apply_corr_transform else all_metrics[metric_y].label
            ax.set_ylabel(mylabel, fontsize=int(fmult*label_fs))

            if corr_stages == 'all':
                corr_stages = ['I', 'II', 'III', 'IV', 'S']
            assert isinstance(corr_stages, list) and all(stage in TOPVIEW_STAGES for stage in corr_stages)
            
            corr_x, corr_y = [], []

            # Add to datasource
            if not 'Stage' in seen:
                datasource['Stage'] = []
            if not mxlabel in seen:
                datasource[mxlabel] = []
            if not mylabel in seen:
                datasource[mylabel] = []

            def fun_corr(stage: str, df: pd.DataFrame):
                stage_xvals = df[metric_x]
                stage_yvals = df[metric_y]
                # Apply coordinate transforms
                stage_xvals, stage_yvals = x_fn(stage_xvals), y_fn(stage_yvals)
                # Add to datasource
                if not 'Stage' in seen:
                    datasource['Stage'].append(np.repeat(stage, len(stage_xvals)))
                if not mxlabel in seen:
                    datasource[mxlabel].append(stage_xvals)
                if not mylabel in seen:
                    datasource[mylabel].append(stage_yvals)
                # Add correlation set
                if stage in corr_stages:
                    corr_x.append(stage_xvals)
                    corr_y.append(stage_yvals)
                # Plot
                ax.scatter(stage_xvals, stage_yvals, s=0.5, alpha=scatter_alpha, color=stage_colors[stage])
            
            iterate_over_stage_metrics(fun_corr, '')
            seen.update(['Stage', mxlabel, mylabel])
                        
            corr_x, corr_y = np.concatenate(corr_x), np.concatenate(corr_y)

            # Regressions
            if show_corr:
                slope, intercept, r_value, p_value, std_err = stats.linregress(corr_x, corr_y)
                x = np.linspace(np.min(corr_x), np.max(corr_x), 100)
                y = slope * x + intercept
                ax.plot(x, y, color='black', linestyle='--', linewidth=means_lw)
                corr_title = f'({", ".join(corr_stages)}) $R^2={r_value**2:.3f}$'
                ax.text(0.05, 0.04, corr_title, transform=ax.transAxes, fontsize=int(fmult*ticks_fs))
            
            pt.label_ax(ax, n, fontsize=int(fmult*title_fs*1.5))
            ax.tick_params(axis='x', labelsize=int(fmult*ticks_fs))
            ax.tick_params(axis='y', labelsize=int(fmult*ticks_fs))

    datasource = {k: np.concatenate(v) for k, v in datasource.items()}
    datasource = pd.DataFrame(datasource)
    return layout, fig, axs, datasource

def make_projected_corr_fig(
        metric_pairs: List[Tuple[str, str, List[str]]], # Metric on X axis, Metrix on Y axis, which stages to compute correlation 
        fmult: float=1, 
        apply_corr_transform: bool=True, 
        xlabel_all: bool=False, 
        show_corr: bool=True,
        raw_metrics_override: List[str]=[], # Overrides to use raw instead of projected
    ):
    assert type(metric_pairs[0]) == tuple
    layout = np.arange(len(metric_pairs) * 2).reshape(2, len(metric_pairs))
    fig, axs = plt.subplot_mosaic(layout, figsize=(5 * len(layout[0]), 4 * len(layout)))
    datasource = dict()
    seen = set()
    
    for i, (metric_x, metric_y, corr_stages) in enumerate(metric_pairs):
        for j, projected in zip([0, 1], [False, True]):
            n = j * len(metric_pairs) + i
            assert metric_x != metric_y
            if apply_corr_transform:
                x_fn, y_fn = all_metrics[metric_x].corr_transform, all_metrics[metric_y].corr_transform
            else:
                x_fn, y_fn = lambda x: x, lambda x: x
            
            # Pair plot
            ax = axs[n]
            mxlabel = all_metrics[metric_x].corr_transform_label if apply_corr_transform else all_metrics[metric_x].label
            if j == 1: # Label only the bottom row
                ax.set_xlabel(mxlabel, fontsize=int(fmult*label_fs))
            else:
                ax.set_xticklabels([])
            mylabel = all_metrics[metric_y].corr_transform_label if apply_corr_transform else all_metrics[metric_y].label
            ax.set_ylabel(mylabel, fontsize=int(fmult*label_fs))
            if i == 0:
                pt.left_title_ax(ax, 'Original' if not projected else 'Projected', fontsize=int(fmult*title_fs), offset=-0.275)

            if corr_stages == 'all':
                corr_stages = ['I', 'II', 'III', 'IV']
            assert isinstance(corr_stages, list) and all(stage in TOPVIEW_STAGES for stage in corr_stages)
            
            corr_x, corr_y = [], []

            # Add to datasource
            if not 'Stage' in seen:
                datasource['Stage'] = []
            seenx, seeny = f'{mxlabel} (projected)' if projected else mxlabel, f'{mylabel} (projected)' if projected else mylabel
            if not seenx in seen:
                datasource[seenx] = []
            if not seeny in seen:
                datasource[seeny] = []

            def fun_corr(stage: str, df_raw: pd.DataFrame, df_proj: pd.DataFrame):
                stage_xvals = (df_raw if (not projected) or (metric_x in raw_metrics_override) else df_proj)[metric_x]
                stage_yvals = (df_raw if (not projected) or (metric_y in raw_metrics_override) else df_proj)[metric_y]
                # Apply coordinate transforms
                stage_xvals, stage_yvals = x_fn(stage_xvals), y_fn(stage_yvals)
                # Add to datasource
                if not 'Stage' in seen:
                    datasource['Stage'].append(np.repeat(stage, len(stage_xvals)))
                if not seenx in seen:
                    datasource[seenx].append(stage_xvals)
                if not seeny in seen:
                    datasource[seeny].append(stage_yvals)
                # Add correlation set
                if stage in corr_stages:
                    corr_x.append(stage_xvals)
                    corr_y.append(stage_yvals)
                # Plot
                ax.scatter(stage_xvals, stage_yvals, s=0.5, alpha=scatter_alpha, color=stage_colors[stage])
            
            iterate_over_stage_metrics_projected(fun_corr)
            seen.update(['Stage', seenx, seeny])
                        
            corr_x, corr_y = np.concatenate(corr_x), np.concatenate(corr_y)

            # Regressions
            if show_corr:
                slope, intercept, r_value, p_value, std_err = stats.linregress(corr_x, corr_y)
                x = np.linspace(np.min(corr_x), np.max(corr_x), 100)
                y = slope * x + intercept
                ax.plot(x, y, color='black', linestyle='--', linewidth=means_lw)
                corr_title = f'({", ".join(corr_stages)}) $R^2={r_value**2:.3f}$'
                ax.text(0.05, 0.04, corr_title, transform=ax.transAxes, fontsize=int(fmult*ticks_fs))
            
            pt.label_ax(ax, n, fontsize=int(fmult*title_fs*1.5))
            ax.tick_params(axis='x', labelsize=int(fmult*ticks_fs))
            ax.tick_params(axis='y', labelsize=int(fmult*ticks_fs))

    datasource = {k: np.concatenate(v) for k, v in datasource.items()}
    datasource = pd.DataFrame(datasource)
    return layout, fig, axs, datasource
    
def plot_spheroid_metrics(metrics: List[str], ncols: int=3):
    assert len(metrics) % ncols == 0
    layout = np.arange(len(metrics)).reshape(-1, ncols)
    fig, axs = plt.subplot_mosaic(layout, figsize=(4 * len(layout[0]), 4 * len(layout)))

    df = pd.read_csv(f'{DATA_PATH}/topview/all_spheroid_metrics.csv')
    assert all(metric in df.columns for metric in metrics)
    
    for i, metric in enumerate(metrics):
        r, c = np.unravel_index(i, layout.shape)
        ax = axs[layout[r][c]]
        ax.set_xlabel(metric, fontsize=label_fs)
        pt.label_ax(ax, i, fontsize=int(title_fs*1.35))
        if c == 0:
            ax.set_yticks(np.arange(len(TOPVIEW_STAGES)))
            ax.set_yticklabels(TOPVIEW_STAGES)
        else:
            ax.set_yticklabels([])

        for j, stage in enumerate(TOPVIEW_STAGES):
            data = np.array(df[df['Stage'] == stage][metric])
            ax.scatter(data, j * np.ones(len(data)), s=60, edgecolors=stage_colors[stage], facecolors='none')
            ax.plot([data.mean(), data.mean()], [j - 0.5, j + 0.5], c=stage_colors[stage], linewidth=means_lw)

        if all_spheroid_metrics[metric].log:
            ax.set_xscale('log')
        ax.tick_params(axis='y', labelsize=label_fs)
        ax.tick_params(axis='x', labelsize=label_fs)

    fig.subplots_adjust(hspace=0.3, wspace=0.05)
    datasource = df
    return fig, axs, layout, datasource

def plot_metrics_rvs(metrics: List[Tuple[str, str]], rvs=[stats.gamma], pvalue_statistic='ks', **kwargs):
    fig, axs = plt.subplots(len(metrics), len(TOPVIEW_STAGES), figsize=(5*len(TOPVIEW_STAGES), 5*len(metrics)), **kwargs)
    for j, (metric, ap) in enumerate(metrics):
        assert ap in ['anterior', 'posterior', None]

        vals = dict(zip(TOPVIEW_STAGES, [[] for _ in TOPVIEW_STAGES]))

        def fun(stage: str, spheroid: int, path: str, ctx: SpheroidMetricContext):
            svals = ctx.nondimensionalized_metrics[metric]
            if not (ap is None):
                svals = svals[ctx.get_hemisphere_index(ap, ctx.compartment_polygons)]
            vals[stage].append(svals.copy())

        iterate_with_spheroid_context(fun, enhanced=True, log=False)
        vals = {k: np.concatenate(v) for k, v in vals.items()}

        for i, stage in enumerate(TOPVIEW_STAGES):
            ax = axs[j][i] if len(metrics) > 1 else axs[i]
            if j == 0:
                pt.title_ax(ax, f'Stage {stage}')
            if i == 0:
                metric_title = metric if ap is None else f'{metric} ({ap})'
                pt.left_title_ax(ax, metric_title, offset=-0.21)

            rvsargs = dict()
            if not all_metrics[metric].floc is None:
                rvsargs['loc'] = all_metrics[metric].floc
            
            pt.plot_hist_rvs(ax, vals[stage], rvs=rvs, bins=50, log=True, pvalues=True, pvalue_statistic=pvalue_statistic, **rvsargs)
            ax.tick_params(axis='x', labelsize=int(ticks_fs))
            ax.tick_params(axis='y', labelsize=int(ticks_fs))
                
    datasource = pd.read_csv(f'{DATA_PATH}/topview/all_polygon_metrics.raw.csv')
    return fig, axs, datasource
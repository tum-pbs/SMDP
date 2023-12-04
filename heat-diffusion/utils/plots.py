import jax.numpy as jnp

import moviepy.editor as mp
import numpy as np

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams, colors
from matplotlib.ticker import LogFormatter

# Plotting settings thanks to Conor O'Riordan

# Plot sizes in inches
# These are roughly correct, they might need some slight adjustment
col_wid = 3.3258 * 1.3
col_sep = 0.3486 * 1.3
col_mar = 0.5990 * 1.3

one_column = [col_wid, col_wid]
two_column = [2 * col_wid + col_sep, col_wid + col_sep]

# White BG, black FG, Tab10 colours for lines etc
plt.style.use('seaborn-white')

# Font Settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{txfonts}'
rcParams['font.serif'] = 'times'

# Default figure size
rcParams['figure.figsize'] = one_column


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Defines a new colourmap using a range from a pre-defined one
    """

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def format_axes(a, log=False):
    """
    MNRAS style asks for axis ticks on every side of the plot, not just
    bottom+left. Run this on an axis object to set the right format, e.g.
    fig, ax = plt.subplots()
    ax.plot(...)
    format_axes(ax)
    fig.savefig(...)

    """

    if log:
        formatter = LogFormatter(10, labelOnlyBase=True)
        a.set_yscale('log')

    a.tick_params(axis='y', which='major', direction='in', color='k',
                  length=5.0, width=1.0, right=True)
    a.tick_params(axis='y', which='minor', direction='in', color='k',
                  length=3.0, width=1.0, right=True)
    a.tick_params(axis='x', which='major', direction='in', color='k',
                  length=5.0, width=1.0, top=True)
    a.tick_params(axis='x', which='minor', direction='in', color='k',
                  length=3.0, width=1.0, top=True)


def make_axes_invisible(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def generate_videos(data, savename):
    """
    Generate videos from the data
    :param data:
    :param savename:
    :return:
    """
    height = 6
    width = 15
    dpi = 200

    vid_files = []

    vmin = jnp.min(jnp.array(data['ground_truth']))
    vmax = jnp.max(jnp.array(data['ground_truth']))

    num_images = len(data['ground_truth'])

    for n in range(num_images):

        images = []

        for t in range(len(data['prediction_0_sequence'])):

            fig = plt.figure(figsize=(width, height))
            fig.set_dpi(dpi)

            gs = GridSpec(2, 5, figure=fig)

            ax_ground_truth = fig.add_subplot(gs[0, 0])
            ax_ground_truth.imshow(data['ground_truth'][n], cmap='jet', vmin=vmin, vmax=vmax)
            make_axes_invisible(ax_ground_truth)

            ax_input = fig.add_subplot(gs[1, 0])
            ax_input.imshow(data['input'][n], cmap='jet', vmin=vmin, vmax=vmax)
            make_axes_invisible(ax_input)

            for i in range(2):
                for j in range(4):
                    ax_prediction = fig.add_subplot(gs[i, j + 1])
                    ax_prediction.imshow(data[f'prediction_{i + j * 2}_sequence'][t][n], cmap='jet', vmin=vmin,
                                         vmax=vmax)
                    make_axes_invisible(ax_prediction)

            plt.tight_layout()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            images.append(image.reshape(dpi * height, dpi * width, 3))
            fig.clf()
            plt.close(fig)

        for i in range(10):
            images.append(images[-1])

        clip = mp.ImageSequenceClip(images, fps=8)

        clip.write_videofile(f'videos/{savename}_{n}.mp4', fps=8)

        vid_files.append(f'videos/{savename}_{n}.mp4')

    return vid_files


def plot_pictures(data):
    height = 6
    width = 15
    dpi = 200

    images = []

    vmin = jnp.min(jnp.array(data['ground_truth']))
    vmax = jnp.max(jnp.array(data['ground_truth']))

    num_images = len(data['ground_truth'])

    for n in range(num_images):

        fig = plt.figure(figsize=(width, height))
        fig.set_dpi(dpi)

        gs = GridSpec(2, 5, figure=fig)

        ax_ground_truth = fig.add_subplot(gs[0, 0])
        ax_ground_truth.imshow(data['ground_truth'][n], cmap='jet', vmin=vmin, vmax=vmax)
        make_axes_invisible(ax_ground_truth)

        ax_input = fig.add_subplot(gs[1, 0])
        ax_input.imshow(data['input'][n], cmap='jet', vmin=vmin, vmax=vmax)
        make_axes_invisible(ax_input)

        for i in range(2):
            for j in range(4):
                ax_prediction = fig.add_subplot(gs[i, j + 1])
                ax_prediction.imshow(data[f'prediction_{i + j * 2}'][n], cmap='jet', vmin=vmin, vmax=vmax)
                make_axes_invisible(ax_prediction)

        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        images.append(image.reshape(dpi * height, dpi * width, 3))
        fig.clf()
        plt.close(fig)

    return images


def plot_fft_power(data_dict, gt, input_):
    height = 7
    width = 15
    dpi = 200

    images = []

    for elem in gt:

        elem_gt = gt[elem]
        elem_input = input_[elem]

        fig = plt.figure(figsize=(width, height))
        fig.set_dpi(dpi)

        ax = fig.add_subplot(111)

        ax.plot(elem_gt, color='red')
        ax.plot(elem_input, color='red')

        for realization in data_dict[elem]:
            ax.plot(realization, color='blue')

        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        images.append(image.reshape(dpi * height, dpi * width, 3))
        fig.clf()
        plt.close(fig)

    return images


def plot_fft_normalized(data_dict, data_normalized, data_normalized_gt, data_normalized_input):
    height = 12
    width = 18
    dpi = 200

    images = []

    num_images = len(data_dict['ground_truth'])

    for n in range(num_images):
        fig = plt.figure(figsize=(width, height))
        fig.set_dpi(dpi)

        vmin = np.min(data_dict['ground_truth'][n])
        vmax = np.max(data_dict['ground_truth'][n])

        gs = GridSpec(2, 3, figure=fig)

        ax_input = fig.add_subplot(gs[0, 0])
        ax_input.imshow(data_dict['input'][n], cmap='jet', vmin=vmin, vmax=vmax)
        make_axes_invisible(ax_input)

        ax_gt = fig.add_subplot(gs[0, 1])
        ax_gt.imshow(data_dict['ground_truth'][n], cmap='jet', vmin=vmin, vmax=vmax)
        make_axes_invisible(ax_gt)

        ax_pred = fig.add_subplot(gs[0, 2])
        ax_pred.imshow(data_dict['prediction_0'][n], cmap='jet', vmin=vmin, vmax=vmax)
        make_axes_invisible(ax_pred)

        ax_fft = fig.add_subplot(gs[1, 0])
        ax_fft.imshow(data_normalized_input[n])
        make_axes_invisible(ax_fft)

        ax_fft = fig.add_subplot(gs[1, 1])
        ax_fft.imshow(data_normalized_gt[n])
        make_axes_invisible(ax_fft)

        ax_fft = fig.add_subplot(gs[1, 2])
        ax_fft.imshow(data_normalized[n][0])
        make_axes_invisible(ax_fft)

        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        images.append(image.reshape(dpi * height, dpi * width, 3))
        fig.clf()
        plt.close(fig)

    return images

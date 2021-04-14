import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from torchvision.utils import make_grid
import numpy as np
import json
import imageio
from tools import get_all_files
import networkx as nx
import seaborn as sns


# Source: https://www.idtools.com.au/3d-network-graphs-python-mplot3d-toolkit/
def network_plot_3D_original(G, angle=-90, elev=15, save_path=None):
    # Get node positions
    pos = nx.get_node_attributes(G, "pos")
    print(pos)

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]

    # 3D network plot
    with plt.style.context('ggplot'):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=20 + 20 * G.degree(key), edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(elev=elev, azim=angle)

    # Hide the axes
    ax.set_axis_off()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close('all')
    else:
        plt.show()

    return


def network_plot_3D(G, angle=-90, elev=15, save_path=None):
    # Get node positions
    vertices = np.array(list(nx.get_node_attributes(G, "pos").values()))
    labels = np.array(list(nx.get_node_attributes(G, "label").values()))

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # 3D network plot
    with plt.style.context("ggplot"):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=labels, cmap=plt.cm.Spectral, s=100, edgecolors='k')

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((vertices[j[0]][0], vertices[j[1]][0]))
            y = np.array((vertices[j[0]][1], vertices[j[1]][1]))
            z = np.array((vertices[j[0]][2], vertices[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(elev=elev, azim=angle)

    # Hide the axes
    ax.set_axis_off()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close('all')
    else:
        plt.show()

    return


def view_2d_emb_mnist(points_list, points_labels, points_titles,
                      iteration, prefix_exp, figsize=5, save=True, view=False):
    num_points_plot = len(points_list)
    fig, axs = plt.subplots(1, num_points_plot, figsize=(figsize * num_points_plot, figsize), squeeze=False)
    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # PLOT POINTS
    for i in range(num_points_plot):
        points = points_list[i]
        labels = points_labels[i]
        name = points_titles[i]

        for j in range(10):
            ids = np.where(labels == j)[0]

            axs[0][i].scatter(points[ids, 0], points[ids, 1], alpha=0.5, color=colors[j])
            axs[0][i].set_title(name)

        axs[0][i].legend(mnist_classes)

    if save:
        plt.savefig(f"{prefix_exp}/output_{'{:08d}'.format(iteration)}.png", bbox_inches='tight')

    if view:
        plt.show()

    plt.close()


def plot_embeddings(embeddings, targets, save_dir, xlim=None, ylim=None, name="train"):
    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(figsize=(10, 10))
    for i in range(10):
        ids = np.where(targets == i)[0]
        plt.scatter(embeddings[ids, 0], embeddings[ids, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.savefig(f"{save_dir}/out_{name}.png")


def view_points(points_list, points_labels=None, points_titles=None,
                iteration=None, save_dir="./", figsize=5, save=True, view=False):
    num_points_plot = len(points_list)
    fig, axs = plt.subplots(1, num_points_plot, figsize=(figsize * num_points_plot, figsize), squeeze=False)

    # PLOT POINTS
    for i in range(num_points_plot):
        points = points_list[i]
        labels = points_labels[i]
        if labels is None:
            labels = np.ones(shape=(points.shape[0],))
        name = points_titles[i]
        dim = points.shape[-1]

        if dim == 2:
            axs[0][i].scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        elif dim == 3:
            axs[0][i] = fig.add_subplot(1, num_points_plot, i + 1, projection='3d')
            axs[0][i].scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].view_init(elev=20, azim=-80)
            axs[0][i].set_title(name)
        elif dim == 1:
            axs[0][i].scatter(points[:, 0], 0, c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        else:
            continue
            # raise NotImplementedError

    if save:
        plt.savefig(f"{save_dir}/output_points_{'{:08d}'.format(iteration)}.png", bbox_inches='tight')

    if view:
        plt.show()

    plt.close()


def view_images(image_batches, channels=None, image_sizes=None,
                titles=None, image_per_batch=16,
                num_row=4, figsize=10, fig_title=None,
                view=False, save_path=None):
    num_batch = len(image_batches)
    print(f"Showing {image_per_batch} image(s) per batch for {num_batch} batch(es).")
    fig, axs = plt.subplots(1, num_batch, figsize=(figsize, figsize * num_batch), squeeze=False)

    for i in range(num_batch):
        image = image_batches[i][:image_per_batch, :].view(image_per_batch, channels[i], image_sizes[i], image_sizes[i]).cpu().detach()
        image = make_grid(image, nrow=num_row)
        image = image.permute(1, 2, 0)

        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].imshow(image)

        if titles is not None:
            axs[0][i].set_title(titles[i])

    if fig_title is not None:
        fig.suptitle(fig_title)

    if view:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def view_points_and_plots(points_list, points_labels, points_titles,
                          losses, loss_titles, logger, iterations,
                          prefix_exp, figsize=5):
    num_points_plot = len(points_list)
    num_loss_plot = len(losses)
    num_fig = num_points_plot + num_loss_plot

    fig, axs = plt.subplots(1, num_fig, figsize=(figsize * num_fig, figsize), squeeze=False)

    # PLOT POINTS
    for i in range(num_points_plot):
        points = points_list[i]
        labels = points_labels[i]
        name = points_titles[i]
        dim = points.shape[-1]

        if dim == 2:
            axs[0][i].scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        elif dim == 3:
            axs[0][i] = fig.add_subplot(1, num_fig, i + 1, projection='3d')
            axs[0][i].scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].view_init(elev=20, azim=-80)
            axs[0][i].set_title(name)
        elif dim == 1:
            axs[0][i].scatter(points[:, 0], np.zeros_like(points[:, 0]), c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        else:
            continue
            # raise NotImplementedError

    # PLOT LOSSES
    for i in range(num_loss_plot):
        axs[0][i + num_points_plot].plot(iterations, losses[i],
                                         color="#66892B", linewidth=1,
                                         label=loss_titles[i])

        axs[0][i + num_points_plot].set_xlim(-1, iterations[-1] + 1)
        axs[0][i + num_points_plot].set_title(loss_titles[i])

    plt.savefig(f"{prefix_exp}/output_points_and_plots_{'{:08d}'.format(iterations[-1])}.png", bbox_inches='tight')
    plt.close()

    with open(f"{prefix_exp}/log.json", 'w') as file:
        json.dump(logger, file)


def view_points_and_plots_with_anchors(points_list, points_labels, points_titles, anchors_list,
                                       losses, loss_titles, logger, iterations,
                                       prefix_exp, figsize=5):
    num_points_plot = len(points_list)
    num_loss_plot = len(losses)
    num_fig = num_points_plot + num_loss_plot

    fig, axs = plt.subplots(1, num_fig, figsize=(figsize * num_fig, figsize), squeeze=False)

    # PLOT POINTS
    for i in range(num_points_plot):
        points = points_list[i]
        labels = points_labels[i]
        name = points_titles[i]
        dim = points.shape[-1]

        anchors = anchors_list[i]

        if dim == 2:
            axs[0][i].scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        elif dim == 3:
            axs[0][i] = fig.add_subplot(1, num_fig, i + 1, projection='3d')
            axs[0][i].scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=plt.cm.Spectral, s=2)

            if anchors is not None:
                axs[0][i].scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], c="k", marker="o", s=200, depthshade=False)

            axs[0][i].view_init(elev=20, azim=-80)
            axs[0][i].set_title(name)
        elif dim == 1:
            axs[0][i].scatter(points[:, 0], np.zeros_like(points[:, 0]), c=labels, cmap=plt.cm.Spectral)
            axs[0][i].set_title(name)
        else:
            raise NotImplementedError

    # PLOT LOSSES
    for i in range(num_loss_plot):
        axs[0][i + num_points_plot].plot(iterations, losses[i],
                                         color="#66892B", marker="o", markersize=2, linewidth=1,
                                         label=loss_titles[i])

        axs[0][i + num_points_plot].set_xlim(-1, iterations[-1] + 1)
        axs[0][i + num_points_plot].set_title(loss_titles[i])

    plt.savefig(f"{prefix_exp}/output_points_and_plots_{'{:08d}'.format(iterations[-1])}.png", bbox_inches='tight')
    plt.close()

    with open(f"{prefix_exp}/log.json", 'w') as file:
        json.dump(logger, file)


def view_images_and_losses(num_grid, images, image_titles, image_sizes, image_channels,
                           losses, loss_titles, logger, iterations,
                           prefix_exp, nrow=4, figsize=5):
    num_image_plot = len(images)
    num_loss_plot = len(losses)
    num_fig = num_image_plot + num_loss_plot

    fig, axs = plt.subplots(1, num_fig, figsize=(figsize * num_fig, figsize), squeeze=False)

    # PLOT IMAGES
    for i in range(num_image_plot):
        image = images[i][:num_grid, :].view(num_grid, image_channels[i], image_sizes[i], image_sizes[i])
        image = make_grid(image, nrow=nrow)
        image = image.permute(1, 2, 0)

        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].imshow(image)
        axs[0][i].set_title(image_titles[i])

    # PLOT LOSSES
    for i in range(num_loss_plot):
        axs[0][i + num_image_plot].plot(iterations, losses[i],
                                        color="#66892B", linewidth=1,
                                        label=loss_titles[i])

        axs[0][i + num_image_plot].set_xlim(-1, iterations[-1] + 1)
        axs[0][i + num_image_plot].set_title(loss_titles[i])

    plt.savefig(f"{prefix_exp}/output_{'{:08d}'.format(iterations[-1])}.png", bbox_inches='tight')
    plt.close()

    with open(f"{prefix_exp}/log.json", 'w') as file:
        json.dump(logger, file)


def to_gif(frame_dir, save_path="out.gif", fps=12, skip=1):
    images = []
    file_names = get_all_files(frame_dir, keep_dir=True, sort=True)
    file_names = file_names[1:]

    num_file = len(file_names)

    for i in range(num_file):
        if i % skip != 0 and i != num_file - 1:
            continue

        filename = file_names[i]

        image = imageio.imread(filename)[:, :, :3]
        H, W, C = image.shape
        Hb = 10

        percentage = int((1.0 * (i + 1) * W) / num_file)

        progress_bar = np.concatenate([
            np.ones((Hb, W, 1)) * 255,
            np.zeros((Hb, W, 1)),
            np.zeros((Hb, W, 1))
        ], axis=-1)

        progress_bar[:, percentage:, :] = 255

        progress_bar = progress_bar.astype("uint8")

        image = np.concatenate([
            image,
            progress_bar
        ], axis=0)

        images.append(image)

    imageio.mimsave(save_path, images, fps=fps)


def plot_2d_distributions(points_list, points_titles,
                          iteration=None, save_dir="./", figsize=5, save=True, view=False):
    num_plot = len(points_list)
    fig, axs = plt.subplots(1, num_plot, figsize=(figsize * num_plot, figsize), squeeze=False)

    for i in range(num_plot):
        points = points_list[i]
        name = points_titles[i]

        sns.kdeplot(points[:, 0], points[:, 1], shade=True, cmap='Blues', n_levels=20, legend=False)
        # axs.set_xlim([-4, 4])
        # axs.set_ylim([-4, 4])
        axs[0][i].set_title(name)

    if save:
        plt.savefig(f"{save_dir}/output_points_{'{:08d}'.format(iteration)}.png", bbox_inches='tight')

    if view:
        plt.show()

    plt.close()
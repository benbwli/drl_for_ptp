from evaluation.trajectory_utils import prediction_output_to_trajectories
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns


def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False,
                      cp_radius_dict=None):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color=cmap[node.type.value],
                    linewidth=line_width, alpha=line_alpha)

        # Ground truth future (drawn once per node, outside sample loop)
        ax.plot(future[:, 0],
                future[:, 1],
                'w--',
                path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

        # Current Node Position (drawn once per node)
        circle = plt.Circle((history[-1, 0],
                             history[-1, 1]),
                            node_circle_size,
                            facecolor='g',
                            edgecolor='k',
                            lw=circle_edge_width,
                            zorder=3)
        ax.add_artist(circle)
        
        # --- Draw the CP Safety Shield (Red Keep-Out Zone) ---
        # Draw a circle around EACH of the 20 trajectory endpoints
        # The union of all circles = the total multimodal Keep-Out Zone
        if cp_radius_dict is not None and node in cp_radius_dict:
            radius = cp_radius_dict[node]
            
            for sample_idx in range(predictions.shape[1]):
                endpoint = predictions[batch_num, sample_idx, -1, :]  # final position of this trajectory
                
                # Draw the shaded red safety circle
                cp_circle = plt.Circle((endpoint[0], endpoint[1]),
                                       radius,
                                       facecolor='red',
                                       alpha=0.03,
                                       zorder=2)
                ax.add_artist(cp_circle)
                
                # Draw the dashed outline
                cp_outline = plt.Circle((endpoint[0], endpoint[1]),
                                        radius,
                                        edgecolor='red',
                                        fill=False,
                                        linestyle='--',
                                        linewidth=0.5,
                                        alpha=0.4,
                                        zorder=2)
                ax.add_artist(cp_outline)

    ax.axis('equal')


def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         cp_radius_dict=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, cp_radius_dict=cp_radius_dict, **kwargs)
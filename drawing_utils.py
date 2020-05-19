import os

import numpy as np



def draw_diagram(voice_positions, candidate_angles, angle_window_size, output_file):
    """
    Draws the setup of all the voices in space, and colored triangles for the beams
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection
    matplotlib.style.use('ggplot')

    fig, ax = plt.subplots()
    ax.set(xlim=(-5, 5), ylim = (-5, 5))
    ax.set_aspect("equal")
    ax.axhline()
    ax.axvline()

    plt.tick_params(axis='both',
        which='both', bottom='off',
        top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )

    for pos in voice_positions:
        if pos[0] != 0.0:
            a_circle = plt.Circle((pos[0], pos[1]), 0.3, color='b', fill=False)
            ax.add_artist(a_circle)

    patches = []
    for target_angle in candidate_angles:
        vertices = angle_to_triangle(target_angle, angle_window_size) * 4.96
        #patches.append(Polygon(vertices*5, True))
        ax.fill(vertices[:, 0], vertices[:, 1], edgecolor='black', linewidth=2, alpha=0.4)
    #p = PatchCollection(patches, alpha=0.4)
    #ax.add_collection(p)

    plt.savefig(output_file)


def angle_to_triangle(target_angle, angle_window_size):
    first_point = [0,0]
    second_point = angle_to_point(np.clip(target_angle - angle_window_size/2, -np.pi, np.pi))
    third_point = angle_to_point(np.clip(target_angle + angle_window_size/2, -np.pi, np.pi))

    return(np.array([first_point, second_point, third_point]))



def angle_to_point(angle):
    """Angle must be -pi to pi"""
    if -np.pi <= angle < -3*np.pi/4:
        return[-1, -np.tan(angle + np.pi)]

    elif -3*np.pi/4 < angle < -np.pi/2:
        return[-np.tan(-np.pi/2 - angle), -1]

    elif -np.pi/2 < angle < -np.pi/4:
        return[np.tan(angle + np.pi/2), -1]

    elif -np.pi/4 < angle < 0:
        return[1, -np.tan(-angle)]

    elif 0 < angle < np.pi / 4:
        return [1, np.tan(angle)]

    elif np.pi/4 < angle < np.pi/2:
        return [np.tan(np.pi/2 - angle), 1]

    elif np.pi/2 < angle < 3*np.pi/4:
        return [-np.tan(angle - np.pi/2), 1]

    elif 3*np.pi/4 < angle <= np.pi:
        return [-1, np.tan(np.pi - angle)]
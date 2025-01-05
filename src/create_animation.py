import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from Gaussian import Gaussian
from Player import Player
from Dartboard import Dartboard

def create_animation(output_path, board):
    fps = 30
    nSeconds = 10
    std_max = 80
    std_min = 2

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(8,8), dpi=120)
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    player = Player(Gaussian(mean=[0, 0], std=[std_min, std_min], corr=0))
    expected_scores, (i_max, ii_max) = board.get_all_expected_scores(player)
    im = ax.imshow(expected_scores, cmap='viridis', interpolation="none")
    ax.set_title(f"Expected Score Distribution - $\\sigma$ = {std_min} mm")
    scat = ax.scatter(i_max, ii_max, c="r", marker="o", s=100, label="Max")

    overlay_board = board.outline
    mask = np.ma.masked_where(overlay_board < 0.5, overlay_board)

    ax.imshow(mask, cmap='binary', interpolation="none")
    fig.colorbar(im, cax=cax, orientation='vertical', label="Expected Score")
    ax.axis("off")
    fig.subplots_adjust(left=0.05, bottom=0, right=0.90, top=1, wspace=None, hspace=None)

    # define the animation function
    def animate_func(i):

        std = std_min + (std_max - std_min) * ((i)/(fps*nSeconds))**2
        player = Player(Gaussian(mean=[0, 0], std=[std, std], corr=0))
        expected_scores, (i_max, ii_max) = board.get_all_expected_scores(player)
        im.set_array(expected_scores)
        # im.autoscale()
        if i % 3 == 0:
            ax.set_title(f"σ = {round(std, 1)} mm")
        scat.set_offsets([i_max, ii_max])
        return [im, scat]

    # run the animation
    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = int(nSeconds * fps),
                                interval = 1000 / fps,
                                    )

    # save the output
    writer=animation.PillowWriter(fps=fps)

    anim.save(output_path, writer=writer)

    print('Done!')

if __name__ == "__main__":
    board = Dartboard()
    create_animation("animation_example.gif", board)
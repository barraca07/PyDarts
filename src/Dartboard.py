import config
import numpy as np
import math
from scipy.signal import fftconvolve
from utils import plot_score_distribution

# calculate the index of the middle of the dartboard and the number of px per mm
MIDDLE = (config.RESOLUTION-1)/2
PX_PER_MM = (config.RESOLUTION-1)/(2*config.OUTER_RADIUS)

class Dartboard:
    def __init__(self):

        self.scores, self.labels = self._get_scores()
        self.outline = self._get_outline()

    def _get_scores(self):
        values = np.array([6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10])

        # Generate a grid of x, y coordinates
        x, y = np.meshgrid(np.arange(config.RESOLUTION), np.arange(config.RESOLUTION))

        # Calculate x and y differences from the center
        x_diff = x - MIDDLE
        y_diff = MIDDLE - y

        # Calculate radius and angles
        r = np.sqrt(x_diff**2 + y_diff**2) / PX_PER_MM
        angle = np.degrees(np.arctan2(y_diff, x_diff))

        # Determine the segment index for each angle
        segment_indices = np.floor((angle + 9) / 18).astype(int)
        segment_values = values[segment_indices]

        # Create masks for different regions of the dartboard

        outer_bull_mask = (r >= config.BULL_RADIUS) & (r < config.OUTER_BULL_RADIUS)
        miss_mask = r > config.OUTER_RADIUS
        treble_mask = (r <= config.TREBLE_RADIUS) & (r > config.TREBLE_RADIUS - config.DOUBLE_TREBLE_THICKNESS)
        double_mask = (r <= config.OUTER_RADIUS) & (r > config.OUTER_RADIUS - config.DOUBLE_TREBLE_THICKNESS)

        # Initialize scores and labels arrays
        scores = np.zeros_like(r, dtype=float)
        labels = np.empty_like(r, dtype=object)

        # Assign values based on regions
        bull_mask = r < config.BULL_RADIUS
        scores[bull_mask] = 50
        labels[bull_mask] = "B"

        outer_bull_mask = (r >= config.BULL_RADIUS) & (r < config.OUTER_BULL_RADIUS)
        scores[outer_bull_mask] = 25
        labels[outer_bull_mask] = "O"

        labels[miss_mask] = "M"

        treble_indices = treble_mask & ~bull_mask & ~outer_bull_mask & ~miss_mask
        double_indices = double_mask & ~bull_mask & ~outer_bull_mask & ~miss_mask

        scores[treble_indices] = segment_values[treble_indices] * 3
        labels[treble_indices] = np.core.defchararray.add("T", segment_values[treble_indices].astype(str))

        scores[double_indices] = segment_values[double_indices] * 2
        labels[double_indices] = np.core.defchararray.add("D", segment_values[double_indices].astype(str))

        regular_indices = ~(bull_mask | outer_bull_mask | miss_mask | treble_mask | double_mask)
        scores[regular_indices] = segment_values[regular_indices]
        labels[regular_indices] = segment_values[regular_indices].astype(str)

        return scores, labels
    
    def _get_outline(self):

        # Generate a grid of x, y coordinates
        x, y = np.meshgrid(np.arange(config.RESOLUTION), np.arange(config.RESOLUTION))

        # Calculate x and y differences from the center
        x_diff = x - MIDDLE
        y_diff = MIDDLE - y

        # Calculate radius and angles
        r = np.sqrt(x_diff**2 + y_diff**2) / PX_PER_MM
        r[r == 0] = 1e-2
        angle = np.degrees(np.arctan2(y_diff, x_diff))

        # Determine the segment index for each angle
        segment = (angle + 9) / 18

        lines = (np.abs(config.BULL_RADIUS - r) < 0.5) | (np.abs(config.OUTER_BULL_RADIUS - r) < 0.5) \
        | (np.abs(config.TREBLE_RADIUS - r) < 0.5) | (np.abs(config.OUTER_RADIUS - r) < 0.5) \
        | (np.abs(config.OUTER_RADIUS - config.DOUBLE_TREBLE_THICKNESS - r) < 0.5) \
        | (np.abs(config.TREBLE_RADIUS - config.DOUBLE_TREBLE_THICKNESS - r) < 0.5) \
        | ((r > config.OUTER_BULL_RADIUS) & (r < config.OUTER_RADIUS) & (segment % 1 < 3/r))

        return lines.astype(int)


    def generate_mask(self, label):

        valid_labels = np.unique(self.labels)

        assert label in valid_labels

        mask = (self.labels == label).astype(int)
        
        return mask
    
    def get_expected_score(self, player, aim_location):
        x_aim, y_aim = aim_location

        x_aim_px = x_aim * PX_PER_MM
        y_aim_px = y_aim * PX_PER_MM
        
        i = round(MIDDLE - y_aim_px)
        ii = round(MIDDLE + x_aim_px)

        expected_scores, _ = self.get_all_expected_scores(player)

        return expected_scores[i, ii]

    def get_all_expected_scores(self, player):
        expected_scores = fftconvolve(self.scores, player.accuracy.pdf, mode="same")
        max_index = np.argmax(expected_scores)

        optimum_aim_location = max_index % config.RESOLUTION, max_index // config.RESOLUTION
        return expected_scores, optimum_aim_location

    def prob_single_target(self, player, segment):
        mask = self.generate_mask(segment)

        probabilities = fftconvolve(mask, player.accuracy.pdf, mode="same")

        return np.max(probabilities)
    
    def prob_multi_target(self, player, segments):
        probability = 1

        for segment in segments:
            probability *= self.prob_single_target(player, segment)

        return probability
    
    def simulate_throw(self, player, aim_location, num_samples=1):

        samples = np.random.multivariate_normal(player.accuracy.mean, player.accuracy.cov, num_samples)
        
        locations = np.array(aim_location) + samples

        locations_px = locations * PX_PER_MM

        indices = np.round(MIDDLE + locations_px[:, [1, 0]]*[-1, 1]).astype(int)

        scores = self.scores[indices[:, 0], indices[:, 1]]

        return scores
        
if __name__ == "__main__":
    from Player import Player
    from Gaussian import Gaussian
    import matplotlib.pyplot as plt
    
    board = Dartboard()

    player = Player(Gaussian(mean=[0, 0], std=[20, 20], corr=0))

    scores = board.simulate_throw(player, [0, 0], num_samples=1000)

    scores_avg = np.average(scores)
    print(f"Average Score: {scores_avg}")
    exp_score = board.get_expected_score(player, [0, 0])
    print(f"Expected Score: {exp_score}")

    plot_score_distribution(board.generate_mask("20"), board.outline)
    scores, (i_max, ii_max) = board.get_all_expected_scores(player)
    plot_score_distribution(scores, board.outline)
    plt.scatter(i_max, ii_max, c="r", marker="o", label="Best Aiming Location")
    plt.legend(loc="lower left")
    plt.show()
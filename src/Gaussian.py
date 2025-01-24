import config
import numpy as np
import scipy.stats as stats

MIDDLE = (config.RESOLUTION-1)/2
PX_PER_MM = (config.RESOLUTION-1)/(2*config.OUTER_RADIUS)

class Gaussian:
    def __init__(self, mean=None, std=None, corr=None):
        self.pdf = None

        self.mean = mean
        self.std = std
        self.corr = corr
        self.cov = None

        if mean is not None and std is not None and corr is not None:
            self.update_pdf(mean, std, corr)
    
    def update_pdf(self, mean, std, corr):

        # convert parameters from mm to px
        mean_px = np.array(mean)*PX_PER_MM
        std_px = np.array(std)*PX_PER_MM

        # create a grid of x, y points at which to sample the probability distribution
        x, y = np.mgrid[0:config.RESOLUTION:1, 0:config.RESOLUTION:1]
        points = np.dstack((x, y))

        # create a gaussian probability distribution
        gaussian = stats.multivariate_normal([MIDDLE-mean_px[1], MIDDLE+mean_px[0]], [[std_px[1]**2, -corr*std_px[1]*std_px[0]], [-corr*std_px[1]*std_px[0], std_px[0]**2]])
        
        # sample the probability distribution according to the grid
        self.pdf = gaussian.pdf(points)

        self.mean = mean
        self.std = std
        self.corr = corr
        self.cov = [[std[0]**2, corr*std[0]*std[1]], [corr*std[0]*std[1], std[1]**2]]

        return self

    def get_parameters_from_samples(self, samples):

        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        corr = np.cov(samples, rowvar=False)[0, 1]/(std[0]*std[1])

        return mean, std, corr
    
if __name__ == "__main__":


    import matplotlib.pyplot as plt
    from utils import generate_test_samples, plot_accuracy_distribution
    from Dartboard import Dartboard

    board = Dartboard()

    test_samples = generate_test_samples(mean_x=10, mean_y=-5, std_x=30, std_y=50, corr=-0.2, num_samples=75)
    print(test_samples)
    gaussian = Gaussian()
    mean, std, corr = gaussian.get_parameters_from_samples(test_samples)
    print(f"Estimated Mean: {mean}")
    print(f"Estimated Std: {std}")
    print(f"Estimated Corr: {corr}")
    gaussian.update_pdf(mean, std, corr)

    plot_accuracy_distribution(gaussian.pdf, board.outline)
    plt.scatter(MIDDLE, MIDDLE, c="r", marker="x", label="Target Location")
    plt.scatter(MIDDLE+test_samples[:, 0]*PX_PER_MM, MIDDLE-test_samples[:, 1]*PX_PER_MM, c="white", edgecolor='black', marker="o", label="Samples", s=10)
    plt.legend(loc="lower left")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    plt.show()
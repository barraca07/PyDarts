import matplotlib.pyplot as plt
import numpy as np

def generate_test_samples(mean_x, mean_y, std_x, std_y, corr, num_samples):
    samples = np.random.multivariate_normal([mean_x, mean_y], [[std_x**2, corr*std_y*std_x], [corr*std_y*std_x, std_y**2]], num_samples)
    return samples

def plot_accuracy_distribution(prob, outline):
    fig = plt.figure(figsize=(8.4, 6.8), dpi=100)
    ax = fig.add_subplot()
    ax.set_title('')
    plt.imshow(prob)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation='vertical', label="Probability Density")
    mask_outline = np.ma.masked_where(outline < 0.5, outline)
    plt.imshow(mask_outline, cmap="binary", interpolation="none", alpha=0.5)
    
def plot_score_distribution(scores, outline):
    fig = plt.figure(figsize=(8.4, 6.8), dpi=100)
    ax = fig.add_subplot()
    ax.set_title('')
    plt.imshow(scores)
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation='vertical', label="Expected Score")
    mask_outline = np.ma.masked_where(outline < 0.5, outline)
    plt.imshow(mask_outline, cmap="binary", interpolation="none", alpha=0.5)
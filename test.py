import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def make_scaled_moons(n_samples=100, noise=0.1, scale_ratio=1.0, random_state=None):
    """Generate two moons with different sizes that don't intersect"""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Separate the moons
    mask_0 = (y == 0)
    mask_1 = (y == 1)
    
    # Get the center of each moon before transformation
    center0 = np.mean(X[mask_0], axis=0)
    center1 = np.mean(X[mask_1], axis=0)
    
    # Move the second moon up
    X[mask_1, 1] -= 0.5

    # Scale the second moon
    X[mask_1] = (X[mask_1] - center1) * scale_ratio + center1
    
    # Calculate vertical separation needed based on scale
    base_separation = 0.5  # minimum separation
    scale_adjustment = max(scale_ratio, 1.0) * 0.5

    
    # # If second moon is smaller, move it further to prevent intersection
    # if scale_ratio < 1.0:
    #     X[mask_1, 1] += (1 - scale_ratio) * 0.5
    
    return X, y

def plot_scaled_moons():
    """Plot non-intersecting moons with different scale ratios"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scale_ratios = [0.5, 1.0, 2.0]
    
    for ax, scale in zip(axes, scale_ratios):
        X, y = make_scaled_moons(n_samples=200, scale_ratio=scale, noise=0.1, random_state=42)
        
        # Plot with distinct colors
        ax.scatter(X[y==0, 0], X[y==0, 1], c='royalblue', label='Moon 1', s=50, alpha=0.6)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='crimson', label='Moon 2', s=50, alpha=0.6)
        
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Scale Ratio: {scale}')
        ax.legend()
        ax.set_aspect('equal')
        
        # Adjust limits based on scale ratio
        ax.set_xlim(-2, 3)
        max_y = 2 + max(scale, 1.0)
        ax.set_ylim(-1, max_y)

    plt.tight_layout()
    plt.show()

# Generate and display the plots
plot_scaled_moons()

    




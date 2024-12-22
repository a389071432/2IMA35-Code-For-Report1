import numpy as np
from sklearn.utils import check_random_state
from sklearn.datasets import make_circles, make_moons, make_blobs, make_swiss_roll, make_s_curve
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial.distance import cdist
from typing import Tuple, List


def generate_points_with_distance_control(min_dist, max_dist, x_range=(-2, 2), y_range=(-2, 2), max_samples=1000, random_state=None):
    """
    Generate points with controlled inter-point distances using Poisson disk sampling.
    
    Parameters:
    -----------
    min_dist : float
        Minimum distance between points
    max_dist : float
        Maximum distance between points
    x_range : tuple
        Range for x coordinates (min_x, max_x)
    y_range : tuple
        Range for y coordinates (min_y, max_y)
    max_samples : int
        Maximum number of points to generate
    random_state : int or None
        Random seed for reproducibility

    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize parameters
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    
    # Cell size for spatial subdivision (optimization)
    cell_size = min_dist / np.sqrt(2)
    
    # Initialize grid for spatial subdivision
    cols = int(np.ceil(width / cell_size))
    rows = int(np.ceil(height / cell_size))
    grid = np.full((rows, cols), -1, dtype=int)
    
    # Lists to store points and active samples
    points = []
    active = []
    
    # Helper function to get grid coordinates
    def get_cell(pt):
        x = int((pt[0] - x_range[0]) / cell_size)
        y = int((pt[1] - y_range[0]) / cell_size)
        return min(max(x, 0), cols-1), min(max(y, 0), rows-1)
    
    # Helper function to check if point is valid
    def is_valid(pt, points_array):
        if not (x_range[0] <= pt[0] <= x_range[1] and y_range[0] <= pt[1] <= y_range[1]):
            return False
        
        if len(points_array) == 0:
            return True
        
        # Check distances to existing points
        distances = cdist([pt], points_array)[0]
        return (np.all(distances >= min_dist) and 
                (max_dist >= width + height or  # If max_dist is very large, ignore it
                 np.any(distances <= max_dist)))
    
    # Generate initial point
    initial_pt = np.array([
        np.random.uniform(x_range[0], x_range[1]),
        np.random.uniform(y_range[0], y_range[1])
    ])
    
    points.append(initial_pt)
    active.append(initial_pt)
    x, y = get_cell(initial_pt)
    grid[y, x] = 0
    
    # Main generation loop
    while len(active) > 0 and len(points) < max_samples:
        # Pick a random active point
        idx = np.random.randint(len(active))
        pt = active[idx]
        
        # Try to generate new points around it
        found_valid = False
        for _ in range(30):  # Number of attempts per active point
            # Generate random point at distance between min_dist and max_dist
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(min_dist, min(max_dist, width + height))
            new_pt = pt + distance * np.array([np.cos(angle), np.sin(angle)])
            
            if is_valid(new_pt, np.array(points)):
                points.append(new_pt)
                active.append(new_pt)
                x, y = get_cell(new_pt)
                grid[y, x] = len(points) - 1
                found_valid = True
                break
        
        if not found_valid:
            active.pop(idx)
    
    return np.array(points)


def generate_points_with_density(x_range, y_range, density):
    """
    Generate evenly distributed points with complete coverage.
    
    Parameters:
    -----------
    x_range : tuple
        Range for x coordinates (min_x, max_x)
    y_range : tuple
        Range for y coordinates (min_y, max_y)
    density : float
        Points per unit area
    
    Returns:
    --------
    points : np.ndarray of shape (N, 2)
        Generated points
    avg_dist : float
        Average distance to nearest neighbor
    """
    # Calculate parameters
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    area = width * height
    n_target = int(area * density)
    
    # Calculate radius based on density
    radius = 0.8 / np.sqrt(density)  # Adjusted factor for better coverage
    
    # Grid for acceleration
    cell_size = radius / np.sqrt(2)
    cols = int(np.ceil(width / cell_size))
    rows = int(np.ceil(height / cell_size))
    grid = {}
    
    def get_cell_indices(pt):
        x_idx = int((pt[0] - x_range[0]) / cell_size)
        y_idx = int((pt[1] - y_range[0]) / cell_size)
        return max(0, min(x_idx, cols - 1)), max(0, min(y_idx, rows - 1))
    
    def add_to_grid(pt, idx):
        cell = get_cell_indices(pt)
        if cell not in grid:
            grid[cell] = []
        grid[cell].append(idx)
    
    def get_nearby_points(pt):
        cell_x, cell_y = get_cell_indices(pt)
        indices = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                cell = (cell_x + i, cell_y + j)
                if cell in grid:
                    indices.extend(grid[cell])
        return indices
    
    def check_valid_point(pt, points_array, radius_check):
        if len(points_array) == 0:
            return True
            
        nearby_indices = get_nearby_points(pt)
        if not nearby_indices:
            return True
            
        distances = cdist([pt], points_array[nearby_indices])[0]
        return np.all(distances >= radius_check)
    
    def try_global_placement():
        """Try to place a point anywhere in the region."""
        for _ in range(100):  # Number of attempts
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            pt = np.array([x, y])
            
            if check_valid_point(pt, np.array(points), radius):
                return pt
        return None
    
    points = []
    active = []
    
    # Place first point
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    first_point = np.array([x, y])
    points.append(first_point)
    active.append(0)
    add_to_grid(first_point, 0)
    
    # Main point generation loop
    global_attempts = 0
    while len(points) < n_target and global_attempts < n_target * 10:
        point_added = False
        
        # Try local placement from active points
        if active:
            idx = np.random.randint(len(active))
            base_point = points[active[idx]]
            
            for _ in range(30):
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(radius, 2*radius)
                new_point = base_point + r * np.array([np.cos(angle), np.sin(angle)])
                
                x, y = new_point
                if (x_range[0] <= x <= x_range[1] and 
                    y_range[0] <= y <= y_range[1] and 
                    check_valid_point(new_point, np.array(points), radius)):
                    
                    points.append(new_point)
                    active.append(len(points) - 1)
                    add_to_grid(new_point, len(points) - 1)
                    point_added = True
                    break
            
            if not point_added:
                active.pop(idx)
        
        # If local placement fails or no active points, try global placement
        if not point_added:
            new_point = try_global_placement()
            if new_point is not None:
                points.append(new_point)
                active.append(len(points) - 1)
                add_to_grid(new_point, len(points) - 1)
                point_added = True
        
        if not point_added:
            global_attempts += 1
    
    points = np.array(points)
    
    # Calculate average nearest neighbor distance
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    nearest_distances = np.min(distances, axis=1)
    avg_dist = np.mean(nearest_distances)
    
    return points, avg_dist


def point_to_boundary_distance(x0, y0, boundary_func, x_range, offset=0):
    """
    Calculate shortest distance from point (x0,y0) to the boundary curve.
    Uses a discrete approximation by sampling many points on the curve.
    """
    # Sample many points along the boundary
    x_samples = np.linspace(x_range[0], x_range[1], 3000)
    y_samples = boundary_func(x_samples, offset)
    
    # Calculate distances to all sampled boundary points
    distances = np.sqrt((x_samples - x0)**2 + (y_samples - y0)**2)
    
    # Return minimum distance
    return np.min(distances)


def make_blur(x_range, n_samples, boundary_func, boundary_width, blur):
    N = n_samples
    # upper boundary
    x_samples = np.linspace(x_range[0], x_range[1], N)
    y_center = boundary_func(x_samples)+boundary_width/2
    noises = []
    yy = []
    for y in y_center:
        while True:
            noise = np.random.normal(loc=y, scale=blur)
            if abs(noise-y)<boundary_width/2:
                break
        noises.append(noise)
        yy.append(0)

    
    # lower boundary
    y_center = boundary_func(x_samples)-boundary_width/2
    for y in y_center:
        while True:
            noise = np.random.normal(loc=y, scale=blur)
            if abs(noise-y)<boundary_width/2:
                break
        noises.append(noise)
        yy.append(1)

    points = np.zeros((2*N,2))
    points[:N,0] = x_samples
    points[N:2*N,0] = x_samples
    points[:,1] = noises

    plt.scatter(points[:, 0], points[:, 1], c=yy, cmap=plt.cm.Spectral, s=3)
    plt.show()

    return points



def make_blur_boundaries(n_samples=1000, boundary_width=0.1, blur=0.1, boundary_type='straight', random_state=None, min_dist=0.099, max_dist=0.1):
    """
    Generate two point sets separated by a perfectly matched boundary.
    Returns format matching sklearn's example datasets.
    """
    rng = check_random_state(random_state)
    
    # Generate uniform points in rectangle
    # X = rng.uniform(-2, 2, size=(n_samples, 2))
    X = generate_points_with_distance_control(
        min_dist=min_dist,
        max_dist=max_dist,
        x_range=(-2, 2),
        y_range=(-2, 2),
        max_samples=n_samples,
        random_state=random_state
    )
    
    # Define boundary function
    if boundary_type == 'straight':
        def boundary_func(x):
            return np.zeros_like(x)
    elif boundary_type == 'sine':
        def boundary_func(x):
            return 0.5 * np.sin(2 * np.pi * x / 4)
    elif boundary_type == 'interleaved':
        def boundary_func(x):
            # Increase frequency and amplitude gradually
            freq = 4 * np.pi  # higher frequency for more oscillations
            # varying_amplitude = 0.3 * (1 + np.abs(x/2))  # amplitude increases with |x|
            varying_amplitude = 0.95 * (1 + np.abs(x/2))  # amplitude increases with |x|
            return varying_amplitude * np.sin(freq * x)
        
    # Calculate true shortest distance for each point
    distances = np.array([point_to_boundary_distance(x, y, boundary_func, (-2,2)) 
                         for x, y in X])
    
    # Remove points within boundary width
    mask_keep = distances >= boundary_width/2
    X_filtered = X[mask_keep]

    # Apply blurring to boundaries
    noise_points = make_blur((-2,2), 200, boundary_func,boundary_width, blur)
    X_final = np.vstack((X_filtered, noise_points))

    # Assign labels based on comparison with boundary
    y_boundary = boundary_func(X_final[:, 0])
    y = (X_final[:, 1] > y_boundary).astype(int)

    return ((X_final, y), {'damping': .75, 'preference': -220, 'n_clusters': 2})



def make_matched_boundaries(n_samples=1000, boundary_width=0.1, boundary_type='straight', random_state=None, min_dist=0.099, max_dist=0.1):
    """
    Generate two point sets separated by a perfectly matched boundary.
    Returns format matching sklearn's example datasets.
    """
    rng = check_random_state(random_state)

    X, avg_dist = generate_points_with_density(
        x_range=(-2, 2),
        y_range=(-2, 2),
        density=50.0  # points per unit area
    )
    print(f'avg dis: {avg_dist}')
    
    # Define boundary function
    if boundary_type == 'straight':
        def boundary_func(x, offset):
            return np.zeros_like(x) + offset
    elif boundary_type == 'sine':
        def boundary_func(x, offset):
            return 0.5 * np.sin(2 * np.pi * x / 4) + offset
    elif boundary_type == 'interleaved':
        def boundary_func(x, offset):
            # # Increase frequency and amplitude gradually
            # freq = 4 * np.pi  # higher frequency for more oscillations
            # varying_amplitude = 0.4 * (1 + np.abs(x/2))  # amplitude increases with |x|
            # return varying_amplitude * np.sin(freq * x) + offset
            return 1.0 * np.sin(1.0 * np.pi * x) + offset
        
    distances = np.array([point_to_boundary_distance(X[i][0], X[i][1], boundary_func, (-2,2), 0) 
                         for i in range(X.shape[0])])
    
    # Remove points within boundary width
    mask_keep = distances >= boundary_width/2
    X = X[mask_keep]

    # filter again
    distances = np.array([point_to_boundary_distance(X[i][0], X[i][1], boundary_func, (-2,2), boundary_width/2) 
                         for i in range(X.shape[0])])
    mask_keep = distances >= min_dist/4
    X = X[mask_keep]

    distances = np.array([point_to_boundary_distance(X[i][0], X[i][1], boundary_func, (-2,2), -boundary_width/2) 
                         for i in range(X.shape[0])])
    mask_keep = distances >= min_dist/4
    X = X[mask_keep]

    # place points on boundaries
    x_samples = np.linspace(-2, 2, int(4/max_dist))
    y_samples = boundary_func(x_samples, boundary_width/2)
    bound_points = np.stack((x_samples, y_samples), axis=1)
    X = np.concatenate((X, bound_points), axis=0)
    
    # Assign labels based on comparison with boundary
    y_boundary = boundary_func(X[:, 0], 0)
    y = (X[:, 1] > y_boundary).astype(int)
    
    # Return as tuple of (X, y)
    return ((X, y), {'damping': .75, 'preference': -220, 'n_clusters': 2})


def make_close_concentric(n_samples=1000, r=0.5, boundary_width=0.1, random_state=None, min_dist=0.099, max_dist=0.1):
    """
    Generate two point sets separated by a perfectly matched boundary.
    Returns format matching sklearn's example datasets.
    """
    rng = check_random_state(random_state)
    

    X, avg_dist = generate_points_with_density(
        x_range=(-2, 2),
        y_range=(-2, 2),
        density=50.0  # points per unit area
    )
    print(f'avg dis: {avg_dist}')

    theta = np.random.uniform(0, 2*np.pi, 3000)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    def point_bound_distance(x0,y0):
        dis = np.sqrt((x - x0)**2 + (y - y0)**2)
        return np.min(dis)

    distances = np.array([point_bound_distance(x,y) for x,y in X])
    mask_keep = distances >= boundary_width/2
    X = X[mask_keep] 

    # Assign labels based on comparison with boundary
    y = (X[:, 0]**2+X[:, 1]**2 > r**2).astype(int)
    
    # Return as tuple of (X, y)
    return ((X, y), {'damping': .75, 'preference': -220, 'n_clusters': 2})


def make_simple_diffusion(n_points=1000, decay=2.0):
    # # Generate x-coordinates with higher density on the left
    # x = np.random.exponential(scale=decay, size=n_points)  # Exponential decay

    # # Normalize x to fit within [0, 1] and then stretch to a desired range
    # x = x / max(x)  

    x = np.linspace(0, 1, n_points)
    x = x**decay
    scale = 0.45
    x = scale*x/max(x)

    # Generate corresponding y-coordinates randomly within a range
    y = np.random.uniform(0, 1, size=n_points)

    # # Optional: Increase variability with blobs for clustering appearance
    # x_blob, y_blob = make_blobs(n_samples=n_points, centers=3, cluster_std=0.05, center_box=(0, 1))
    # y += y_blob / 3  


    # the mirror
    x1 = 1-x
    y1 = np.random.uniform(0, 1, size=n_points)
    # x1_blob, y1_blob = make_blobs(n_samples=n_points, centers=3, cluster_std=0.05, center_box=(0, 1))
    # y1 += y1_blob / 3  

    X = np.concatenate((x, x1))
    Y = np.concatenate((y, y1))
    Z = np.array([np.array([X[i],Y[i]]) for i in range(len(X))])

    labels = [0]*n_points + [1]*n_points

    return ((Z, labels), {'damping': .75, 'preference': -220, 'n_clusters': 2})


def sample_uniform_annulus(center, r_inner, r_outer, n_points, seed):
    """
    Sample points uniformly from an annulus with specified density.
    
    Parameters:
    -----------
    r_inner : float
        Inner radius of annulus
    r_outer : float
        Outer radius of annulus
    density : float
        Points per unit area
    """

    if seed is not None:
        random.seed(seed)
        
    # Calculate annulus area
    area = math.pi * (r_outer**2 - r_inner**2)
        
    points = []
    for _ in range(n_points):
        # Generate random radius with sqrt for uniform area distribution
        r = math.sqrt(random.uniform(r_inner**2, r_outer**2))
        
        # Generate random angle
        theta = random.uniform(0, 2 * math.pi)
        
        # Convert to Cartesian coordinates
        x = center[0] + r * math.cos(theta)
        y = center[1] + r * math.sin(theta)
        points.append((x, y))
    
    return points


def sample_uniform_rectangle(N, L, R):   
    # random.seed(42)   

    n_points = int(N)
    points = []
    for _ in range(n_points):        
        x = random.uniform(L, R)
        y = random.uniform(0, 1)
        points.append((x, y))
    
    return points


def sample_ellipse(N, a, b, center=(0,0), random_state=None):
    """
    Sample N points uniformly from an ellipse.
    
    Parameters:
    -----------
    N : int
        Number of points to sample
    a : float
        Semi-major axis length
    b : float
        Semi-minor axis length
    center : tuple of float
        Center coordinates (x, y)
    random_state : int or None
        Random seed for reproducibility
    """

    # np.random.seed(42)
    
    # Generate random angles
    theta = np.random.uniform(0, 2*np.pi, N)
    
    # Generate random radii with sqrt for uniform distribution
    r = np.sqrt(np.random.uniform(0, 1, N))
    
    # Convert to ellipse coordinates
    x = a * r * np.cos(theta) + center[0]
    y = b * r * np.sin(theta) + center[1]

    points = [(x[i], y[i]) for i in range(N)]
    
    return points


def make_diffusion_circular(n_points=1000, layers = 4, inter_radius=0.8, outer_radius=2.0 ,init_density = 300, decay=0.55):
    interval = inter_radius/layers

    points1 = []
    dense = init_density
    for i in range(layers):
        inner_r = interval*i
        outer_r = interval*(i+1)
        points1 += sample_uniform_annulus([0,0], inner_r, outer_r, dense)
        dense *= decay
        dense = int(dense)
    cnt1 = len(points1)
        

    interval *= 0.31
    points2 = []
    dense = init_density
    for i in range(layers):
        outer_r = outer_radius - interval*i
        inner_r = outer_r - interval
        points2 += sample_uniform_annulus([0,0], inner_r, outer_r, dense)
        dense *= decay
        dense = int(dense)
    cnt2 = len(points2)

    points = points2 + points1
    y = [0]*cnt2 + [1]*cnt1

    
    # shuffle points
    indices = list(range(len(points)))
    random.shuffle(indices)
    Z = np.array([np.array([points[i][0],points[i][1]]) for i in indices])
    labels = [y[i] for i in indices]

    return ((Z, labels), {'damping': .75, 'preference': -220, 'n_clusters': 2})



def make_diffusion_exp(layers = 8, right=0.5 ,init_density = 300, decay=0.55):
    interval = right/layers


    points1 = []
    dense = init_density
    for i in range(layers):
        L = interval*i
        R = interval*(i+1)
        points1 += sample_uniform_rectangle(dense, L, R)
        dense *= decay
        dense = int(dense)
    cnt1 = len(points1)

        

    points2 = []
    dense = init_density
    for i in range(layers):
        R = 1.0 - interval*i
        L = R - interval
        points2 += sample_uniform_rectangle(dense, L, R)
        dense *= decay
        dense = int(dense)
    cnt2 = len(points2)


    points = points2 + points1

    
    Z = np.array([np.array([x,y]) for x, y in points])
    labels = [0]*cnt2 + [1]*cnt1

    return ((Z, labels), {'damping': .75, 'preference': -220, 'n_clusters': 2})



def make_moon_noised(n_samples=500, noise=0.05, num_noise_points=50, noise_bounds=None):
    # Generate the moons dataset
    X, y = make_moons(n_samples=n_samples, noise=noise)
    
    # Determine noise bounds
    if noise_bounds is None:
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
    else:
        (x_min, x_max), (y_min, y_max) = noise_bounds

    # Generate sparse noise points
    noise_x = np.random.uniform(x_min, x_max, num_noise_points)
    noise_y = np.random.uniform(y_min, y_max, num_noise_points)
    noise = np.column_stack((noise_x, noise_y))

    # Combine data and noise
    X_combined = np.vstack((X, noise))
    y_combined = np.hstack((y, [-1] * num_noise_points))  # Label noise as -1

    return ((X_combined, y_combined), {'damping': .75, 'preference': -220, 'n_clusters': 2})



def make_circle_noised(n_samples=500, noise=0.05, num_noise_points=50, noise_bounds=None):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    
    # Determine noise bounds
    if noise_bounds is None:
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
    else:
        (x_min, x_max), (y_min, y_max) = noise_bounds

    # Generate sparse noise points
    noise_x = np.random.uniform(x_min, x_max, num_noise_points)
    noise_y = np.random.uniform(y_min, y_max, num_noise_points)
    noise = np.column_stack((noise_x, noise_y))

    # Combine data and noise
    X_combined = np.vstack((X, noise))
    y_combined = np.hstack((y, [-1] * num_noise_points))  # Label noise as -1

    return ((X_combined, y_combined), {'damping': .77, 'preference': -240, 'quantile': .2, 'n_clusters': 2, 'min_samples': 20, 'xi': 0.25})


def make_blobs_size_dis(size_ratio=0.5, center_dis=1.0, var = 0.5, n_samples=1500):
    """
    Generate two blobs with controlled size ratio and center distance.
    
    Parameters:
    -----------
    size_ratio : float, default=0.5
        Ratio of the second blob's radius to the first blob's radius (0 < size_ratio <= 1)
    center_dis : float, default=1.0
        Distance between the centers of the two blobs
    n_samples : int, default=1500
        Number of samples in the larger blob
    """

    # Input validation
    if not 0 < size_ratio <= 1:
        raise ValueError("size_ratio must be between 0 and 1")
    if center_dis <= 0:
        raise ValueError("center_dis must be positive")
        
    # Calculate number of samples for smaller blob based on area ratio
    area_ratio = size_ratio ** 2
    n_samples_smaller = int(n_samples * area_ratio)
    
    # Generate centers
    centers = np.array([[0, 0], [center_dis, 0]])
    
    # Generate larger blob
    X1, y1 = make_blobs(n_samples=n_samples, 
                        centers=[centers[0]], 
                        cluster_std=var,
                        random_state=42)
    
    # Generate smaller blob
    X2, y2 = make_blobs(n_samples=n_samples_smaller, 
                        centers=[centers[1]], 
                        cluster_std=size_ratio*var,
                        random_state=42)
    
    # Combine the blobs
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2 + 1))  # Add 1 to second blob's labels to distinguish
    
    return ((X, y),{})



def final_make_multiple_circles(infos, seed):
    points = []
    y = []
    for i, info in enumerate(infos):
        points += sample_uniform_annulus(info['c'], 0, info['r'], info['n'], seed)
        y += [i] * info['n']

    indices = list(range(len(points)))
    random.shuffle(indices)
    Z = np.array([np.array([points[i][0],points[i][1]]) for i in indices])
    labels = [y[i] for i in indices]


    return ((Z, labels), {})


def final_make_concentric(info1, info2, seed):
    points = []
    points += sample_uniform_annulus([0,0], info1['inner'], info1['outer'], info1['n'], seed=seed)
    points += sample_uniform_annulus([0,0], info2['inner'], info2['outer'], info2['n'], seed=seed)
    y = [0] * info1['n'] + [1] * info2['n']


    indices = list(range(len(points)))
    random.shuffle(indices)
    Z = np.array([np.array([points[i][0],points[i][1]]) for i in indices])
    labels = [y[i] for i in indices]

    return ((Z, labels), {})


def final_make_ellipse(infos):
    points = []
    y = []
    for i, info in enumerate(infos):
        points += sample_ellipse(info['n'], info['a'], info['b'], center=(info['c'][0],info['c'][1]), random_state=None)
        y += [i] * info['n']

    indices = list(range(len(points)))
    random.shuffle(indices)
    Z = np.array([np.array([points[i][0],points[i][1]]) for i in indices])
    labels = [y[i] for i in indices]


    return ((Z, labels), {})


def final_make_pair_rectangle(info1, info2):
    points = []
    y = []
    points += sample_uniform_rectangle(info1['N'], info1['L'], info1['R'])
    points += sample_uniform_rectangle(info2['N'], info2['L'], info2['R'])
    y = [0] * info1['N'] + [1] * info2['N']

    indices = list(range(len(points)))
    random.shuffle(indices)
    Z = np.array([np.array([points[i][0],points[i][1]]) for i in indices])
    labels = [y[i] for i in indices]

    return ((Z, labels), {}) 



def plot_datasets(datasets, titles=None):
    """
    Plot the datasets in a grid layout.
    
    Parameters:
    -----------
    datasets : list of tuples
        Each tuple contains ((X, y), params) where X is the data array,
        y is the labels array, and params is the parameters dictionary
    titles : list of str, optional
        Titles for each dataset plot
    """
    # Number of datasets
    n_datasets = len(datasets)
    
    # # Create figure
    # fig = plt.figure(figsize=(12, 3))
    
    # Plot each dataset
    for i, ((X, y), params) in enumerate(datasets):
        # # Create subplot
        # plt.subplot(1, n_datasets, i + 1)
        
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s=3)
        
        # # Set title
        # if titles is not None and i < len(titles):
        #     plt.title(titles[i])
            
        # Set axes properties
        plt.xticks(())
        plt.yticks(())
        plt.axis('equal')

    
    plt.tight_layout()
    plt.show()


def calcu_acc(pred, y):
    distinct_keys = list(set(pred)) 

    temp = [None] * len(y)
    for i in range(len(y)):
        if pred[i] == distinct_keys[0]:
            temp[i] = 0
        else:
            temp[i] = 1
    acc1 = 0
    for i in range(len(pred)):
        acc1 += 1 if temp[i]==y[i] else 0


    for i in range(len(y)):
        if pred[i] == distinct_keys[0]:
            temp[i] = 1
        else:
            temp[i] = 0

    acc2 = 0
    for i in range(len(pred)):
        acc2 += 1 if temp[i]==y[i] else 0

    return max(acc1,acc2)/len(y)


def calcu_precision(yhat, true_labels):
    """
    Calculate clustering precision by comparing with ground truth
    yhat: list of predicted cluster assignments
    true_labels: list of true cluster assignments
    """
    correct = 0
    total = 0
    
    # Compare each pair of points
    for i in range(len(yhat)):
        for j in range(i+1, len(yhat)):
            # Check if the algorithm agrees with ground truth
            # about whether these points should be in same cluster
            pred_same = (yhat[i] == yhat[j])
            true_same = (true_labels[i] == true_labels[j])
            
            if pred_same == true_same:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0



def final_massive_circles(
    N: int,                   # Number of circles
    space_bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
    r0: float,               # Mean radius
    sigma_r: float,          # Std dev of radius
    n0: int,                 # Mean number of points per circle
    sigma_n: float, 
    min_bound: float,        # Minimum distance between circle boundaries
    max_attempts: int = 1000 # Maximum attempts to place a circle
) -> Tuple[Tuple[np.ndarray, List[int]], dict]:
    """
    Generate N non-intersecting circles and sample points from within them.
    Maintains fixed point density and minimum boundary distance between circles.
    
    Returns:
        ((points, labels), {}): Tuple containing:
            - points: Array of all points, shape (total_points, 2)
            - labels: List of circle indices for each point
            - Empty dict for consistency with expected return format
    """
    x_min, x_max, y_min, y_max = space_bounds
    
    max_global_attempts = 100  # Maximum attempts to find valid configuration
    for global_attempt in range(max_global_attempts):
        centers = []
        radii = []
        
        # Generate circles
        success = True
        for _ in range(N):
            attempts = 0
            while attempts < max_attempts:
                # Sample radius from normal distribution
                radius = max(0.1, np.random.normal(r0, sigma_r))  # Ensure positive radius
                
                # Sample center coordinates
                x = np.random.uniform(x_min + radius, x_max - radius)
                y = np.random.uniform(y_min + radius, y_max - radius)
                
                # Check intersection with existing circles
                valid = True
                for existing_center, existing_radius in zip(centers, radii):
                    distance = np.sqrt(np.sum((existing_center - np.array([x, y])) ** 2))
                    if distance < (radius + existing_radius + min_bound):
                        valid = False
                        break
                
                if valid:
                    centers.append(np.array([x, y]))
                    radii.append(radius)
                    break
                    
                attempts += 1
            
            if attempts == max_attempts:
                success = False
                break
        
        if success:
            centers = np.array(centers)
            radii = np.array(radii)
            
            # Generate points within each circle
            points = []
            labels = []
            i = 0
            for center, radius in zip(centers, radii):
                # Sample number of points for this circle (fixed density)
                n = max(1, int(np.random.normal(n0, sigma_n)))
                
                # Generate random points in polar coordinates
                r = radius * np.sqrt(np.random.uniform(0, 1, n))
                theta = np.random.uniform(0, 2*np.pi, n)
                
                # Convert to Cartesian coordinates
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                
                points.append(np.column_stack((x, y)))
                labels += n*[i]
                i += 1
            
            Z = np.concatenate(points, axis=0)
            return ((Z, labels), {})
    
    raise ValueError(f"Could not find valid configuration after {max_global_attempts} global attempts")



def final_massive_circles_fixed_density(
    N: int,                   # Number of circles
    space_bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
    r0: float,               # Mean radius
    sigma_r: float,          # Std dev of radius
    n0: int,                 # Mean number of points per circle
    sigma_n: float, 
    min_bound: float,        # Minimum distance between circle boundaries
    max_attempts: int = 1000 # Maximum attempts to place a circle
) -> Tuple[Tuple[np.ndarray, List[int]], dict]:
    """
    Generate N non-intersecting circles and sample points from within them.
    Maintains fixed point density and minimum boundary distance between circles.
    
    Returns:
        ((points, labels), {}): Tuple containing:
            - points: Array of all points, shape (total_points, 2)
            - labels: List of circle indices for each point
            - Empty dict for consistency with expected return format
    """
    x_min, x_max, y_min, y_max = space_bounds
    
    max_global_attempts = 100  # Maximum attempts to find valid configuration
    for global_attempt in range(max_global_attempts):
        centers = []
        radii = []
        
        # Generate circles
        success = True
        for _ in range(N):
            attempts = 0
            while attempts < max_attempts:
                # Sample radius from normal distribution
                radius = max(0.1, np.random.normal(r0, sigma_r))  # Ensure positive radius
                
                # Sample center coordinates
                x = np.random.uniform(x_min + radius, x_max - radius)
                y = np.random.uniform(y_min + radius, y_max - radius)
                
                # Check intersection with existing circles
                valid = True
                for existing_center, existing_radius in zip(centers, radii):
                    distance = np.sqrt(np.sum((existing_center - np.array([x, y])) ** 2))
                    if distance < (radius + existing_radius + min_bound):
                        valid = False
                        break
                
                if valid:
                    centers.append(np.array([x, y]))
                    radii.append(radius)
                    break
                    
                attempts += 1
            
            if attempts == max_attempts:
                success = False
                break
        
        if success:
            centers = np.array(centers)
            radii = np.array(radii)
            
            # Generate points within each circle
            points = []
            labels = []
            i = 0
            for center, radius in zip(centers, radii):
                # Sample number of points for this circle (fixed density)
                n = max(1, int(n0*(radius**2/r0**2)))
                
                # Generate random points in polar coordinates
                r = radius * np.sqrt(np.random.uniform(0, 1, n))
                theta = np.random.uniform(0, 2*np.pi, n)
                
                # Convert to Cartesian coordinates
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                
                points.append(np.column_stack((x, y)))
                labels += n*[i]
                i += 1
            
            Z = np.concatenate(points, axis=0)
            return ((Z, labels), {})
    
    raise ValueError(f"Could not find valid configuration after {max_global_attempts} global attempts")



def calculate_balanced_precision(yhat, true_labels):
    # Separate metrics for same-cluster and different-cluster pairs
    same_cluster_correct = 0
    same_cluster_total = 0
    diff_cluster_correct = 0
    diff_cluster_total = 0
    
    for i in range(len(yhat)):
        for j in range(i+1, len(yhat)):
            pred_same = (yhat[i] == yhat[j])
            true_same = (true_labels[i] == true_labels[j])
            
            if true_same:
                same_cluster_total += 1
                if pred_same:
                    same_cluster_correct += 1
            else:
                diff_cluster_total += 1
                if not pred_same:
                    diff_cluster_correct += 1
    
    # Calculate balanced precision
    same_precision = same_cluster_correct / same_cluster_total
    diff_precision = diff_cluster_correct / diff_cluster_total
    
    return (same_precision + diff_precision) / 2



def calcu_precision(data, yhat, true_labels):
    """
    Calculate clustering precision by comparing with ground truth
    yhat: list of predicted cluster assignments
    true_labels: list of true cluster assignments
    """
    correct = 0
    total = 0
    
    # Compare each pair of points
    for i in range(len(yhat)):
        for j in range(i+1, len(yhat)):
            dist = np.linalg.norm(data[i] - data[j])
            weight = np.exp(-dist)
            # weight = 1.0/dist

            # Check if the algorithm agrees with ground truth
            # about whether these points should be in same cluster
            pred_same = (yhat[i] == yhat[j])
            true_same = (true_labels[i] == true_labels[j])
            
            if pred_same == true_same:
                correct += 1 * weight
            total += 1 * weight
    
    return correct / total if total > 0 else 0




def sample_half_annulus(isUp, center, r_inner, r_outer, n_points, seed):
    """
    Sample points uniformly from an annulus with specified density.
    
    Parameters:
    -----------
    r_inner : float
        Inner radius of annulus
    r_outer : float
        Outer radius of annulus
    density : float
        Points per unit area
    """

    if seed is not None:
        random.seed(seed)
        
    # Calculate annulus area
    area = math.pi * (r_outer**2 - r_inner**2)
        
    points = []
    for _ in range(n_points):
        # Generate random radius with sqrt for uniform area distribution
        r = math.sqrt(random.uniform(r_inner**2, r_outer**2))
        
        # Generate random angle
        if isUp:
            theta = random.uniform(0, math.pi)
        else:
            theta = random.uniform(math.pi, 2*math.pi)
        
        # Convert to Cartesian coordinates
        x = center[0] + r * math.cos(theta)
        y = center[1] + r * math.sin(theta)
        points.append((x, y))
    
    return points



def final_make_interleaved_anuulus(ratio, w, b, n1, n2, seed):

    points = []

    points += sample_half_annulus(isUp=True, center=[0,0], r_inner=1.0-w, r_outer=1.0, n_points=n1, seed=seed)

    points += sample_half_annulus(isUp=False, center=[1.0, 1.0-b], r_inner=1.0-ratio*w/2, r_outer=1.0+ratio*w/2, n_points=n2, seed=seed)

    y = [0] * n1 + [1] * n2

    indices = list(range(len(points)))
    random.shuffle(indices)
    Z = np.array([np.array([points[i][0],points[i][1]]) for i in indices])
    labels = [y[i] for i in indices]

    return ((Z, labels), {}) 
      



import numpy as np
import matplotlib.pyplot as plt

def render_points(points: np.ndarray, colors: np.ndarray, centroid: np.ndarray=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, marker='.')
    
    if centroid is not None:
        ax.scatter(centroid[:,0], centroid[:,1], centroid[:,2], s=32.0, c='purple', marker='o', alpha=1.0)
        
    
    plt.show()

def main():
    N = 384
    
    colors = np.array([
        [241, 142, 0],
        [0, 68, 137],
        [19, 128, 0],
    ]) / 255.0
    
    centroids = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0]
    ])
    residuals = np.random.rand(N, 3)
    residuals -= 0.5
    classes = np.random.randint(low=0, high=len(centroids), size=len(residuals))
    colors = colors[classes]
    
    points = residuals + centroids[classes]
    
    render_points(points, colors, centroids)   
    render_points(residuals, colors)   
    

if __name__ == '__main__':
    main()
    
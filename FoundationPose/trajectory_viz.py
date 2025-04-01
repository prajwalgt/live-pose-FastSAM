import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_trajectory_data(filename):
    """
    Read trajectory data from the specified file.
    Returns arrays of coordinates for both points.
    """
    pt1_x, pt1_y, pt1_z = [], [], []
    pt2_x, pt2_y, pt2_z = [], [], []
    obj_x, obj_y, obj_z = [], [], []  # Object center coordinates in robot frame
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Process object center in robot frame
        if line.startswith('# object center in robot frame'):
            if i+3 < len(lines):
                obj_x.append(float(lines[i+1].strip()))
                obj_y.append(float(lines[i+2].strip()))
                obj_z.append(float(lines[i+3].strip()))
        
        # Process pt_transformed1
        elif line.startswith('# pt_transformed1'):
            if i+3 < len(lines):
                pt1_x.append(float(lines[i+1].strip()))
                pt1_y.append(float(lines[i+2].strip()))
                pt1_z.append(float(lines[i+3].strip()))
        
        # Process pt_transformed2
        elif line.startswith('# pt_transformed2'):
            if i+3 < len(lines):
                pt2_x.append(float(lines[i+1].strip()))
                pt2_y.append(float(lines[i+2].strip()))
                pt2_z.append(float(lines[i+3].strip()))
        
        i += 1
    
    return (np.array(pt1_x), np.array(pt1_y), np.array(pt1_z)), \
           (np.array(pt2_x), np.array(pt2_y), np.array(pt2_z)), \
           (np.array(obj_x), np.array(obj_y), np.array(obj_z))

def plot_3d_trajectory(pt1_coords, pt2_coords, obj_coords=None, title='3D Trajectory'):
    """
    Create a 3D plot of both point trajectories.
    
    Parameters:
    -----------
    pt1_coords: tuple of (x, y, z) arrays for point 1
    pt2_coords: tuple of (x, y, z) arrays for point 2
    obj_coords: optional tuple of (x, y, z) arrays for object center
    title: plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    pt1_x, pt1_y, pt1_z = pt1_coords
    pt2_x, pt2_y, pt2_z = pt2_coords
    
    # Plot trajectories
    ax.plot(pt1_x, pt1_y, pt1_z, 'r-', linewidth=2, label='Point 1')
    ax.plot(pt2_x, pt2_y, pt2_z, 'b-', linewidth=2, label='Point 2')
    
    # Plot object center trajectory if provided
    if obj_coords is not None and len(obj_coords[0]) > 0:
        obj_x, obj_y, obj_z = obj_coords
        ax.plot(obj_x, obj_y, obj_z, 'g-', linewidth=2, label='Object Center')
    
    # Mark start and end points
    ax.scatter(pt1_x[0], pt1_y[0], pt1_z[0], c='r', marker='o', s=100, label='Start Point 1')
    ax.scatter(pt1_x[-1], pt1_y[-1], pt1_z[-1], c='r', marker='x', s=100, label='End Point 1')
    ax.scatter(pt2_x[0], pt2_y[0], pt2_z[0], c='b', marker='o', s=100, label='Start Point 2')
    ax.scatter(pt2_x[-1], pt2_y[-1], pt2_z[-1], c='b', marker='x', s=100, label='End Point 2')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add a grid
    ax.grid(True)
    
    # Add a legend
    ax.legend()
    
    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Equal aspect ratio for all axes
    max_range = max([
        max(pt1_x.max(), pt2_x.max()) - min(pt1_x.min(), pt2_x.min()),
        max(pt1_y.max(), pt2_y.max()) - min(pt1_y.min(), pt2_y.min()),
        max(pt1_z.max(), pt2_z.max()) - min(pt1_z.min(), pt2_z.min())
    ])
    
    mid_x = (max(pt1_x.max(), pt2_x.max()) + min(pt1_x.min(), pt2_x.min())) * 0.5
    mid_y = (max(pt1_y.max(), pt2_y.max()) + min(pt1_y.min(), pt2_y.min())) * 0.5
    mid_z = (max(pt1_z.max(), pt2_z.max()) + min(pt1_z.min(), pt2_z.min())) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)
    
    return fig, ax

def plot_trajectory_projections(pt1_coords, pt2_coords, obj_coords=None):
    """
    Create 2D projection plots of the trajectories onto the XY, XZ, and YZ planes.
    
    Parameters:
    -----------
    pt1_coords: tuple of (x, y, z) arrays for point 1
    pt2_coords: tuple of (x, y, z) arrays for point 2
    obj_coords: optional tuple of (x, y, z) arrays for object center
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract coordinates
    pt1_x, pt1_y, pt1_z = pt1_coords
    pt2_x, pt2_y, pt2_z = pt2_coords
    
    # XY Projection (Top View)
    ax_xy = axes[0]
    ax_xy.plot(pt1_x, pt1_y, 'r-', linewidth=2, label='Point 1')
    ax_xy.plot(pt2_x, pt2_y, 'b-', linewidth=2, label='Point 2')
    if obj_coords is not None and len(obj_coords[0]) > 0:
        ax_xy.plot(obj_coords[0], obj_coords[1], 'g-', linewidth=2, label='Object Center')
    ax_xy.set_title('XY Projection (Top View)')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.grid(True)
    ax_xy.axis('equal')
    
    # XZ Projection (Front View)
    ax_xz = axes[1]
    ax_xz.plot(pt1_x, pt1_z, 'r-', linewidth=2, label='Point 1')
    ax_xz.plot(pt2_x, pt2_z, 'b-', linewidth=2, label='Point 2')
    if obj_coords is not None and len(obj_coords[0]) > 0:
        ax_xz.plot(obj_coords[0], obj_coords[2], 'g-', linewidth=2, label='Object Center')
    ax_xz.set_title('XZ Projection (Front View)')
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.grid(True)
    ax_xz.axis('equal')
    
    # YZ Projection (Side View)
    ax_yz = axes[2]
    ax_yz.plot(pt1_y, pt1_z, 'r-', linewidth=2, label='Point 1')
    ax_yz.plot(pt2_y, pt2_z, 'b-', linewidth=2, label='Point 2')
    if obj_coords is not None and len(obj_coords[0]) > 0:
        ax_yz.plot(obj_coords[1], obj_coords[2], 'g-', linewidth=2, label='Object Center')
    ax_yz.set_title('YZ Projection (Side View)')
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.grid(True)
    ax_yz.axis('equal')
    
    # Add legend to the first subplot
    axes[0].legend()
    
    plt.tight_layout()
    return fig, axes

def calculate_statistics(pt1_coords, pt2_coords):
    """
    Calculate and print statistics about the trajectories.
    
    Parameters:
    -----------
    pt1_coords: tuple of (x, y, z) arrays for point 1
    pt2_coords: tuple of (x, y, z) arrays for point 2
    """
    pt1_x, pt1_y, pt1_z = pt1_coords
    pt2_x, pt2_y, pt2_z = pt2_coords
    
    # Calculate total distance traveled
    def calculate_distance(coords):
        x, y, z = coords
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    
    pt1_distance = calculate_distance(pt1_coords)
    pt2_distance = calculate_distance(pt2_coords)
    
    # Calculate distance between the two points
    dist_between = np.sqrt(
        (pt1_x - pt2_x)**2 + 
        (pt1_y - pt2_y)**2 + 
        (pt1_z - pt2_z)**2
    )
    
    # Print statistics
    print("=== Trajectory Statistics ===")
    print(f"Number of frames: {len(pt1_x)}")
    
    print("\nPoint 1:")
    print(f"  Start position: ({pt1_x[0]:.6f}, {pt1_y[0]:.6f}, {pt1_z[0]:.6f})")
    print(f"  End position: ({pt1_x[-1]:.6f}, {pt1_y[-1]:.6f}, {pt1_z[-1]:.6f})")
    print(f"  Total distance traveled: {pt1_distance:.6f}")
    print(f"  X range: {pt1_x.min():.6f} to {pt1_x.max():.6f}")
    print(f"  Y range: {pt1_y.min():.6f} to {pt1_y.max():.6f}")
    print(f"  Z range: {pt1_z.min():.6f} to {pt1_z.max():.6f}")
    
    print("\nPoint 2:")
    print(f"  Start position: ({pt2_x[0]:.6f}, {pt2_y[0]:.6f}, {pt2_z[0]:.6f})")
    print(f"  End position: ({pt2_x[-1]:.6f}, {pt2_y[-1]:.6f}, {pt2_z[-1]:.6f})")
    print(f"  Total distance traveled: {pt2_distance:.6f}")
    print(f"  X range: {pt2_x.min():.6f} to {pt2_x.max():.6f}")
    print(f"  Y range: {pt2_y.min():.6f} to {pt2_y.max():.6f}")
    print(f"  Z range: {pt2_z.min():.6f} to {pt2_z.max():.6f}")
    
    print("\nDistance between points:")
    print(f"  Initial: {dist_between[0]:.6f}")
    print(f"  Final: {dist_between[-1]:.6f}")
    print(f"  Average: {np.mean(dist_between):.6f}")
    print(f"  Min: {np.min(dist_between):.6f}")
    print(f"  Max: {np.max(dist_between):.6f}")

def main():
    # File path
    file_path = 'transformed_points.txt'  # Update this with your file path
    
    # Read trajectory data
    print("Reading trajectory data...")
    pt1_coords, pt2_coords, obj_coords = read_trajectory_data(file_path)
    
    # Calculate and print statistics
    calculate_statistics(pt1_coords, pt2_coords)
    
    # Create and show plots
    print("\nCreating 3D trajectory plot...")
    fig_3d, _ = plot_3d_trajectory(pt1_coords, pt2_coords, obj_coords)
    
    print("Creating 2D projection plots...")
    fig_proj, _ = plot_trajectory_projections(pt1_coords, pt2_coords, obj_coords)
    
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
import numpy as np
# from sapien import Pose
def xyzw_to_pcd(in_points):
    # Check if input has at least 4 columns (xyzw)
    if type(in_points) == dict:
        points = in_points["xyzw"]
        rgb = in_points["rgb"] if "rgb" in in_points else None
    else:
        points = in_points[:,:4]
        rgb = in_points[:,4:] if in_points.shape[-1] > 4 else None

    if points.shape[1] < 4:
        raise ValueError("Input points should have at least 4 columns (xyzw)")
    
    # Extract w component and use it as a validity mask
    w = points[:, 3]
    valid_mask = w > 0.5  # Consider points with w > 0.5 as valid
    
    # Filter out invalid points
    valid_points = points[valid_mask]
    
    # If RGB values are present (7 columns), preserve them
    # if points.shape[1] >= 7:
    if rgb is not None:
        # Return xyz+rgb without the w column
        return np.hstack([valid_points[:, :3], rgb[valid_mask]])
    
    # If only xyzw provided, return xyz
    return valid_points[:, :3]

def is_xyzw(points):
    if type(points)==dict:
        return "xyzw" in points
    return points.shape[-1] == 4 or points.shape[-1]==7

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion and build 3×3 회전행렬."""
    x, y, z, w = q

    n = x*x + y*y + z*z + w*w
    if abs(n - 1.0) > 1e-8:
        x, y, z, w = q / np.sqrt(n)

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

def viz_world_pcd_img(
    points: np.ndarray,
    obs_frame_pose: np.ndarray,
    width=256, height=256, colored=True,
    camera_position=None, camera_focal_point=None,
    camera_up=None, camera_zoom=1.2, point_size=3
):
    # obs_frame_pose: [px,py,pz, qx,qy,qz,qw] 형태
    if obs_frame_pose.ndim > 1:
        obs_frame_pose = obs_frame_pose[0]
    position = obs_frame_pose[:3]
    quat     = obs_frame_pose[3:]
    
    world_points = points.copy()
    if is_xyzw(points):
        world_points = xyzw_to_pcd(world_points)
    
    R = quaternion_to_rotation_matrix(quat)          # (3,3)
    pts = world_points[:, :3]                        # (N,3)
    pts_world = pts.dot(R.T) + position[None, :]     # (N,3)
    
    if world_points.shape[1] > 3:
        world_points = np.concatenate([pts_world, world_points[:, 3:]], axis=-1)
    else:
        world_points = pts_world
    
    return viz_pcd_img(
        world_points,
        width=width, height=height, colored=colored,
        camera_position=camera_position,
        camera_focal_point=camera_focal_point,
        camera_up=camera_up,
        camera_zoom=camera_zoom,
        point_size=point_size
    )

def viz_pcd_img(points, width=256, height=256, colored=True, 
                camera_position=None, camera_focal_point=None, 
                camera_up=None, camera_zoom=1.2, point_size=3):
    """
    Visualize point cloud data as an RGB image.
    
    Args:
        points: Point cloud data in the format of n*3 (xyz) or n*6 (xyz+rgb)
        width: Width of output image
        height: Height of output image
        colored: Whether to use RGB colors from the point cloud data
        camera_position: Custom camera position (xyz). If None, auto-calculated
        camera_focal_point: Custom camera focal point (xyz). If None, uses the center of points
        camera_up: Camera up direction. Default is [0, 1, 0]
        camera_zoom: Camera zoom factor. Default is 1.2
        point_size: Size of points in the visualization. Default is 3
    
    Returns:
        np.ndarray: RGB image (height, width, 3)
    """
    if is_xyzw(points):
        points = xyzw_to_pcd(points)

    try:
        import pyvista as pv
        import os
        import numpy as np
        from contextlib import contextmanager
        
        @contextmanager
        def temp_opengl_context():
            """Temporary OpenGL context manager for PyVista rendering"""
            old_pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", None)
            
            try:
                os.environ["PYOPENGL_PLATFORM"] = "osmesa"
                yield
            finally:
                if old_pyopengl_platform is None:
                    if "PYOPENGL_PLATFORM" in os.environ:
                        del os.environ["PYOPENGL_PLATFORM"]
                else:
                    os.environ["PYOPENGL_PLATFORM"] = old_pyopengl_platform
        
        with temp_opengl_context():
            # PyVista setup
            pv.set_plot_theme("document")
            pv.global_theme.transparent_background = True
            pv.OFF_SCREEN = True
            
            # Extract point information based on format
            if points.shape[1] >= 6 and colored:
                xyz = points[:, :3]
                rgb = points[:, 3:6]  # RGB values (typically in 0-1 range)
                
                # Convert RGB values to 0-255 range if they're in 0-1
                # if np.max(rgb) <= 1.0:
                #     rgb = rgb * 255
                rgb = rgb.astype(np.uint8)
            else:
                xyz = points[:, :3]
                rgb = None
            
            # Create plotter
            plotter = pv.Plotter(off_screen=True, window_size=[width, height])
            
            # Create point cloud
            point_cloud = pv.PolyData(xyz)
            
            # Add colors if available
            if rgb is not None:
                point_cloud["colors"] = rgb
                plotter.add_points(
                    point_cloud, 
                    render_points_as_spheres=False, 
                    point_size=point_size, 
                    rgb=True, 
                    scalars="colors",
                    reset_camera=False,
                )
            else:
                plotter.add_points(
                    point_cloud, 
                    render_points_as_spheres=False, 
                    point_size=point_size, 
                    color=[0.7, 0.7, 0.7],
                    reset_camera=False,
                )
            
            # Calculate center of point cloud for camera setup
            center = np.mean(xyz, axis=0)
            
            # Set up camera
            if camera_focal_point is None:
                camera_focal_point = center
                
            if camera_position is None:
                # Default camera position: 2 units above center on Z axis
                camera_position = center + np.array([0, 0, 2])
                
            if camera_up is None:
                camera_up = [0, 1, 0]
                
            # Apply camera settings
            plotter.camera.position = camera_position
            plotter.camera.focal_point = camera_focal_point
            plotter.camera.up = camera_up
            # plotter.reset_camera()
            plotter.camera.zoom(camera_zoom)
            
            # Render and capture image
            img = plotter.screenshot(return_img=True)
            
            # Convert from RGBA to RGB (remove alpha channel)
            if img.shape[-1] == 4:  # If RGBA format
                img = img[:, :, :3]  # Keep only RGB channels
            
            # Close plotter to free resources
            plotter.close()
            
            return img.astype(np.uint8)
            
    except Exception as e:
        print(f"Error visualizing pointcloud: {e}")
        # Return empty image in case of error
        return np.zeros((height, width, 3), dtype=np.uint8)
    
def save_interactive_pcd(points, filename="interactive_pointcloud.html", colored=True, width=800, height=600):
    """
    Save point cloud data as an interactive HTML visualization.
    
    Args:
        points: Point cloud data (n*3 or n*6 format)
        filename: Output HTML filename
        colored: Whether to use RGB colors if available
        width: Viewport width
        height: Viewport height
        
    Returns:
        str: Path to saved HTML file
    """
    try:
        import pyvista as pv
        import numpy as np
        import os
        
        # Extract point information based on format
        if points.shape[1] >= 6 and colored:
            xyz = points[:, :3]
            rgb = points[:, 3:6]
            
            # Convert RGB values to 0-255 range if they're in 0-1
            if np.max(rgb) <= 1.0:
                rgb = rgb * 255
        else:
            xyz = points[:, :3]
            rgb = None
        
        # Create plotter with desired size
        pl = pv.Plotter(window_size=[width, height], notebook=False)
        
        # Create point cloud
        point_cloud = pv.PolyData(xyz)
        
        # Add colors if available
        if rgb is not None:
            point_cloud["colors"] = rgb
            pl.add_points(
                point_cloud, 
                render_points_as_spheres=False, 
                point_size=5, 
                rgb=True, 
                scalars="colors"
            )
        else:
            pl.add_points(
                point_cloud, 
                render_points_as_spheres=False, 
                point_size=5, 
                color=[0.7, 0.7, 0.7]
            )
        
        # Add camera position widget
        pl.add_camera_orientation_widget()
        
        # Add instructions text
        pl.add_text(
            "Left-click and drag to rotate\n"
            "Right-click and drag to zoom\n"
            "Middle-click and drag to pan\n"
            "Press 'c' to save camera position",
            position='upper_left',
            font_size=12,
            color='white',
            shadow=True
        )
        
        # Add a callback to extract camera parameters
        pl.add_key_event('c', lambda: save_camera_position(pl))
        
        # Try to use export_html, handling different PyVista versions
        try:
            # Check if the backend parameter is supported
            import inspect
            if 'backend' in inspect.signature(pl.export_html).parameters:
                pl.export_html(filename, backend='static')
            else:
                # Older version of PyVista
                pl.export_html(filename)
        except Exception as e:
            print(f"Warning: HTML export failed: {e}")
            print("Using interactive display only")
        
        print(f"Interactive point cloud visualization ready")
        print("Adjust the camera view and press 'c' to save the camera position to 'camera_settings.py'")
        
        # Create an interactive window
        pl.show()
        
        return os.path.abspath(filename)
        
    except Exception as e:
        print(f"Error creating interactive pointcloud: {e}")
        return None
    
def save_camera_position(plotter):
    """
    Extract and save current camera settings to a Python file.
    This function is called when the user presses 'c' in the interactive view.
    
    Args:
        plotter: The PyVista plotter instance
    """
    try:
        # Get camera position
        pos = plotter.camera.position
        focal = plotter.camera.focal_point
        up = plotter.camera.up
        
        # Create camera settings Python code
        camera_code = f"""
# Camera settings saved from interactive view
# Copy these values to your visualization code

camera_settings = {{
    'camera_position': {pos},
    'camera_focal_point': {focal},
    'camera_up': {up},
}}

# Example usage:
# img = viz_pcd_img(
#     points, 
#     camera_position=camera_settings['camera_position'],
#     camera_focal_point=camera_settings['camera_focal_point'],
#     camera_up=camera_settings['camera_up'],
#     camera_zoom=camera_settings['camera_zoom']
# )
"""
        
        # Save to file
        with open('camera_settings.py', 'w') as f:
            f.write(camera_code)
        
        print("\nCamera settings saved to 'camera_settings.py'")
        print("You can import these settings in your main code.")
        
    except Exception as e:
        print(f"Error saving camera position: {e}")

# Example usage function
def extract_camera_settings(points, colored=True):
    """
    Open an interactive visualization to extract optimal camera settings.
    
    Args:
        points: Point cloud data (n*3 or n*6 format)
        colored: Whether to use RGB colors if available
        
    Returns:
        str: Path to saved camera settings file
    """
    # Save interactive HTML and open it
    html_path = save_interactive_pcd(points, colored=colored)
    
    if html_path:
        print("\nInstructions:")
        print("1. Adjust the camera angle using mouse controls")
        print("2. When you find a good view, press 'c' to save camera settings")
        print("3. Camera settings will be saved to 'camera_settings.py'")
        print("4. Import these settings in your main code")
        
        return 'camera_settings.py'
    
    return None

def apply_pose_to_points(x, pose):
  return to_normal(to_generalized(x) @ pose.to_transformation_matrix().T)

def to_generalized(x):
  if x.shape[-1] == 4:
      return x
  assert x.shape[-1] == 3
  output_shape = list(x.shape)
  output_shape[-1] = 4
  ret = np.ones(output_shape).astype(np.float32)
  ret[..., :3] = x
  return ret

def to_normal(x):
  if x.shape[-1] == 3:
      return x
  assert x.shape[-1] == 4
  return x[..., :3] / x[..., 3:]


if __name__ == "__main__":
    test_pcd_path = "pcd_test.pkl"
    import pickle
    with open(test_pcd_path, "rb") as f:
        test_pcd = pickle.load(f)

    extract_camera_settings(test_pcd)
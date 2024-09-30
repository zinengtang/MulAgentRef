import numpy as np

from matplotlib import pyplot as plt

from PIL import Image

# import habitat

import quaternion

import cv2

import trimesh
import magnum as mn 
from matplotlib.patches import Wedge
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

def plot_relative_locations_with_fov(agent_locs, centroid_locs, object_locs, yaw_angle, fov_degrees):
    # Plotting setup
    plt.figure(figsize=(4, 4))
    print(yaw_angle)
    # Plot agent and objects
    plt.plot(agent_locs[0], agent_locs[2], 'bo', markersize=10, label='Agent')
    plt.plot(object_locs[:, 0], object_locs[:, 2], 'ro', markersize=5, label='Objects')
    plt.plot(centroid_locs[0], centroid_locs[2], 'go', markersize=5, label='Centroids')
    
    # Calculate the FOV coverage
    yaw_angle += 1.5 * np.pi
    fov_half_angle = fov_degrees / 2
    left_fov_angle = np.degrees(yaw_angle) - fov_half_angle
    right_fov_angle = np.degrees(yaw_angle) + fov_half_angle
    
    # Draw the FOV area
    fov_radius = 10  # Arbitrary, just to show the FOV extent visually
    wedge = Wedge((agent_locs[0], agent_locs[2]), fov_radius, left_fov_angle, right_fov_angle, color='yellow', alpha=0.2, label='FOV')
    plt.gca().add_patch(wedge)
    
    # Final plot adjustments
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title('Top-Down Map of Agent and Objects with FOV')
    plt.show()

                       
# def construct_habitat_config(path="configs/social_nav.yaml"):
#     config = habitat.get_config(
#         config_path=path,
#         overrides=[
#             "habitat.environment.max_episode_steps=100000",
#             "habitat.environment.iterator_options.shuffle=False",
#             "habitat.dataset.split=val"
#         ],
#     )
#     return config

def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])
):  # noqa: B006
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

def simulate(sim, dt=1.0, get_frames=False):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    # print("Simulating " + str(dt) + " world seconds.")
    # observations = []
    # start_time = sim.get_world_time()
    sim.step_physics(1.0)
    # while sim.get_world_time() < start_time + dt:
    #     # obj.apply_force(sim.get_gravity() * obj.mass, [0.0, 0.0, 0.0])
    #     sim.step_physics(1.0 / 60.0)
    #     if get_frames:
    #         observations.append(sim.get_sensor_observations())

    # return observations


def calculate_euler_angles(target_location, agent_location):
    # Calculate the direction vector
    dx, dy, dz = np.subtract(target_location, agent_location)
    dy = 0
    # Calculate yaw and pitch
    yaw = np.arctan2(dx, dz)
    # pitch = np.arctan2(dy, np.sqrt(dx**2 + dz**2))
    return np.array([0, yaw, 0])

def sample_uniform_points_in_box(center, dimensions, n=16):
    """
    Uniformly sample n points within a box defined by its center and dimensions.

    :param center: The center of the box [x, y, z].
    :param dimensions: The dimensions of the box [width, height, depth].
    :param n: The number of points to sample.
    :return: An array of shape (n, 3) containing the sampled points.
    """
    # Calculate half dimensions to determine sampling ranges
    half_dims = dimensions / 2.0

    # Determine the ranges for x, y, and z
    x_range = [center[0] - half_dims[0], center[0] + half_dims[0]]
    y_range = [center[1] - 1 * half_dims[1], center[1] + 1 * half_dims[1]]
    z_range = [center[2] - half_dims[2], center[2] + half_dims[2]]

    # Uniformly sample n points within the ranges
    x_samples = np.random.uniform(x_range[0], x_range[1], n)
    y_samples = np.random.uniform(y_range[0], y_range[1], n)
    z_samples = np.random.uniform(z_range[0], z_range[1], n)

    # Combine the samples into an (n, 3) array
    sampled_points = np.vstack((x_samples, y_samples, z_samples)).T

    return sampled_points

def world_to_camera_coordinates(world_position, camera_position, camera_orientation, hfov, width, height):
    # Calculate the intrinsic matrix K
    def compute_vfov(hfov, width, height):
        aspect_ratio = height / width
        hfov_rad = np.radians(hfov)
        vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * aspect_ratio) * 1.05
        return vfov_rad

    vfov_rad = compute_vfov(hfov, width, height)
    fx = (width / 2) / np.tan(np.radians(hfov) / 2)
    fy = (height / 2) / np.tan(vfov_rad / 2)

    K = np.array([
        [fx, 0, width / 2],
        [0, fy, height / 2],
        [0, 0, 1]
    ])

    # Convert camera orientation from quaternion to rotation matrix
    rotation_matrix = quaternion.as_rotation_matrix(camera_orientation)
    # Calculate the extrinsic matrix
    T_world_camera = np.eye(4)
    T_world_camera[0:3, 0:3] = rotation_matrix
    T_world_camera[0:3, 3] = camera_position
    # Invert to get camera --> world transformation
    T_camera_world = np.linalg.inv(T_world_camera)
    
    # Transform the world coordinate into the camera coordinate system
    world_position_homogeneous = np.append(world_position, 1) # Convert to homogeneous coordinate
    camera_coordinate = np.dot(T_camera_world, world_position_homogeneous)
    # Project onto the image plane
    image_coordinate_homogeneous = np.dot(K, camera_coordinate[:3])
    # Normalize to get pixel coordinates
    pixel_coordinate = image_coordinate_homogeneous[:2] / image_coordinate_homogeneous[2]
    return pixel_coordinate


def draw_bounding_box(image, corners_image_plane, width, draw=True):
    min_x, min_y = np.min(corners_image_plane, axis=0)
    max_x, max_y = np.max(corners_image_plane, axis=0)
    # print( (int(min_x), int(min_y)), (int(max_x), int(max_y)))
    bbox = [width-int(max_x), int(min_y), width-int(min_x), int(max_y)]
    # Draw rectangle on the image
    # You might need to convert these coordinates to integer values
    if draw:
        cv2.rectangle(image, (width-int(max_x), int(min_y)), (width-int(min_x), int(max_y)), (100, 100, 255), 2)
    return bbox, image

def select_poses(poses, minimum_distance_between_agents=1.0):
    # Function to calculate distance between two poses in the x-z plane
    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5
    selected_poses = []
    for pose in poses:
        # Assume pose is close enough to be excluded, prove otherwise
        close_enough = False
        for selected_pose in selected_poses:
            if distance(pose, selected_pose) < minimum_distance_between_agents:
                close_enough = True
                break
        if not close_enough:
            selected_poses.append(pose)
            
    return selected_poses

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
        
def optimize_raytrace_filtering(candidates, sample_poses, sensor_pos_adjust=None, sim=None):
    """
    Filter navigable candidates based on optimized raytrace evaluation.
    
    Arguments:
    candidates -- List of candidate positions.
    sample_poses -- Sample poses for evaluation.
    sensor_pos_adjust -- Sensor position adjustment.
    sim -- Simulation environment.
    """
    passed_cands = []
    for agent_cand in candidates:
        # print(agent_cand, sensor_pos_adjust)
        if sensor_pos_adjust is not None:
            raytrace_fail = any(eval_raytrace(pose, agent_cand+sensor_pos_adjust, sim) for pose in sample_poses)
        else:
            raytrace_fail = any(eval_raytrace(pose, agent_cand, sim) for pose in sample_poses)
        if not raytrace_fail:
            passed_cands.append(agent_cand)
    return passed_cands


def eval_raytrace(obj_pos, camera_pos, sim, threshold=0.02, num_samples=100):
    """
    Evaluate raytrace using vectorized operations and early termination.
    
    Arguments:
    obj_pos -- Position of the object.
    camera_pos -- Position of the camera.
    sim -- Simulation environment with pathfinder.
    threshold -- Distance threshold to consider as an obstacle hit.
    num_samples -- Number of samples for raytracing.
    """
    # offset = 0.0
    # offset = max(0.25 - np.sqrt((obj_pos[0] - camera_pos[0])**2 + (obj_pos[2] - camera_pos[2])**2) / num_samples, 0)
    # print(offset)
    t_values = np.linspace(0, 1, num_samples, endpoint=False)
    sampled_raytrace_points = (1 - t_values)[:, np.newaxis] * obj_pos + t_values[:, np.newaxis] * camera_pos
    raytrace_obstacles = np.array([sim.pathfinder.distance_to_closest_obstacle(point) for point in sampled_raytrace_points])
    return raytrace_obstacles.min() < threshold

def rotate_vector(vector, max_angle_deg=2):
    """
    Rotate a vector by a random angle within a specified range.
    
    Arguments:
    vector -- Initial vector to be rotated.
    max_angle_deg -- Maximum angle in degrees by which to rotate the vector.
    """
    # Convert maximum angle to radians
    max_angle_rad = np.radians(max_angle_deg)
    # Generate random rotation angles within ±max_angle_rad for azimuthal and polar angles
    delta_theta = np.random.uniform(-max_angle_rad, max_angle_rad)
    delta_phi = np.random.uniform(-max_angle_rad, max_angle_rad)
    
    # Convert vector to spherical coordinates
    rho = np.linalg.norm(vector)
    theta = np.arctan2(vector[1], vector[0]) + delta_theta  # Azimuthal angle
    phi = np.arccos(vector[2] / rho) + delta_phi  # Polar angle
    
    # Ensure the angles are within the valid range
    theta = theta % (2 * np.pi)
    phi = np.clip(phi, 0, np.pi)
    
    # Convert back to Cartesian coordinates
    rotated_vector = np.array([
        rho * np.sin(phi) * np.cos(theta),
        rho * np.sin(phi) * np.sin(theta),
        rho * np.cos(phi)
    ])
    
    return rotated_vector

import numpy as np

def select_facing_agents(all_pos, all_rot, high_facing_threshold, low_facing_threshold):
    for i in range(len(all_pos)):
        for j in range(i + 1, len(all_pos)):
            pos1, rot1 = all_pos[i], all_rot[i]
            pos2, rot2 = all_pos[j], all_rot[j]
            vector1 = np.array([np.cos(rot1[1]), np.sin(rot1[1])])  # Assuming rot stores (roll, pitch, yaw)
            vector2 = np.array([np.cos(rot2[1]), np.sin(rot2[1])])
            angle = angle_between_vectors(vector1, vector2)
            # print(angle, low_facing_threshold, high_facing_threshold)
            # print(low_facing_threshold < angle, angle < high_facing_threshold)
            if high_facing_threshold < angle < low_facing_threshold:
                return [pos1, pos2], [rot1, rot2]
    return [], []

def calculate_intersection(pos1, pos2, rot1, rot2):
    # Extract yaw angles, assuming the yaws are stored as the second element
    yaw1 = rot1[1]
    yaw2 = rot2[1]
    
    # Calculate direction vectors
    dir1 = np.array([np.cos(yaw1), np.sin(yaw1)])
    dir2 = np.array([np.cos(yaw2), np.sin(yaw2)])
    
    # Formulate the line equations
    # Line 1: pos1 + t * dir1
    # Line 2: pos2 + s * dir2
    # Set up the equation (dir1_x)t + (-dir2_x)s = pos2_x - pos1_x
    #                     (dir1_y)t + (-dir2_y)s = pos2_y - pos1_y
    A = np.array([dir1, -dir2]).T  # Coefficient matrix
    b = np.array([pos2[0] - pos1[0], pos2[2] - pos1[2]])  # Constant terms
    
    # Solve for t and s
    try:
        t_s = np.linalg.solve(A, b)
        t = t_s[0]
        s = t_s[1]
        
        tdir = t * dir1

        # Calculate intersection point using one of the line equations
        intersection = pos1 + [tdir[0], 0, tdir[1]]
        return intersection
    except np.linalg.LinAlgError:
        # If the lines are parallel or coincident, there may not be a solution
        return None


def sample_points_near(all_pos, radius=6.0, num_points=200):
    # Calculating the centroid of human positions
    human_centroid = np.mean(all_pos, 0)
    samples = (np.random.sample([num_points, 3]) - 0.5) * radius * 2
    samples[:, 1] = 0.0
    # print('222', human_centroid, samples[:2])
    sampled_points = human_centroid + np.array([0.0, 1.0, 0.0]) + samples
    return sampled_points

def generate_surrounding_points(center_point, radius):
    """
    Generates points surrounding a given center point within a specified radius.
    This is a placeholder function that needs proper implementation.
    
    Args:
    - center_point: The center point around which to generate points.
    - radius: The radius within which to generate points.
    
    Returns:
    - A list of points around the center point within the specified radius.
    """
    # Placeholder implementation. Ideally, this should generate points around the center point.
    # A simple, naive approach could just generate points at the cardinal directions.
    directions = np.array([[radius, 0, 0], [-radius, 0, 0], [0, radius, 0], [0, -radius, 0], [0, 0, radius], [0, 0, -radius]])
    surrounding_points = center_point + directions
    return surrounding_points


def optimize_raytrace_filtering_object(candidates, sample_poses, sensor_pos_adjust, sim=None):
    """
    Filter navigable candidates based on optimized raytrace evaluation.
    
    Arguments:
    candidates -- List of candidate positions.
    sample_poses -- Sample poses for evaluation.
    sensor_pos_adjust -- Sensor position adjustment.
    sim -- Simulation environment.
    """
    passed_cands = []
    for agent_cand in candidates:
        raytrace_fail = any(eval_raytrace(pose + sensor_pos_adjust, agent_cand.translation, sim) for pose in sample_poses)
        if not raytrace_fail:
            passed_cands.append(agent_cand)
    return passed_cands

import numpy as np

def angle_between_vectors(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return angle

def calculate_wedge_intersection(pos1, pos2, rot1, rot2, fov, radius):
    flat_pos1 = pos1[[0, 2]]
    flat_pos2 = pos2[[0, 2]]
    z = (pos1[1]+pos2[1])/2.0
    wedge1 = Wedge(flat_pos1, radius, np.degrees(rot1[1] - np.pi/4 - fov/2), np.degrees(rot1[1] - np.pi/4 + fov/2))
    wedge2 = Wedge(flat_pos2, radius, np.degrees(rot2[1] - np.pi/4 - fov/2), np.degrees(rot2[1] - np.pi/4 + fov/2))
    
    # Convert wedges to polygons
    poly1 = Polygon(wedge1.get_path().vertices)
    poly2 = Polygon(wedge2.get_path().vertices)
    
    # Calculate intersection
    intersection = poly1.intersection(poly2)
    
    # Check if intersection is a valid polygon
    if not intersection.is_empty and isinstance(intersection, Polygon):
        intersection_area = intersection.area
        overlap_percentage = intersection_area / poly1.area

        # x, y = intersection.exterior.coords.xy
        # intersection_points = np.c_[x, [z]*len(x), y]
        centroid = intersection.centroid.coords[0]
        centroid = np.array([centroid[0], z, centroid[1]])
        return overlap_percentage, centroid
    else:
        return None, None

def transform_vertex(vertex):
    vertex_transformed = [0, 0, 0]
    vertex_transformed[0] = vertex[0]
    vertex_transformed[1] = -vertex[2]
    vertex_transformed[2] = vertex[1]

    return vertex_transformed

def calculate_points_location(mesh_index=0, vertex_index=359):
    scene = trimesh.load("/home/terran/lingjunmao/scannetpp/test_scene/objects/example_objects/00582.obj")

    vertices = scene.vertices
    vertices[9120] = transform_vertex(vertices[9120])
    vertices[9448] = transform_vertex(vertices[9448])
    vertices[9929] = transform_vertex(vertices[9929])
    vertices[6] = transform_vertex(vertices[6])
    vertices[616] = transform_vertex(vertices[616])

    return [vertices[9120], vertices[9448], vertices[9929], vertices[6], vertices[616]]

def add_objects_to_env(rigid_obj_mgr, template_id, data_path, object_folder, pose_points, position, rotate=[-1.6, 0, -1.5], scale=[1.0, 1.0, 1.0]):
    """
    Adds multiple objects to the environment at different locations.

    :param env: The Habitat sim environment.
    :param data_path: The base path where object assets are stored.
    :param object_folder: The folder name under data_path containing the object config files.
    :param num_objects: The number of objects to add. Default is 3.
    :param scale: Uniform scaling factor for the object. Default is 0.2.
    """
    
    object_locs = []
    # Load the object's configurations and get the first template ID
    # Adjust the scale of the template
    
    # Instantiate the object in the environment
    obj = rigid_obj_mgr.add_object_by_template_id(template_id)
    obj.translation = position

    # w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
    # q_magnum = mn.Quaternion((x, y, z), w)
    # obj.rotation = q_magnum

    rotation_x = mn.Matrix4.rotation_x(mn.Rad(rotate[0]))
    rotation_y = mn.Matrix4.rotation_y(mn.Rad(rotate[1]))
    rotation_z = mn.Matrix4.rotation_z(mn.Rad(rotate[2]))

    # Combine these rotations to create a quaternion
    rotation_matrix_4 = (rotation_x @ rotation_y @ rotation_z)
    rotation_matrix_np = np.array(rotation_matrix_4, dtype=np.float32).reshape((4, 4))

    rotation_matrix = rotation_matrix_4.rotation_scaling()
    rotation_quaternion = mn.Quaternion.from_matrix(rotation_matrix)
    obj.rotation = rotation_quaternion

    return obj
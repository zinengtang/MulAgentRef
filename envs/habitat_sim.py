import os
import time
import random
import magnum
import copy
import numpy as np
import habitat_sim
from envs.habitat_lab_utils import *

HUMAN_SCALE = 0.0075
human_pose_mappings = {
    "00582": [[0.0, 0.7, -0.05], [-1.6, 0, 3.15], [HUMAN_SCALE]*3],
    "00382": [[0.05, 0.65, 0.05], [-1.6, 0, 3.15], [HUMAN_SCALE]*3],
    "00386": [[0.02, 0.7, 0.02], [-1.6, 0, 3.15], [HUMAN_SCALE]*3],
    "00479": [[0.0, 0.72, 0.05], [-1.6, 0, 3.45], [HUMAN_SCALE]*3],
    "00530": [[-0.0, 0.75, -0.03], [-1.6, 0.0, 3.45], [HUMAN_SCALE]*3],
}

class BaseSim():
    def __init__(self, scene_path="data/versioned_data/hm3d-0.2/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
                 gpu_id=0):
        self.num_objects = 3
        self.sample_radius = 0.2
        self.height = 0.7
        self.num_static_agents = 2
        self.minimum_distance_between_agents = 1.5
        self.minimum_distance_agent_object = 2.0
        self.scale_object = 0.3

        self.observation_key = 'rgb_camera'
        self.W = 256 * 4
        self.H = 256 * 4
        camera_resolution = [self.W, self.H]
        self.agent_cfgs = []
        for _ in range(self.num_static_agents):
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.height = self.height
            agent_cfg.radius = 0.17

            sensor_specs = []
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = self.observation_key
            rgb_sensor_spec.resolution = camera_resolution
            rgb_sensor_spec.position = 1.5 * habitat_sim.geo.UP + 0.25 * habitat_sim.geo.LEFT
            rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

            self.hfov = rgb_sensor_spec.hfov
            sensor_specs.append(rgb_sensor_spec)

            agent_cfg.sensor_specifications = sensor_specs
            self.agent_cfgs.append(agent_cfg)

        self.data_path = 'data'
        self.check_outofview = True
        self.check_hidden = True
        self.remove_objects = True
        
        self.gpu_id = gpu_id

        self.pose_points = calculate_points_location()

        self.reset(scene_path)
        self.register_object()

    def register_object(self):
        sim = self.sim
        self.rigid_obj_mgr = sim.get_rigid_object_manager()
        self.obj_templates_mgr = sim.get_object_template_manager()

        template_id = self.obj_templates_mgr.load_configs(
            str(os.path.join(self.data_path, "test_assets/objects/sphere"))
        )[0]
        template = self.obj_templates_mgr.get_template_by_id(template_id)
        template.scale = [self.scale_object] * 3
        self.obj_templates_mgr.register_template(template)
        self.template_id = template_id

        data_path = "/home/terran/lingjunmao/scannetpp/test_scene/objects/example_objects/"
        self.human_template_id = [self.obj_templates_mgr.load_configs(
            os.path.join(data_path, object_folder)
        )[0] for object_folder in human_pose_mappings]
        human_template = [self.obj_templates_mgr.get_template_by_id(_id) for _id in self.human_template_id]
        for i in range(len(human_template)):
            human_template[i].scale = list(human_pose_mappings.values())[i][2]
            self.obj_templates_mgr.register_template(human_template[i])

        target_template_id = self.obj_templates_mgr.load_configs(
            str(os.path.join(self.data_path, "test_assets/objects/sphere_blue"))
        )[0]
        template = self.obj_templates_mgr.get_template_by_id(target_template_id)
        template.scale = [self.scale_object] * 3
        self.obj_templates_mgr.register_template(template)
        self.target_template_id = target_template_id

    def reset(self, scene_path):
        print("-----", scene_path, "---------")
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = self.gpu_id
        
        navmesh_path = scene_path.replace('glb', 'navmesh')
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.read_from_json(navmesh_path)
        backend_cfg.navmesh_settings = navmesh_settings
        assert os.path.exists(backend_cfg.scene_id)
        
        cfg = habitat_sim.Configuration(backend_cfg, self.agent_cfgs)
        sim = habitat_sim.Simulator(cfg)
        self.sim = sim

        agent = sim.agents[0]
        self.sensor_pos_adjust = np.array([0, self.height, 0])
        sensor_pos = agent._sensors[self.observation_key]
        self.hfov = float(sensor_pos.hfov)

    def get_observations_at(self, position, look_at_quaternion):
        sim = self.sim
        agent = sim.agents[0]
        state = copy.deepcopy(agent.get_state())

        state.position = position
        state.rotation = look_at_quaternion
        agent.set_state(state, reset_sensors=False)
        obs = sim.get_sensor_observations()
        return obs
    
    def add_human(self, rot, pos, human_key):
        theta = rot[1] - np.pi / 2
        mapping = human_pose_mappings[human_key][0]
        x_prime = mapping[0] * np.cos(theta) - mapping[2] * np.sin(theta)
        y_prime = mapping[0] * np.sin(theta) + mapping[2] * np.cos(theta)
        human = add_objects_to_env(
            self.rigid_obj_mgr,
            self.human_template_id[list(human_pose_mappings).index(human_key)],
            "/home/terran/lingjunmao/scannetpp/test_scene/objects/example_objects/",
            human_key,
            self.pose_points,
            pos + np.array([0, mapping[1], 0]),
            np.array([0, 0, rot[1]]) + np.array(human_pose_mappings[human_key][1]),
            human_pose_mappings[human_key][2],
        )
        return np.array([x_prime, -mapping[1] + 0.5, y_prime]), human

    def generate_single_instance(self, height_diff=False, far_apart=False):
        while True:
            start = time.time()
            paired_all_obs = []
            paired_all_pos = []
            paired_all_rot = []

            while True:
                # start=time.time()
                objects, _ = self.generate_objects(height_diff=height_diff, far_apart=far_apart)
                # print(1, time.time()-start)
                cand_pair_all_pos, cand_pair_all_rot = self.generate_pos(objects)
                # print(2, time.time()-start)
                if len(cand_pair_all_pos) >= 1:
                    break
            
            objects_positions = [obj.translation for obj in objects]
            self.rigid_obj_mgr.remove_all_objects()

            for all_pos, all_rot in zip(cand_pair_all_pos, cand_pair_all_rot):
                all_obs = []
                count = 0
                human_keys = [random.choice(list(human_pose_mappings.keys())) for _ in all_pos]
                for pos, rot, human_key in zip(all_pos, all_rot, human_keys):
                    self.add_human(rot, pos, human_key)
                for pos, rot in zip(all_pos, all_rot):
                    objects_tmp = []
                    for i, point in enumerate(objects_positions):
                        if i == 0 and count == 1:
                            obj = self.rigid_obj_mgr.add_object_by_template_id(
                                self.target_template_id
                            )
                            obj.translation = point
                            objects_tmp.append(obj)
                        else:
                            obj = self.rigid_obj_mgr.add_object_by_template_id(
                                self.template_id
                            )
                            obj.translation = point
                            objects_tmp.append(obj)

                    obs = self.get_observations_at(
                        pos, quaternion.from_euler_angles(rot)
                    )
                    all_obs.append(obs)
                    count += 1
                    for obj in objects_tmp:
                        self.rigid_obj_mgr.remove_object_by_id(obj.object_id)
                paired_all_obs.append(all_obs)
                paired_all_pos.append(all_pos)
                paired_all_rot.append(all_rot)

                if self.remove_objects:
                    self.rigid_obj_mgr.remove_all_objects()
            if len(paired_all_obs) > 0:
                return (
                    paired_all_obs,
                    paired_all_pos,
                    paired_all_rot,
                    objects_positions,
                )
                
    def generate_objects(self, height_diff=False, far_apart=False):
        sim = self.sim
        min_distance_between_objects = 0.4  # Minimum desired distance between objects

        # Sample points on circles with varying radii centered at init_point
        while True:
            
            init_point = sim.pathfinder.get_random_navigable_point()
            if not height_diff:
                positions = []
                if far_apart:
                    surround_distance = 6.0
                else:
                    surround_distance = 1.0
                init_point = sim.pathfinder.get_random_navigable_point()
                for _ in range(self.num_objects):
                    # Get 3 navigable points near init_point
                    pos = sim.pathfinder.get_random_navigable_point_near(init_point, surround_distance, 100)
                    positions.append(list(pos))
                positions = np.array(positions)
                if all(
                    np.linalg.norm(pos1 - pos2) >= min_distance_between_objects
                    for i, pos1 in enumerate(positions)
                    for pos2 in positions[i + 1 :]
                ):
                # Positions satisfy distance criteria
                    break
            else:
                # start=time.time()
                landing_positions = []
                num_samples = 30  # Number of sample points
                radius_min = 0.01  # Minimum radius
                radius_max = 0.8  # Maximum radius
                sampled_points = []
                for _ in range(num_samples):
                    angle = random.uniform(0, 2 * np.pi)
                    radius = random.uniform(radius_min, radius_max)
                    x = init_point[0] + radius * np.cos(angle)
                    y = init_point[1]
                    z = init_point[2] + radius * np.sin(angle)
                    point = np.array([x, y+1.5, z])
                    sampled_points.append(point)
                
                landing_positions = []
                # For each sampled point, drop an object from a certain height and record where it lands
                previous_force_direction = None
                all_force_vector_magnum = []
                objects = []
                for point in sampled_points:
                    obj = self.rigid_obj_mgr.add_object_by_template_id(self.template_id)
                    obj.translation = point
                    objects.append(obj)
                        
                    if previous_force_direction is None:
                        # Generate a random initial force direction for the first object
                        force_direction = np.random.normal(0, 1, size=(3,))
                        previous_force_direction = force_direction
                    else:
                        # Rotate the previous force direction within 15 degrees for subsequent objects
                        force_direction = rotate_vector(previous_force_direction, 5)
                    force_direction[1] = np.clip(np.abs(force_direction[1]), 0.0, 0.8)
                    force_magnitude = np.random.uniform(15, 20)  # Adjust magnitude as needed
                    force_vector = force_direction / np.linalg.norm(force_direction) * force_magnitude
                    # Convert numpy array to magnum.Vector3
                    force_vector_magnum = magnum.Vector3(force_vector.tolist())
                    all_force_vector_magnum.append(force_vector_magnum)
                        
                    # obj = self.rigid_obj_mgr.add_object_by_template_id(self.template_id)
                    # obj.translation = point + np.array([0.0, 1.2, 0.0])
                    # # application_point_magnum = magnum.Vector3(0.0, 0.0, 0.0)
                    # # obj.apply_force(1.0 * sim.get_gravity() * obj.mass, application_point_magnum)
                    # sim.step_physics(1.0)  # Simulate for enough time for the object to land
                    # x, y, z = list(obj.translation)
                    # # near_navigable_point = list(sim.pathfinder.get_random_navigable_point_near(obj.translation, 4.0, 100))
                    # # distance = np.sqrt((near_navigable_point[0] - x)**2 + (near_navigable_point[2] - z)**2)
                    # if y < init_point[1] - 0.5:
                    #     continue
                    # landing_positions.append(list(obj.translation))
                    # self.rigid_obj_mgr.remove_object_by_id(obj.object_id)
                application_point_magnum = magnum.Vector3(0.0, 0.0, 0.0)
                for i in range(len(objects)):
                    objects[i].apply_force(all_force_vector_magnum[i], application_point_magnum)
                    objects[i].apply_force(-1.0 * sim.get_gravity() * objects[i].mass, application_point_magnum)
                sim.step_physics(0.4)
                for i in range(len(objects)):
                    objects[i].linear_velocity = magnum.Vector3(0.0, 0.0, 0.0)
                    objects[i].apply_force(4*sim.get_gravity() * objects[i].mass, application_point_magnum)
                sim.step_physics(1.0)
                positions = np.array([obj.translation for obj in objects])
                self.rigid_obj_mgr.remove_all_objects()
                landing_positions = []
                for item in positions:
                    # print(item[1], init_point[1])
                    if item[1] > init_point[1] - 0.001:
                        landing_positions.append(item)
                # print('a', time.time()-start)
                landing_positions = np.array(landing_positions)
                # print('----', len(landing_positions), '-----')
                if len(landing_positions) < 3:
                    continue

                # Organize positions into height bins
                bin_size = 0.1
                height_min = landing_positions[:, 1].min()
                height_max = landing_positions[:, 1].max()
                if height_max - height_min < 0.3:
                    continue

                bin_edges = np.arange(height_min, height_max + bin_size, bin_size)
                # Create bins and assign positions to them
                bins_indices = np.digitize(landing_positions[:, 1], bin_edges)
                height_bins = {}
                for idx, bin_idx in enumerate(bins_indices):
                    bin_key = bin_edges[bin_idx - 1]  # bin_idx starts from 1
                    if bin_key not in height_bins:
                        height_bins[bin_key] = []
                    height_bins[bin_key].append(landing_positions[idx])
                if height_diff:
                    if len(height_bins) >= 2:
                        break
                else:
                    break

        if height_diff:
            # Try to select positions from min, mid, and max bins
            sorted_bins = sorted(height_bins.keys())
            num_bins = len(sorted_bins)
            attempts = 0
            max_attempts = 5
            while attempts < max_attempts:
                attempts += 1
                selected_positions = []
                if num_bins >= self.num_objects:
                    bin_indices = [0, num_bins // 2, num_bins - 1]
                else:
                    bin_indices = [i % num_bins for i in range(self.num_objects)]
                for idx in bin_indices:
                    bin_key = sorted_bins[idx]
                    positions_in_bin = height_bins[bin_key]
                    pos = random.choice(positions_in_bin)
                    selected_positions.append(pos)
                # Check for minimum distance between selected positions
                if all(
                    np.linalg.norm(pos1 - pos2) >= min_distance_between_objects
                    for i, pos1 in enumerate(selected_positions)
                    for pos2 in selected_positions[i + 1 :]
                ):
                    # Positions satisfy distance criteria
                    break

            positions = np.array(selected_positions[: self.num_objects])
            
        # Now, create objects at these positions
        objects = []
        for i in range(self.num_objects):
            obj = self.rigid_obj_mgr.add_object_by_template_id(self.template_id)
            obj.translation = positions[i]
            objects.append(obj)

        print(positions, init_point)
        return objects, init_point

    def generate_pos(self, objects):
        sim = self.sim
        sample_poses = [obj.translation for obj in objects]
        centroid = np.mean(sample_poses, axis=0)
        new_agent_position_navigable_cands = [
            sim.pathfinder.get_random_navigable_point_near(centroid, 5.0, 100)
            for _ in range(60)
        ]
        new_agent_position_navigable_cands = [
            pos for pos in new_agent_position_navigable_cands
            if np.abs(pos[1] - centroid[1]) < 1.5 and
            all(np.linalg.norm(pos - obj_pos) >= self.minimum_distance_agent_object for obj_pos in sample_poses)
        ]
        # print('candidates0', len(new_agent_position_navigable_cands))
        passed_cands = optimize_raytrace_filtering(
            new_agent_position_navigable_cands, sample_poses, self.sensor_pos_adjust, sim
        )
        all_pos = []
        all_rot = []
        pair_all_pos = []
        pair_all_rot = []

        for pos in passed_cands:
            sensor_pos = self.sensor_pos_adjust + pos
            euler_angles = calculate_euler_angles(sensor_pos, centroid)
            all_pos.append(pos)
            all_rot.append(euler_angles)

        for i in range(len(all_pos)):
            for j in range(len(all_pos)):
                if i == j:
                    continue
                pos_speaker = all_pos[j]
                pos_listener = all_pos[i]
                rot_speaker = all_rot[j]
                rot_listener = all_rot[i]
                distance = calculate_distance(pos_speaker, pos_listener)

                if self.minimum_distance_between_agents <= distance:
                    # Check if listener is within speaker's FOV
                    if self.is_in_fov(pos_speaker, rot_speaker, pos_listener):
                        # Check if all spheres are in FOV of both speaker and listener
                        if self.check_objects_in_fov(pos_speaker, rot_speaker, sample_poses) and \
                                self.check_objects_in_fov(pos_listener, rot_listener, sample_poses):

                            # Simulate human obstruction without adding humans
                            human_width = 0.6  # Width of the human in meters
                            if self.is_human_obstructing_view(pos_speaker, pos_listener, sample_poses, human_width):
                                continue  # Human obstructs the view to the spheres from speaker
                            if self.is_human_obstructing_view(pos_listener, pos_speaker, sample_poses, human_width):
                                continue  # Human obstructs the view to the spheres from listener
                            # if len(optimize_raytrace_filtering([pos_listener], sample_poses, self.sensor_pos_adjust, sim)) == 0:
                            #     continue
                            # if len(optimize_raytrace_filtering([pos_speaker], sample_poses, self.sensor_pos_adjust, sim)) == 0:
                                # continuex
                            pair_all_pos.append([pos_speaker, pos_listener])
                            pair_all_rot.append([rot_speaker, rot_listener])
                            return pair_all_pos, pair_all_rot
        return pair_all_pos, pair_all_rot
    def is_human_obstructing_view(self, agent_pos, human_pos, objects_positions, human_width):
        """
        Checks if a human at human_pos with given width obstructs the view from agent_pos to any of the objects.
        """
        for obj_pos in objects_positions:
            if self.does_human_block_line_of_sight(agent_pos, obj_pos, human_pos, human_width):
                return True  # Human obstructs the view
        return False  # No obstruction

    def does_human_block_line_of_sight(self, camera_pos, object_pos, human_pos, human_width):
        """
        Determines if a human at human_pos with width human_width blocks the line of sight from camera_pos to object_pos.
        This simplified version interpolates between camera_pos and object_pos and checks if human_pos is close to any point.
        """
        num_samples = 20  # Number of interpolation points
        threshold_distance = human_width / 2.0

        for i in range(num_samples + 1):
            t = i / num_samples  # Parameter t varies from 0 to 1
            interpolated_point = camera_pos * (1 - t) + object_pos * t
            distance = np.linalg.norm(interpolated_point - human_pos)
            if distance <= threshold_distance:
                return True  # Human obstructs the view
        return False  # No obstruction


    def is_in_fov(self, pos_speaker, rot_speaker, pos_listener):
        """
        Checks if the listener is within the speaker's horizontal field of view, ignoring pitch.
        """
        # Calculate the direction vector from speaker to listener
        vector_to_listener = pos_listener - pos_speaker
        vector_to_listener[1] = 0  # Project onto horizontal plane
        vector_to_listener = - vector_to_listener / np.linalg.norm(vector_to_listener)

        # Extract yaw from rot_speaker
        yaw = rot_speaker[1]    # Rotation around Y-axis (in radians)

        # Adjust yaw by adding or subtracting pi radians to align coordinate frames
        yaw_adjusted = yaw - 1.0 * np.pi  # You may need to adjust this based on your coordinate system

        # Compute the speaker's forward vector from adjusted yaw
        # Assuming the coordinate system where Z is forward
        speaker_forward = np.array([
            np.sin(yaw_adjusted),  # X component
            0,                     # Y component (horizontal plane)
            np.cos(yaw_adjusted)   # Z component
        ])

        speaker_forward = speaker_forward / np.linalg.norm(speaker_forward)

        # Compute the angle between the speaker's forward vector and the vector to the listener
        dot_product = np.dot(speaker_forward, vector_to_listener)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # Convert FOV from degrees to radians if necessary
        hfov_rad = np.deg2rad(self.hfov) if self.hfov > np.pi else self.hfov

        # Check if the angle is within half of the horizontal FOV
        if angle <= hfov_rad / 2:
            return True
        else:
            return False
    
    def check_objects_in_fov(self, agent_pos, agent_rot, objects_positions, offset=64):
        """
        Checks if all objects are within the agent's field of view.
        """
        camera_position = self.sensor_pos_adjust + agent_pos
        camera_rotation = quaternion.from_euler_angles(agent_rot)
        in_view = True
        for obj_pos in objects_positions:
            # For each object, get its bounding box in image coordinates
            collision_asset_size = [0.1, 0.1, 0.1]  # Assuming spheres of radius 0.1
            box_corners = sample_uniform_points_in_box(obj_pos, np.array(collision_asset_size))
            pixel_coordinates = [world_to_camera_coordinates(corner, camera_position, camera_rotation, self.hfov, self.W, self.H) for corner in box_corners]
            # Compute bounding box
            x_coords = [coord[0] for coord in pixel_coordinates]
            y_coords = [coord[1] for coord in pixel_coordinates]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            # Check if bbox is within image boundaries
            if bbox[0] >= self.W-offset or bbox[2]<offset or bbox[1] >= self.H-offset or bbox[3] < offset:
                # Object is outside the image
                in_view = False
                break
        return in_view

    def get_draw_bounding_boxes(self, inputs, obs, camera_position, camera_rotation, use_translation=False):
        bbox_image = obs[self.observation_key].copy()
        bboxes = []
        for i, inputs_i in enumerate(inputs):
            if use_translation:
                translation = inputs_i
                collision_asset_size = [0.1] * 3
            else:
                translation = inputs_i.translation
                collision_asset_size = [0.25] * 3
            box_corners = sample_uniform_points_in_box(translation, np.array(collision_asset_size))
            pixel_coordinates = [world_to_camera_coordinates(corner, camera_position, camera_rotation, self.hfov, self.W, self.H) for corner in box_corners]
            bbox, bbox_image = draw_bounding_box(bbox_image, pixel_coordinates, self.W, draw=(i == 0))
            bboxes.append(bbox)
        return bboxes, bbox_image
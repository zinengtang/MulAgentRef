import argparse
import glob
import json
import logging
import os
import random
import warnings
from multiprocessing import Pool

import pandas as pd
from PIL import Image
from tqdm import tqdm

from envs import BaseSim  # Assuming BaseSim is defined appropriately

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate scene observations and save them.")
    parser.add_argument('--job_id', type=int, default=2, help='Job ID for processing')
    parser.add_argument('--gpu_id', type=int, default=3, help='Base GPU ID to use for processing')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process')
    parser.add_argument('--height_diff', action='store_true', help='Whether to use height differences')
    parser.add_argument('--far_apart', action='store_true', help='Whether to make spheres far apart')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes to use for multi-processing')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval to save data CSV in terms of processed paths')
    return parser.parse_args()


def initialize_output_dirs(output_base_dir):
    """
    Removes the existing output directory and creates a new one with an images subdirectory.

    Args:
        output_base_dir (str): Path to the output base directory.

    Returns:
        str: Path to the images directory within the output base directory.
    """
    if os.path.exists(output_base_dir):
        logging.info(f"Removing existing directory: {output_base_dir}")
        os.system(f"rm -rf {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)
    images_dir = os.path.join(output_base_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    return images_dir

def process_path(args):
    """
    Processes a single path by generating instances, saving images, and collecting metadata.
    
    Args:
        args (tuple): Arguments tuple containing path, gpu_id, and other parameters.

    Returns:
        tuple: A tuple containing a boolean success status and an error message (if any).
    """
    path, gpu_id, images_dir, instance_id_start, height_diff, far_apart, output_base_dir, instances_per_path, csv_filename = args
    data_list = []
    key = os.path.basename(path).split('-')[-1]
    glb_file = os.path.join(path, f'{key}.basis.glb')

    try:
        logging.info(f"Processing GLB file: {glb_file}")
        env = BaseSim(glb_file, gpu_id=gpu_id)

        for i in range(instances_per_path):
            instance_id = instance_id_start + i

            (
                paired_all_obs,
                paired_all_pos,
                paired_all_rot,
                objects_loc
            ) = env.generate_single_instance(height_diff=height_diff, far_apart=far_apart)

            data = {
                'instance_id': instance_id,
                'scene_key': key,
            }

            for role_idx, role_name in enumerate(['listener', 'speaker']):
                obs = paired_all_obs[0][role_idx]
                obs_image_filename = f'{role_name}.jpg'

                instance_images_dir = os.path.join(images_dir, str(instance_id))
                os.makedirs(instance_images_dir, exist_ok=True)

                obs_image_path = os.path.join(instance_images_dir, obs_image_filename)

                try:
                    img = Image.fromarray(obs['rgb_camera']).convert('RGB')
                    img.save(obs_image_path)
                except Exception as e:
                    logging.error(f"Error saving {role_name} image for instance {instance_id}: {e}")
                    return (False, f"Error saving {role_name} image: {e}")

                pos = paired_all_pos[0][role_idx].tolist()
                rot = paired_all_rot[0][role_idx].tolist()

                pos_json = json.dumps(str({'pos': pos, 'rot': rot}))
                data[f'{role_name}_view_path'] = os.path.relpath(obs_image_path, output_base_dir)
                data[f'{role_name}_pos'] = pos_json

            data['object_loc'] = json.dumps({
                'target': list(objects_loc[0]),
                'distract': [list(objects_loc[1]), list(objects_loc[2])]
            })

            data_list.append(data)

        env.sim.close()

        # Save the data directly after processing each path
        if data_list:
            df = pd.DataFrame(data_list)
            # Append to CSV file instead of overwriting
            df.to_csv(os.path.join(output_base_dir, csv_filename), mode='a', header=not os.path.exists(os.path.join(output_base_dir, csv_filename)), index=False)

        return (True, None)  # Success, no error
    except Exception as e:
        logging.error(f"Error processing path {path}: {e}")
        return (False, f"Error processing path {path}: {e}")  # Failure with error message


def main():
    args = parse_arguments()

    # Assign variables from arguments
    job_id = args.job_id
    gpu_id = args.gpu_id
    height_diff = args.height_diff
    far_apart = args.far_apart
    if height_diff:
        extra_path = '_hd'
    elif far_apart:
        extra_path = '_far'
    else:
        extra_path = ''
    output_base_dir = f'final_data/output_data_{args.split}{extra_path}_{job_id}'
    scannet_list_proc = list(glob.iglob(f'data/versioned_data/hm3d-0.2/hm3d/{args.split}/*'))
    random.shuffle(scannet_list_proc)

    images_dir = initialize_output_dirs(output_base_dir)

    instances_per_path = 5 if height_diff else 30  # Number of instances to generate per path
    instance_id_start = 0
    num_gpus = 8  # Adjust based on your system
    csv_filename = 'data.csv'  # Final CSV file

    # Multi-processing using Pool
    num_processes = args.num_processes
    save_interval = args.save_interval

    with Pool(processes=num_processes) as pool:
        path_args = []
        for idx, path in enumerate(tqdm(scannet_list_proc, desc='Processing paths')):
            # Rotate GPU ID for each path
            current_gpu_id = idx % num_gpus
            path_args.append((path, current_gpu_id, images_dir, instance_id_start, height_diff, far_apart, output_base_dir, instances_per_path, csv_filename))
            instance_id_start += instances_per_path

        # Process the paths in parallel using Pool
        for success, error_message in tqdm(pool.imap_unordered(process_path, path_args), total=len(path_args), desc="Processing Results"):
            if not success:
                logging.error(f"Processing failed: {error_message}")

if __name__ == "__main__":
    main()

import blenderproc as bproc
import sys
import argparse
import os
import numpy as np
import random
from pathlib import Path
import json
import signal
from contextlib import contextmanager
import blenderproc.python.renderer.RendererUtility as RendererUtility
from time import time
import bpy # Import bpy to access Blender's internal data structures

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", help="Path to the 3D FRONT texture folder.")
    parser.add_argument("front_json", help="Path to a 3D FRONT scene json file, e.g.6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json.")
    parser.add_argument('cc_material_folder', nargs='?', default="resources/cctextures",
                        help="Path to CCTextures folder, see the /scripts for the download script.")
    parser.add_argument("output_folder", nargs='?', default="examples/datasets/front_3d_with_improved_mat/renderings",
                        help="Path to where the data should be saved")
    parser.add_argument("--n_views_per_scene", type=int, default=100,
                        help="The number of views to render in each scene.")
    parser.add_argument("--append_to_existing_output", type=bool, default=True,
                        help="If append new renderings to the existing ones.")
    # FOV is not used for panoramic rendering
    # parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=1024, help="Image width. Height will be width / 2 for a 2:1 aspect ratio.")
    # res_y is now calculated from res_x
    # parser.add_argument("--res_y", type=int, default=512, help="Image height.")
    return parser.parse_args()


class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_folders(args):
    front_folder = Path(args.front_folder)
    future_folder = Path(args.future_folder)
    front_3D_texture_folder = Path(args.front_3D_texture_folder)
    cc_material_folder = Path(args.cc_material_folder)
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir()
    return front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder


def check_name(name, category_name):
    return True if category_name in name.lower() else False


if __name__ == '__main__':
    '''Parse folders / file paths'''
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder = get_folders(args)
    front_json = front_folder.joinpath(args.front_json)
    n_cameras = args.n_views_per_scene

    failed_scene_name_file = output_folder.parent.joinpath('failed_scene_names.txt')

    if not front_folder.exists() or not future_folder.exists() \
            or not front_3D_texture_folder.exists() or not cc_material_folder.exists():
        raise Exception("One of these folders does not exist!")

    scene_name = front_json.name[:-len(front_json.suffix)]
    print('Processing scene name: %s.' % (scene_name))

    '''Pass those failure cases'''
    if failed_scene_name_file.is_file():
        with open(failed_scene_name_file, 'r') as file:
            failure_scenes = file.read().splitlines()
        if scene_name in failure_scenes:
            print('File in failure log: %s. Continue.' % (scene_name))
            sys.exit(0)

    '''Pass already generated scenes.'''
    scene_output_folder = output_folder.joinpath(scene_name)
    existing_n_renderings = 0

    if scene_output_folder.is_dir():
        existing_n_renderings = len(list(scene_output_folder.iterdir()))
        if existing_n_renderings >= n_cameras:
            print('Scene %s is already generated.' % (scene_output_folder.name))
            sys.exit(0)

    if args.append_to_existing_output:
        n_cameras = n_cameras - existing_n_renderings

    # try:
    if 1 < 2:  # for debugging purposes
        with time_limit(600): # per scene generation would not exceeds X seconds.
            start_time = time()

            bproc.init()

            # --- START: Panoramic Camera Configuration ---

            # 1. Set the render engine to CYCLES. This is mandatory for panoramic rendering.
            # bproc.renderer.set_engine('CYCLES')
            RendererUtility.set_max_amount_of_samples(32)

            # 2. Set camera resolution to a 2:1 aspect ratio.
            # The height is automatically calculated from the width.
            image_width = args.res_x
            image_height = args.res_x // 2
            bproc.camera.set_resolution(image_width, image_height)

            # 3. Access the camera data block and set it to panoramic mode.
            camera = bpy.context.scene.camera
            cam_data = camera.data
            cam_data.type = 'PANO'
            cam_data.cycles.panorama_type = 'EQUIRECTANGULAR'

            # NOTE: Camera intrinsics (K matrix) and FOV are not applicable
            # to equirectangular cameras, so we skip setting/saving them.

            # --- END: Panoramic Camera Configuration ---

            mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
            mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

            # set the light bounces
            bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                             transmission_bounces=200, transparent_max_bounces=200)

            # read 3d future model info
            with open(future_folder.joinpath('model_info_revised.json'), 'r') as f:
                model_info_data = json.load(f)
            model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' for
                                 m in
                                 model_info_data}

            # load the front 3D objects
            loaded_objects = bproc.loader.load_front3d(
                json_path=str(front_json),
                future_model_path=str(future_folder),
                front_3D_texture_path=str(front_3D_texture_folder),
                label_mapping=mapping,
                model_id_to_label=model_id_to_label)

            print('Loaded %d objects.' % len(loaded_objects))

            # Material sampling remains the same
            print('Sampling materials...')
            cc_materials = bproc.loader.load_ccmaterials(str(cc_material_folder), ["Bricks", "Wood", "Carpet", "Tile", "Marble"])
            print('Loaded %d CC materials.' % len(cc_materials))
            floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
            for floor in floors:
                for i in range(len(floor.get_materials())):
                    floor.set_material(i, random.choice(cc_materials))
            print('Set materials for %d floors.' % len(floors))

            baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
            wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
            for obj in baseboards_and_doors:
                # For each material of the object
                for i in range(len(obj.get_materials())):
                    # Replace the material with a random one
                    obj.set_material(i, random.choice(wood_floor_materials))

            walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
            marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
            for wall in walls:
                # For each material of the object
                for i in range(len(wall.get_materials())):
                    wall.set_material(i, random.choice(marble_materials))

            # Camera pose sampling remains the same
            point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)
            bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

            cam_Ts = []
            for _ in range(n_cameras):
                # sample cam loc inside house
                height = np.random.uniform(1.4, 1.8)
                location = point_sampler.sample(height)
                # Sample rotation
                rotation = np.random.uniform([np.pi*0.4, 0, 0], [np.pi*0.6, 0, np.pi * 2])
                cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

                # Add pose to the camera
                bproc.camera.add_camera_pose(cam2world_matrix)
                cam_Ts.append(cam2world_matrix)

            print('Sampled %d camera poses.' % len(cam_Ts))

            # Enable depth output (will also be panoramic)
            bproc.renderer.enable_depth_output(activate_antialiasing=False)

            print(f'Rendering {n_cameras} panoramic images...')
            data = bproc.renderer.render(return_data=True, output_dir=str(scene_output_folder))

            print(f"Rendered {len(data['colors'])} panoramic images and {len(data['depth'])} panoramic depth maps.")

            # # Add camera poses to the data for saving
            # data['cam_poses'] = cam_Ts

            # # Write the data to a .hdf5 container
            # bproc.writer.write_hdf5(str(scene_output_folder), data,
            #                         append_to_existing_output=args.append_to_existing_output)

            # print('Time elapsed: %f.' % (time()-start_time))

    # except TimeoutException as e:
    #     print('Time is out: %s.' % scene_name)
    #     with open(failed_scene_name_file, 'a') as file:
    #         file.write(scene_name + "\n")
    #     sys.exit(0)
    # except Exception as e:
    #     print(f'Failed scene name: {scene_name} with error: {e}')
    #     with open(failed_scene_name_file, 'a') as file:
    #         file.write(scene_name + "\n")
    #     sys.exit(0)

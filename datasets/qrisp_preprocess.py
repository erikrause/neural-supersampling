from PIL import Image
import numpy as np
import os
import shutil
import imageio
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# Input parameters
dataset_path = r"D:\Downloads\Qualcomm Unity Dataset\QRISP\TrainSet"

input_res_h = 270
input_res_w = 480
gt_res_h = 1080
gt_res_w = 1920

gt_ext = "png"
color_ext = "png"
depth_ext = "png"
motion_ext = "exr"

gt_folder = "Enhanced"
color_folder = "MipBiasMinus2"
depth_folder = "DepthMipBiasMinus2"
motion_folder = "MotionVectorsMipBiasMinus2"


# Output parameters
dataset_output_root = r"D:\repos\MultiframeSR\neural-supersampling\datasets\QRISP"

depth_out_ext = "exr"

gt_out_folder = "enhanced"
color_out_folder = "color"
depth_out_folder = "depth"
motion_out_folder = "motion"

rendering_engine = "unity"


if __name__ == "__main__":
    # scenes = os.listdir(dataset_path)
    # #sequences = []
    # for scene in scenes:
    #     print(scene)
    #     gt_sequences_path = os.path.join(dataset_path, scene, f"{gt_res_h}p", gt_folder)
    #     color_sequences_path = os.path.join(dataset_path, scene, f"{input_res_h}p", color_folder)
    #     depth_sequences_path = os.path.join(dataset_path, scene, f"{input_res_h}p", depth_folder)
    #     motion_sequences_path = os.path.join(dataset_path, scene, f"{input_res_h}p", motion_folder)
    #     gt_sequences = os.listdir(gt_sequences_path)
    #     color_sequences = os.listdir(color_sequences_path)
    #     depth_sequences = os.listdir(depth_sequences_path)
    #     motion_sequences = os.listdir(motion_sequences_path)
    #
    #     assert len(color_sequences) == len(depth_sequences) \
    #            and len(color_sequences) == len(motion_sequences) \
    #            and len(color_sequences) == len(gt_sequences)
    #     for sequence in color_sequences:
    #         #sequences.append(sequence)
    #         print(sequence)
    #         curr_gt_sequence_path = os.path.join(gt_sequences_path, sequence)
    #         curr_color_sequence_path = os.path.join(color_sequences_path, sequence)
    #         curr_depth_sequence_path = os.path.join(depth_sequences_path, sequence)
    #         curr_motion_sequence_path = os.path.join(motion_sequences_path, sequence)
    #
    #         assert len(os.listdir(curr_color_sequence_path)) == len(os.listdir(curr_depth_sequence_path)) \
    #                and len(os.listdir(curr_color_sequence_path)) == len(os.listdir(curr_motion_sequence_path)) \
    #                and len(os.listdir(curr_color_sequence_path)) == len(os.listdir(curr_gt_sequence_path))
    #         for filename in os.listdir(curr_color_sequence_path):
    #             frame_idx, _ext = os.path.splitext(filename)
    #             curr_gt = os.path.join(curr_gt_sequence_path, f"{frame_idx}.{gt_ext}")
    #             curr_color = os.path.join(curr_color_sequence_path, f"{frame_idx}.{color_ext}")
    #             curr_depth = os.path.join(curr_depth_sequence_path, f"{frame_idx}.{depth_ext}")
    #             curr_motion = os.path.join(curr_motion_sequence_path, f"{frame_idx}.{motion_ext}")
    #
    #             shutil.copy(curr_gt,
    #                         os.path.join(dataset_output_root,
    #                                      gt_out_folder,
    #                                      f"{scene}{sequence}_{rendering_engine}_{gt_res_w}_{gt_res_h}_{frame_idx}.{gt_ext}"))
    #             shutil.copy(curr_color,
    #                         os.path.join(dataset_output_root,
    #                                      color_out_folder,
    #                                      f"{scene}{sequence}_{rendering_engine}_{input_res_w}_{input_res_h}_{frame_idx}.{color_ext}"))
    #             shutil.copy(curr_depth,
    #                         os.path.join(dataset_output_root,
    #                                      depth_out_folder,
    #                                      f"{scene}{sequence}_{rendering_engine}_{input_res_w}_{input_res_h}_{frame_idx}.{depth_ext}"))
    #             shutil.copy(curr_motion,
    #                         os.path.join(dataset_output_root,
    #                                      motion_out_folder,
    #                                      f"{scene}{sequence}_{rendering_engine}_{input_res_w}_{input_res_h}_{frame_idx}.{motion_ext}"))


    # Decode depth from QRISP encoded value:
    depth_folder = os.path.join(dataset_output_root,
                                depth_out_folder)
    for filename in os.listdir(depth_folder):
        filepath = os.path.join(depth_folder, filename)
        encoded_depth = np.array(Image.open(filepath))
        depth = (encoded_depth[..., 0] / 255 +
                 encoded_depth[..., 1] / 255 ** 2 +
                 encoded_depth[..., 2] / 255 ** 3 +
                 encoded_depth[..., 3] / 255 ** 4)

        filename, _ext = os.path.splitext(filename)
        filepath_out = os.path.join(depth_folder, f"{filename}.{depth_out_ext}")
        depth = depth.astype(np.float32)
        cv2.imwrite(filepath_out, depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

        os.remove(filepath)

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import ray
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor


class BatchPredictor:

    def __init__(self):
        self.predictor = build_sam2_video_predictor(model_cfg_path, checkpoint_path)
        self.mask_generator = SAM2AutomaticMaskGenerator(model)

    # This shows the mask on the image
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap('tab10')
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    # Show the points on the image
    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color='green',
            marker='*',
            s=marker_size,
            edgecolor='white',
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color='red',
            marker='*',
            s=marker_size,
            edgecolor='white',
            linewidth=1.25,
        )

    # Define the function to extract frames from the video
    def extract_frames(self, video_path, output_dir):
        os.system(f'ffmpeg -i {video_path} -q:v 2 -start_number 0 {output_dir}/"%05d.jpg"')
    
    def __call__(self)
        


# Define the inference function
def inference(name: str,
              data_sample_name: str,
              dataset_dir: str,
              output_dir: str,
              model_cfg_path: str,
              checkpoint_path: str,
              model: 'torch.nn.Module'):
    predictor = build_sam2_video_predictor(model_cfg_path, checkpoint_path)

    mask_generator = SAM2AutomaticMaskGenerator(model)

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        video_file = f'{dataset_dir}/{data_sample_name}/{name}.mp4'
        video_frame_dir = f'{output_dir}/{data_sample_name}/frames/{name}'
        output_folder = f'{output_dir}/{data_sample_name}/{name}'
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(video_frame_dir, exist_ok=True)
        extract_frames(video_file, video_frame_dir)

        frame_names = [
            p
            for p in os.listdir(video_frame_dir)
            if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_frame_dir)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # get the first frame
        first_frame = Image.open(os.path.join(video_frame_dir, frame_names[ann_frame_idx]))
        first_frame = np.array(first_frame.convert('RGB'))
        masks = mask_generator.generate(first_frame)
        if not masks:
            with open(f'{output_folder}/{name}_no_mask.txt', 'w') as f:
                f.write('No mask found')
            return

        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=masks[0]['segmentation'],
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # render the segmentation results every few frames
        vis_frame_stride = 15
        plt.close('all')

        # render the segmentation results every few frames
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.close('all')
            plt.figure(figsize=(6, 4))
            plt.title(f'frame {out_frame_idx}')
            plt.imshow(Image.open(os.path.join(video_frame_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                plt.savefig(f'{output_folder}/{name}_{out_frame_idx}.jpeg')


if __name__ == '__main__':
    # get the data path
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default='sam2_hiera_tiny.pt')
    parser.add_argument('--model-cfg-path', type=str, default='sam2_hiera_t.yaml')
    parser.add_argument('--data-sample-name', type=str, default='sav_000')
    parser.add_argument('--dataset-dir', type=str, default='/dataset')
    parser.add_argument('--output-dir', type=str, default='/outputs')
    args = parser.parse_args()

    args.checkpoint_path = os.path.expanduser(args.checkpoint_path)
    args.model_cfg_path = os.path.expanduser(args.model_cfg_path)
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.output_dir = os.path.expanduser(args.output_dir)

    # Get all videos
    videos = os.listdir(os.path.join(args.dataset_dir, args.data_sample_name))
    videos = [video for video in videos if video.endswith('.mp4')]
    videos.sort()

    model = build_sam2(args.model_cfg_path, args.checkpoint_path, device='cuda', apply_postprocessing=False)

    # Run inference on all videos
    for video in videos:
        inference(name=video[:-len('.mp4')],
                  data_sample_name=args.data_sample_name,
                  dataset_dir=args.dataset_dir,
                  output_dir=args.output_dir,
                  model_cfg_path=args.model_cfg_path,
                  checkpoint_path=args.checkpoint_path,
                  model=model)

import glob
import json
import copy
import os
import pandas as pd
import numpy as np
import cv2
import librosa
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch

from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import binary_dilation
import warnings
warnings.filterwarnings("ignore")

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except ModuleNotFoundError:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None
    SamPredictor = None


class GPT4V(object):

    def __init__(self, cfg):
        self.cfg = cfg

        if 'sam' in self.cfg.region_division_methods:
            if sam_model_registry is None or SamAutomaticMaskGenerator is None:
                raise ModuleNotFoundError(
                    "segment_anything is required for method 'sam' but is not installed. "
                    "Remove 'sam' from --region_division_methods or install segment_anything."
                )
            self.sam = sam_model_registry['vit_h'](checkpoint='pretrain/sam_vit_h_4b8939.pth')
            self.sam.to(device=self.cfg.device)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def region_division(self):
        image_files = []
        if getattr(self.cfg, "input_dir", None):
            image_files = glob.glob(os.path.join(self.cfg.input_dir, "**", "*.png"), recursive=True)
        elif self.cfg.dataset_name in ['mvtec']:
            root = self.cfg.dataset_name
            image_files = glob.glob(f'{root}/*/test/*/???.png')
            # image_files = [image_file.replace(f'{root}/', '') for image_file in image_files]
        elif self.cfg.dataset_name in ['visa']:
            root = self.cfg.dataset_name
            CLSNAMES = [
                'pcb1', 'pcb2', 'pcb3', 'pcb4',
                'macaroni1', 'macaroni2', 'capsules', 'candle',
                'cashew', 'chewinggum', 'fryum', 'pipe_fryum',
            ]
            csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)
            columns = csv_data.columns  # [object, split, label, image, mask]
            test_data = csv_data[csv_data[columns[1]] == 'test']
            for cls_name in CLSNAMES:
                cls_data = test_data[test_data[columns[0]] == cls_name]
                cls_data.index = list(range(len(cls_data)))
                for idx in range(cls_data.shape[0]):
                    data = cls_data.loc[idx]
                    image_files.append(data[3])
            image_files = [f'{root}/{image_file}' for image_file in image_files]
        if len(image_files) == 0:
            print(f"No PNG images found under: {getattr(self.cfg, 'input_dir', self.cfg.dataset_name)}")
            return -1

        image_files.sort()

        if getattr(self.cfg, "samples_per_vocoder", None):
            sampled = []
            grouped = {}
            for image_file in image_files:
                vocoder = self._infer_vocoder_name(image_file)
                grouped.setdefault(vocoder, []).append(image_file)
            for vocoder in sorted(grouped):
                sampled.extend(grouped[vocoder][: self.cfg.samples_per_vocoder])
            image_files = sampled

        if self.cfg.num_shards > 1:
            image_files = image_files[self.cfg.shard_id::self.cfg.num_shards]
        if self.cfg.limit is not None:
            image_files = image_files[: self.cfg.limit]
        for idx1, image_file in enumerate(image_files):
            self.img_size = self.cfg.img_size
            self.edge_pixel = self.cfg.edge_pixel
            img = cv2.imread(image_file)
            img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if getattr(self.cfg, "flip_vertical_input", False):
                img = np.flipud(img).copy()
            H, W, _ = img.shape

            if 'grid' in self.cfg.region_division_methods:
                div_candidates = self.cfg.grid_div_nums if self.cfg.grid_div_nums else [self.cfg.div_num]
                for div_num in div_candidates:
                    div_size = self.cfg.img_size // div_num
                    masks = []
                    for i in range(div_num):
                        for j in range(div_num):
                            mask = np.zeros((self.img_size, self.img_size), dtype=np.bool)
                            x1, x2 = j * div_size, (j + 1) * div_size
                            y1, y2 = i * div_size, (i + 1) * div_size
                            mask[y1:y2, x1:x2] = True
                            masks.append(mask)
                    method_name = 'grid' if len(div_candidates) == 1 else f'grid_n{div_num}'
                    self.sovle_masks(img, image_file, masks, method_name)
                    print(
                        f'{self.cfg.dataset_name} --> {idx1 + 1}/{len(image_files)} {image_file} for {method_name}'
                    )
            if 'superpixel' in self.cfg.region_division_methods:
                seg_candidates = self.cfg.slic_n_segments_list if self.cfg.slic_n_segments_list else [self.cfg.slic_n_segments]
                for n_segments in seg_candidates:
                    for compactness in self.cfg.slic_compactness:
                        regions = slic(
                            img,
                            n_segments=n_segments,
                            compactness=compactness,
                        )
                        masks = []
                        for label in range(regions.max() + 1):
                            mask = (regions == label)
                            masks.append(mask)
                        method_name = f"superpixel_n{n_segments}_c{compactness}"
                        self.sovle_masks(img, image_file, masks, method_name)
                        print(
                            f"{self.cfg.dataset_name} --> {idx1 + 1}/{len(image_files)} {image_file} for {method_name}"
                        )
            if 'sam' in self.cfg.region_division_methods:
                masks = self.mask_generator.generate(img)
                masks = [mask['segmentation'] for mask in masks]
                self.sovle_masks(img, image_file, masks, 'sam')
                print(f'{self.cfg.dataset_name} --> {idx1 + 1}/{len(image_files)} {image_file} for sam')

    def _infer_cls_name(self, image_file):
        if getattr(self.cfg, "input_dir", None):
            if getattr(self.cfg, "prompt_class", None):
                return self.cfg.prompt_class
            try:
                rel = os.path.relpath(image_file, self.cfg.input_dir)
            except ValueError:
                rel = image_file
            parts = rel.split(os.sep)
            if len(parts) > 1:
                return parts[0]
            return os.path.splitext(os.path.basename(image_file))[0]
        image_file_tmp = image_file
        if 'mvtec' in image_file_tmp:   # MVTec
            image_file_tmp = image_file_tmp.replace(image_file_tmp.split('/')[0], '')
        if 'visa' in image_file_tmp:  # VisA
            image_file_tmp = image_file_tmp.replace(image_file_tmp.split('/')[0], '')
        if image_file_tmp.startswith('/'):
            image_file_tmp = image_file_tmp[1:]
        cls_name = image_file_tmp.split('/')[0]
        number_list = [str(n) for n in list(range(10))]
        if cls_name[-1] in number_list:
            cls_name = cls_name[:-1]
        return cls_name

    def _output_paths(self, image_file, method):
        out_root = getattr(self.cfg, "output_dir", None) or os.path.dirname(image_file)
        method_dir = os.path.join(out_root, method)
        os.makedirs(method_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        suffix = os.path.splitext(image_file)[-1]
        return method_dir, base_name, suffix

    def _save_with_axes_image(self, rgb_img, out_path):
        # Match flip_region_outputs_with_axes.py axis style.
        h, w = rgb_img.shape[:2]
        dpi = int(getattr(self.cfg, "axis_dpi", 120))
        duration_sec = float(getattr(self.cfg, "axis_duration_sec", 4.0))
        sr = int(getattr(self.cfg, "axis_sr", 16000))
        n_mels = int(getattr(self.cfg, "axis_n_mels", 128))

        fig, ax = plt.subplots(figsize=((w + 220) / dpi, (h + 140) / dpi), dpi=dpi)
        ax.imshow(
            rgb_img,
            origin="lower",
            aspect="auto",
            extent=[0.0, duration_sec, 0, n_mels - 1],
        )

        mel_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
        y_tick_bins = np.linspace(0, n_mels - 1, 6).round().astype(int)
        y_tick_labels = [f"{mel_hz[b] / 1000.0:.1f}" for b in y_tick_bins]
        x_ticks = np.linspace(0.0, duration_sec, 6)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_tick_bins)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title("")

        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.98)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

    def _infer_vocoder_name(self, image_file):
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        sep_token = getattr(self.cfg, "vocoder_sep_token", None)
        if sep_token and sep_token in base_name:
            return base_name.split(sep_token, 1)[0]
        return base_name.split('_', 1)[0]

    def sovle_masks(self, img, image_file, masks, method):
        mask_edge = np.zeros_like(img, dtype=np.bool).astype(np.uint8) * 255
        mask_edge_number = np.zeros_like(img, dtype=np.uint8)
        img_edge = copy.deepcopy(img)
        img_edge_number = copy.deepcopy(img)

        masks = [mask for mask in masks if 600 < mask.sum() < 120000]
        for idx, mask in enumerate(masks):
            y_idx, x_idx = np.where(mask)

            center_y = y_idx.mean()
            center_x = x_idx.mean()
            distances_squared = (y_idx - center_y) ** 2 + (x_idx - center_x) ** 2
            min_index = np.argmin(distances_squared)
            xc, yc = x_idx[min_index], y_idx[min_index]

            mask1 = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
            boundaries = find_boundaries(mask1, mode='inner')
            boundaries = boundaries[1:-1, 1:-1]
            if self.edge_pixel > 1:
                boundaries = binary_dilation(boundaries, iterations=self.edge_pixel - 1)
            mask_edge[boundaries == True] = 255

            text = str(idx + 1)
            img_pil = Image.fromarray(mask_edge_number)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype('assets/arial.ttf', 10)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            x1_bg, y1_bg = xc - text_width // 2, yc - text_height // 2,
            x2_bg, y2_bg = x1_bg + text_width, y1_bg + text_height
            draw.rectangle([(x1_bg, y1_bg + 2), (x2_bg, y2_bg)], fill=(128, 128, 128))
            draw.text((x1_bg, y1_bg), text, font=font, fill=(255, 255, 255))
            mask_edge_number = np.array(img_pil)
            mask_edge_number = np.maximum(mask_edge_number, mask_edge)

            img_edge[mask_edge > 100] = mask_edge[mask_edge > 100]
            img_edge_number[mask_edge_number > 100] = mask_edge_number[mask_edge_number > 100]

        method_dir, base_name, suffix = self._output_paths(image_file, method)
        mask_path = os.path.join(method_dir, f'{base_name}_{method}_masks.pth')
        if os.path.exists(mask_path) and not getattr(self.cfg, "overwrite", False):
            return
        cv2.imwrite(os.path.join(method_dir, f'{base_name}_{method}_mask_edge.png'), cv2.cvtColor(mask_edge, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(method_dir, f'{base_name}_{method}_mask_edge_number.png'), cv2.cvtColor(mask_edge_number, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(method_dir, f'{base_name}_{method}_img_edge.png'), cv2.cvtColor(img_edge, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(method_dir, f'{base_name}_{method}_img_edge_number.png'), cv2.cvtColor(img_edge_number, cv2.COLOR_BGR2RGB))
        self._save_with_axes_image(
            img_edge_number,
            os.path.join(method_dir, f'{base_name}_{method}_img_edge_number_axes.png'),
        )
        torch.save(dict(masks=masks), mask_path)

        cls_name = self._infer_cls_name(image_file)
        prompt_cls = f"This is an image of {cls_name}."
        prompt_describe = f"The image has different region divisions, each distinguished by white edges and each with a unique numerical identifier within the region, starting from 1. Each region may exhibit anomalies of unknown types, and if any region exhibits an anomaly, the normal image is considered anomalous. Anomaly scores range from 0 to 1, with higher values indicating a higher probability of an anomaly. Please output the image anomaly score, as well as the anomaly scores for the regions with anomalies. Provide the answer in the following format: \"image anomaly score: 0.9; region 1: 0.9; region 3: 0.7.\". Ignore the region that does not contain anomalies."
        f = open(os.path.join(method_dir, f'{base_name}_prompt_wo_cls.txt'), 'w')
        f.write(f'{prompt_describe}')
        f.close()

        f = open(os.path.join(method_dir, f'{base_name}_prompt.txt'), 'w')
        f.write(f'{prompt_cls} {prompt_describe}')
        f.close()

        f = open(os.path.join(method_dir, f'{base_name}_{method}_out.txt'), 'w')
        f.write(f'')
        f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')

    # generate region divisions with labeled number
    parser.add_argument('--input-dir', type=str, default=None, help='Custom folder of PNG images.')
    parser.add_argument('--prompt-class', type=str, default=None, help='Class name to use in prompts.')
    parser.add_argument('--dataset_name', type=str, default='mvtec')
    parser.add_argument('--output-dir', type=str, default=None, help='Root output folder for method subdirs.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing outputs.')
    parser.add_argument('--shard-id', type=int, default=0, help='Shard index for parallel runs.')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of shards.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process.')
    parser.add_argument('--samples-per-vocoder', type=int, default=None, help='Sample this many images per vocoder when using --input-dir.')
    parser.add_argument('--vocoder-sep-token', type=str, default='_LA_', help='Token used to split vocoder name from filename stem.')
    # parser.add_argument('--dataset_name', type=str, default='visa')
    parser.add_argument('--region_division_methods', nargs='+', default=['superpixel'])
    # parser.add_argument('--region_division_methods', nargs='+', default=['grid', 'superpixel', 'sam'])
    parser.add_argument('--slic-n-segments', type=int, default=60, help='Target number of SLIC superpixels.')
    parser.add_argument(
        '--slic-n-segments-list',
        nargs='+',
        type=int,
        default=None,
        help='Optional list of SLIC n_segments values to sweep.',
    )
    parser.add_argument('--slic-compactness', nargs='+', type=float, default=[20.0], help='One or more SLIC compactness values.')
    parser.add_argument('--img_size', type=int, default=768)
    parser.add_argument('--div_num', type=int, default=16)
    parser.add_argument(
        '--grid-div-nums',
        nargs='+',
        type=int,
        default=None,
        help='Optional list of grid div_num values to sweep.',
    )
    parser.add_argument('--edge_pixel', type=int, default=1)
    parser.add_argument(
        '--flip-vertical-input',
        action='store_true',
        default=False,
        help='Flip input spectrograms vertically before region division.',
    )
    parser.add_argument('--axis-duration-sec', type=float, default=4.0, help='X-axis span in seconds for *_axes.png outputs.')
    parser.add_argument('--axis-sr', type=int, default=16000, help='Sample rate used for y-axis mel scaling in *_axes.png outputs.')
    parser.add_argument('--axis-n-mels', type=int, default=128, help='Mel bins used for y-axis labels in *_axes.png outputs.')
    parser.add_argument('--axis-dpi', type=int, default=120, help='DPI for *_axes.png outputs.')

    cfg = parser.parse_args()
    runner = GPT4V(cfg)
    runner.region_division()

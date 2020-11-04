import itertools
import json
import math
import os
import random
import string
import tempfile
import time
from os import path

import numpy as np
import torch
import vapoursynth as vs
from matplotlib import rcParams
from numba import njit
from PIL import Image
from vapoursynth import core

from Chars import *

if not hasattr(core, 'ffms2'):
    # core.std.LoadPlugin('C:\\Program Files M\\vapoursynth\\vapoursynth64\\plugins\\ffms2.dll')
    core.std.LoadPlugin('/usr/local/lib/libffms2.so')

core.max_cache_size = 2048 # 2G per worker

@njit(nogil=True, cache=True)
def _find_bounding_box(mask, shape):
    pos_up = -1
    pos_left = np.iinfo(np.int64).max
    pos_down = -1
    pos_right = -1
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if mask[i, j] != 0:
                if pos_up == -1:
                    pos_up = i
                else:
                    pos_down = i
                pos_left = min(pos_left, j)
                pos_right = max(pos_right, j)
    return pos_left, pos_up, pos_right, pos_down


class VideoDataset(torch.utils.data.Dataset):
    '''
    Load video and subtitle as a pytorch dataset
    '''

    def __init__(self, video, ass=None, fonts=None):
        super().__init__()
        clip = core.ffms2.Source(source=video)
        clip = core.resize.Bicubic(clip, format=vs.RGB24, matrix_in_s="709")
        if ass is not None:
            sub, mask = core.sub.TextFile(clip, ass, blend=False, fontdir=fonts)
            self.clip = core.std.MaskedMerge(clip, sub, mask)
            del sub
            self.mask = mask
        else:
            self.clip = clip

    def _clipToTensor(self, clip):
        frame = clip.get_frame(0)
        img = np.zeros((3, frame.height, frame.width), dtype='float32')

        for i, plane in enumerate(frame.planes()):
            img[i, :, :] = plane
        img = torch.as_tensor(img).true_divide_(255)

        del frame
        return img

    def __len__(self):
        return len(self.clip)

    def __getitem__(self, index):
        return self._clipToTensor(self.clip[index])


class VideoDatasetTorchScript(VideoDataset):
    '''
    Load video and subtitle as a pytorch dataset
    '''

    def _clipToTensor(self, clip):
        frame = clip.get_frame(0)
        img = np.zeros((frame.height, frame.width, 3), dtype='float32')

        for i, plane in enumerate(frame.planes()):
            img[:, :, i] = plane
        img = torch.as_tensor(img)

        del frame
        return img


class SubtitleDataset(torch.utils.data.IterableDataset):
    def __init__(self, styles_json=None, samples=None, fonts=None, start_frame=0, end_frame=None, chars=BasicChars(), texts=None):
        super().__init__()
        self.styles_json = path.join('data', 'styles', 'styles.json') if styles_json is None else styles_json
        self.samples = path.join('data', 'samples') if samples is None else samples
        self.fonts = path.join('data', 'fonts') if fonts is None else fonts
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.chars = chars
        self.texts = texts

    def __iter__(self):
        return SubtitleDatasetIterator(self)


class SubtitleDatasetIterator():
    def __init__(self, dataset: SubtitleDataset):
        super().__init__()
        self._init(dataset)

    def _init(self, dataset: SubtitleDataset):
        with open(dataset.styles_json, 'r', encoding='utf-8') as f:
            self.styles = json.load(f)

        sample = path.join(dataset.samples, random.choice([f for f in os.listdir(dataset.samples) if f.endswith('.mkv') or f.endswith('.mp4')]))

        clip = core.ffms2.Source(source=sample)
        clip = core.resize.Bicubic(clip, format=vs.RGB24, matrix_in_s="709")

        self.clip = clip
        self.end_frame = dataset.end_frame
        self.index = dataset.start_frame
        self.fonts = dataset.fonts
        self.chars = dataset.chars
        self.texts = dataset.texts
        self.dataset = dataset
        self._generateText()
        self._generateSub()

    def _clipToTensor(self, clip):
        frame = clip.get_frame(0)
        img = np.zeros((3, frame.height, frame.width), dtype='float32')

        for i, plane in enumerate(frame.planes()):
            img[i, :, :] = plane
        img = torch.as_tensor(img).true_divide_(255)

        del frame
        return img

    def _frameToTime(self, frame, fps, intf=math.floor):
        time_int = intf(frame * 100 / fps)
        time_struct = time.gmtime(time_int // 100)
        return f'{time_struct.tm_hour}:{time_struct.tm_min:02d}:{time_struct.tm_sec:02d}.{time_int % 100:02}'

    def _generateText(self):
        def genRandomText():
            text = random.sample(self.chars.chars[1:], random.randint(3, 22))
            if random.random() < 0.2:
                text = text[:-8] if len(text) > 10 else text
                text.insert(random.randrange(0, len(text)), ''.join(random.sample(string.ascii_letters, random.randint(3, 7))))
            if random.random() < 0.1:
                text = text[:-5] if len(text) > 10 else text
                start = random.randrange(0, len(text))
                text.insert(start, "『")
                text.insert(random.randrange(start, len(text)), "』")
            if random.random() < 0.1:
                text = text[:-5] if len(text) > 10 else text
                start = random.randrange(0, len(text))
                text.insert(start, "「")
                text.insert(random.randrange(start, len(text)), "」")
            return ''.join(text)

        if self.texts == None:
            self.texts = [genRandomText() for _ in range(self.clip.num_frames)]
        else:
            self.texts = [genRandomText() if random.random() < 0.5 else random.choice(self.texts) for _ in range(self.clip.num_frames)]

    def _generateSub(self):
        ass_file_text = "[Script Info]\n" + \
            "ScriptType: v4.00+\n" + \
            f"PlayResX: {self.clip.width}\n" + \
            f"PlayResY: {self.clip.height}\n" + \
            "[V4+ Styles]\n" + \
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"

        for i, style in enumerate(itertools.product(self.styles['fonts'], self.styles['styles'])):
            ass_file_text += f"Style: Default{i},{style[0]}{style[1]}\n"

        ass_file_text += "[Events]\n" + \
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

        for i in range(self.clip.num_frames):
            start_time = self._frameToTime(i, fps=self.clip.fps, intf=math.floor)
            end_time = self._frameToTime(i + 1, fps=self.clip.fps, intf=math.floor)
            ass_file_text += f"Dialogue: 0,{start_time},{end_time},Default{random.randrange(0, len(self.styles['fonts']) * len(self.styles['styles']))},,0,0,0,,{self.texts[i]}\n"

        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as f:
            f.write(ass_file_text)
            tmp_file_name = f.name

        sub, mask = core.sub.TextFile(self.clip, tmp_file_name, blend=False, fontdir=self.fonts)
        os.remove(tmp_file_name)

        merged_clip = core.std.MaskedMerge(self.clip, sub, mask)
        del sub
        self.merged_clip = merged_clip
        self.mask = mask

    def __next__(self):
        if self.end_frame is not None and self.index >= self.end_frame:
            raise StopIteration()

        if self.index >= self.clip.num_frames:
            del self.merged_clip
            self.index = self.dataset.start_frame
            self._generateSub()

        index = self.index
        self.index += 1

        mask_frame = self.mask.get_frame(index)
        mask_img = mask_frame.get_read_array(0)
        del mask_frame
        bounding_box = _find_bounding_box(mask_img, mask_img.shape)

        return self.merged_clip[index], bounding_box, mask_img.shape, self.texts[index]


class SubtitleDatasetRCNN(SubtitleDataset):
    def __iter__(self):
        return SubtitleDatasetIteratorRCNN(self)


class SubtitleDatasetIteratorRCNN(SubtitleDatasetIterator):
    def __next__(self):
        clip, bounding_box, shape, _ = super().__next__()
        img_height, img_width = shape
        crop_pos = (
            random.randint(0, bounding_box[0]),  # left
            random.randint(img_height // 1.5, bounding_box[1]),  # top
            random.randint(0, img_width - bounding_box[2]),  # right
            random.randint(0, img_height - bounding_box[3])  # bottom
        )

        clip = core.std.Crop(clip, left=crop_pos[0], top=crop_pos[1], right=crop_pos[2], bottom=crop_pos[3])
        img = self._clipToTensor(clip)

        bounding_box = torch.tensor([[
            bounding_box[0] - crop_pos[0],
            bounding_box[1] - crop_pos[1],
            bounding_box[2] - crop_pos[0],
            bounding_box[3] - crop_pos[1]
        ]], dtype=torch.int64)

        return img, bounding_box


class SubtitleDatasetOCR(SubtitleDataset):
    def __init__(self, styles_json=None, samples=None, fonts=None, start_frame=0, end_frame=None, chars=BasicChars(), texts=None, grayscale=0.5):
        super().__init__(styles_json=styles_json, samples=samples, fonts=fonts, start_frame=start_frame, end_frame=end_frame, chars=chars, texts=texts)
        self.grayscale = grayscale

    def __iter__(self):
        return SubtitleDatasetIteratorOCR(self)


class SubtitleDatasetIteratorOCR(SubtitleDatasetIterator):
    def _rgb_to_grayscale(self, img: torch.Tensor):
        img_gray = torch.zeros_like(img)
        img_gray[0] = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
        img_gray[1] = img_gray[0]
        img_gray[2] = img_gray[0]
        return img_gray

    def __next__(self):
        clip, bounding_box, shape, text = super().__next__()
        img_height, img_width = shape

        crop_pos = (
            bounding_box[0] - random.randint(2, 10),  # left
            bounding_box[1] - random.randint(2, 10),  # top
            img_width - (bounding_box[2] + random.randint(2, 10)),  # right
            img_height - (bounding_box[3] + random.randint(2, 4))  # bottom
        )

        if crop_pos[0] + crop_pos[2] >= img_width or crop_pos[1] + crop_pos[3] >= img_height:
            return self.__next__()

        clip = core.std.Crop(clip, left=crop_pos[0], top=crop_pos[1], right=crop_pos[2], bottom=crop_pos[3])
        clip = core.resize.Bicubic(clip, width=clip.width // 2, height=clip.height // 2)

        img = self._clipToTensor(clip)
        if random.random() < self.dataset.grayscale:
            img = self._rgb_to_grayscale(img)

        encoded_text = torch.tensor([self.chars.chars.index(char) for char in text], dtype=torch.long)

        return img, encoded_text


class SubtitleDatasetOCREval(SubtitleDatasetOCR):
    def __init__(self, styles_json=None, samples=None, fonts=None, start_frame=0, end_frame=None, chars=BasicChars(), texts=None, grayscale=0.5):
        super().__init__(styles_json=styles_json, samples=samples, fonts=fonts, start_frame=start_frame, end_frame=end_frame, chars=chars, texts=texts, grayscale=grayscale)

    def __iter__(self):
        return SubtitleDatasetIteratorOCREval(self)


class SubtitleDatasetIteratorOCREval(SubtitleDatasetIteratorOCR):
    def _generateText(self):
        self.texts = []
        for i in range(11, len(self.chars.chars) // 10):
            self.texts.append(self.chars.chars[i * 10: i * 10 + 10])
        self.texts.append(self.chars.chars[len(self.chars.chars) // 10 * 10: len(self.chars.chars) // 10 * 10 + len(self.chars.chars) % 10])
        self.texts = [self.texts[i % len(self.texts)] for i in range(self.clip.num_frames)]

    def __next__(self):
        return super().__next__()

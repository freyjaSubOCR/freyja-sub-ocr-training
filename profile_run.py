from Chars import *
from SubtitleDataset import SubtitleDatasetOCR
import torchvision.transforms.functional as f

if __name__ == "__main__":
    chars = CJKChars()
    train_dataset = SubtitleDatasetOCR(chars=chars, )

    for i, (img, encoded_text) in enumerate(train_dataset):
        if i > 43200:
            break
        if (i > 60 and i < 100) or i % 1000 == 1:
            text = ''.join([chars.chars[i] for i in encoded_text])
            f.to_pil_image(img).save(f'data/samples2/{text}.jpg')

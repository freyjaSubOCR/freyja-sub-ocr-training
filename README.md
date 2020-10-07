# Freyja subtitle OCR trainer

PyTorch based deep learning model trainer designed for training the models used by Freyja subtitle OCR.

## Install

Having a CUDA capable GPU is strongly recommended.

### Docker

It is recommended to use the docker environment for training.

To use the docker image, just follow instructions on <https://github.com/freyjaSubOCR/freyja-sub-ocr-docker>.

### Windows

1. If you have a GPU, first install [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork) and [cuDNN 7](https://developer.nvidia.com/cudnn).

2. Install [Python3](https://www.python.org/).

3. Install [vapoursynth](http://www.vapoursynth.com/), and put [ffms2](https://github.com/FFMS/ffms2/releases) in vapoursynth's plugin folder.

4. Run following commands

    ```bash
    pip install -r requirements.txt
    pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
    ```

5. Clone the repo to a local directory with enough space (>40GB)

### MacOS

Please use the Docker Image <https://github.com/freyjaSubOCR/freyja-sub-ocr-docker>.

### Linux

1. If you have a GPU, first install [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64) and [cuDNN 7](https://developer.nvidia.com/cudnn).

2. Run following commands (Use fedora as example)

    ```bash
    dnf -y install git python3 python3-Cython ffmpeg libass zimg && \
        rm -rf /var/cache/dnf/*

    dnf -y install python3-devel ffmpeg-devel libass-devel zimg-devel autoconf automake libtool gcc gcc-c++ yasm make && \
        rm -rf /var/cache/dnf/* && \
        git clone https://github.com/vapoursynth/vapoursynth.git && \
        cd vapoursynth && \
        ./autogen.sh && \
        ./configure && \
        make && \
        make install && \
        cd .. && \
        rm -rf vapoursynth

    git clone https://github.com/FFMS/ffms2 && \
        cd ffms2 && \
        ./autogen.sh && \
        make && \
        make install && \
        cd / && \
        rm -rf ffms2

    export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

    pip3 install -r requirements.txt
    pip3 install torch torchvision
    ```

3. Clone the repo to a local directory with enough space (>40GB)

## Data preparation

Real world ASS files can be provided to enhance OCR training quality. These files should be placed in ```data/ass``` folder.

Real world movie samples should be placed in ```data/samples``` and ```data/samples_eval```. Only mp4 and mkv files are supported. These movie files should not contain embedded subtitles. You should put more files in ```data/samples``` folder. One or two movie files in ```data/samples_eval``` are enough for evaluation.

You can change training fonts and styles in the json files in ```data/styles``` and ```data/styles_eval```. These styles are in ASS styling format. The fonts stated in the json files in ```data/styles``` and ```data/styles_eval``` folders should have corresponding font files being placed in ```data/fonts``` folder.

## Training

### Faster RCNN

Faster RCNN is a deep learning model that can detect the bounding boxes of objects. In Freyja, it is mainly used to detect the bounding boxes of subtitles.

To train the model, run ```FasterRCNNTraining.py```.

Parameters you can change:

```python
    train_dataset = SubtitleDatasetRCNN(chars=SC3500Chars())
    test_dataset = SubtitleDatasetRCNN(chars=SC3500Chars(), start_frame=500, end_frame=500 + 64)
```

You can change ```SC3500Chars()``` to other character sets like ```TC3600Chars()``` or ```CJKChars()```

```python
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=8, collate_fn=RCNN_collate_fn, timeout=60)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=RCNN_collate_fn)
```

You can change the number of ```num_workers``` to reduce memory usage / increase data loading speed

You can change the number of ```batch_size``` to reduce GPU memory usage / increase training speed

### OCR

The OCR model uses Resnet CNN + GRU RNN to recognize subtitles.

To train the model, run ```OCRTraining.py```.

Parameters you can change:

```python
    chars = TC3600Chars()
```

You can change ```TC3600Chars()``` to other character sets like ```SC3500Chars()``` or ```CJKChars()```

```python
    train_dataset = SubtitleDatasetOCR(chars=chars, styles_json=path.join('data', 'styles', 'styles_yuan.json'), texts=texts)
    test_dataset = SubtitleDatasetOCR(chars=chars, start_frame=500, end_frame=500 + 16, grayscale=1,
                                      styles_json=path.join('data', 'styles', 'styles_yuan.json'), texts=texts)
    eval_dataset = SubtitleDatasetOCR(styles_json=path.join('data', 'styles_eval', 'styles_yuan.json'),
                                      samples=path.join('data', 'samples_eval'),
                                      chars=chars, start_frame=500, end_frame=500 + 16, grayscale=1, texts=texts)
```

You can change ```styles_yuan.json``` to other style files like ```styles_hei.json```

```python
    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=OCR_collate_fn, num_workers=16, timeout=60)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=OCR_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=OCR_collate_fn)
```

You can change the number of ```num_workers``` to reduce memory usage / increase data loading speed

You can change the number of ```batch_size``` to reduce GPU memory usage / increase training speed

```python
    model = CRNNResnext101(len(chars.chars), rnn_hidden=1280)
```

You can change ```CRNNResnext101``` to other CNN models like ```CRNNResnext50```. Large character sets need larger models.

You can change the number of ```rnn_hidden```. Large character sets need more RNN hidden states.

```python
    train(model, 'CRNNResnext101_1280', train_dataloader, test_dataloader, eval_dataloader, chars.chars, 'ocr_TC3600Chars_yuan',
          backbone_url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth')
```

You can change string ```CRNNResnext101_1280``` and ```ocr_TC3600Chars_yuan```. These are plain strings that help you identify trained models.

You needs to change the backbone url according to the CNN model you choose. Pre-trained backbone urls can be find on <https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>.

## Export model

After the training process, you can find the models in ```models``` folder. These files only contain the parameters of the models.

To export the model that can be used by Freyja, run the ```export_model.py```. You may need to change the path string in the ```export_model.py```.

```python
device = torch.device('cuda')
```

This line controls the export target of the model. If you want the models to run on CPU, use ```torch.device('cpu')```, or If you want the models to run on GPU, use ```torch.device('cuda')```.

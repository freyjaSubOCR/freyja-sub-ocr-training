import os
import re
from os import path

from Chars import *


class ASSReader():
    def __init__(self, regex=r'.', asses="data/ass"):
        super().__init__()
        self.texts = []
        for ass in os.listdir(asses):
            if re.search(regex, ass) is not None:
                with open(path.join(asses, ass), 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.replace('{\\be1}', '')
                        line = line.replace('\\N', ' ')
                        match = re.search(r'Dialogue: \d+,(?P<time>\d:\d{2}:\d{2}.\d{2}),([^,]*?,){7}(?P<text>[^—{\-,]+)\n', line)
                        if match is not None:
                            time = match.groupdict()['time']
                            text = match.groupdict()['text']
                            frame = int((int(time[2:4]) * 60 + int(time[5:7]) + int(time[-2:]) * 0.01) * (23.976))
                            if frame == 0:
                                continue
                            self.texts.append(text)

    def getCompatible(self, chars):
        for text in self.texts:
            compatible = True
            for char in text:
                if char not in chars.chars:
                    compatible = False
                    break
            if compatible:
                yield text

    def getIncompatible(self, chars, verbose=False):
        for text in self.texts:
            for char in text:
                if char not in chars.chars:
                    yield text
                    if verbose:
                        print(f'Find \'{char}\' in \'{text}\' which is not compatible with the {type(chars).__name__}')
                    break


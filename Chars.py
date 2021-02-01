import random
import string


class BasicChars():
    '''
    Chars containing letters, digits and punctuation
    '''

    def __init__(self):
        super().__init__()
        self.blank = '\x00'
        self.chars = string.ascii_letters + string.digits
        # with open('data/chars/kanaChars.txt', 'r', encoding='utf-8') as f:
        #     self.chars = self.chars + f.read()
        with open('data/chars/punctuationChars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.blank + self.chars + f.read()

    def export(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.chars)


class LatinChars(BasicChars):
    '''
    Contains all common Latin-script alphabets. Source: https://en.wikipedia.org/wiki/List_of_Latin-script_alphabets
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/latinChars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()

    def generateRandomText(self, max_length=20):
        text = ' '.join([''.join(random.sample(self.chars[1:], random.randint(2, 10))) for _ in range(random.randint(1, 10))])[:max_length]
        text = list(text)
        if random.random() < 0.1:
            text = text[:-5] if len(text) > 10 else text
            start = random.randrange(0, len(text))
            text.insert(start, "\"")
            text.insert(random.randrange(start, len(text)), "\"")
        if random.random() < 0.1:
            text = text[:-5] if len(text) > 10 else text
            start = random.randrange(0, len(text))
            text.insert(start, "\'")
            text.insert(random.randrange(start, len(text)), "\'")
        return ''.join(text)


class ChineseChars(BasicChars):
    '''
    Base class for all Chinese chars. Add Chinese punctuation chars.
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/chinesePunctuationChars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()

    def generateRandomText(self, max_length=22):
        text = random.sample(self.chars[1:], random.randint(3, max_length))
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


class TC3600Chars(ChineseChars):
    '''
    Chars containing BaseChars and frequently used 3500 simplified chinese chars
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/TC3600Chars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()


class SC3500Chars(ChineseChars):
    '''
    Chars containing BaseChars and frequently used 3500 simplified chinese chars
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/SC3500Chars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()


class SC5000Chars(ChineseChars):
    '''
    Chars containing BaseChars and frequently used 5000 simplified chinese chars
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/SC5000Chars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()


class TC5000Chars(ChineseChars):
    '''
    Chars containing BaseChars and frequently used 5000 traditional chinese chars
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/TC5000Chars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()


class SC7000Chars(ChineseChars):
    '''
    Chars containing BaseChars and frequently used 7000 simplified chinese chars
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/SC7000Chars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()


class TinyCJKChars(ChineseChars):
    '''
    Chars containing BaseChars and frequently used 8000 CJK chars
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/TinyCJKChars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()


class CJKChars(ChineseChars):
    '''
    Chars containing BaseChars and Unicode CJK Unified Ideographs (No extensions)
    '''

    def __init__(self):
        super().__init__()
        self.chars = self.chars + ''.join([chr(c) for c in range(0x4e00, 0x9fa6)])

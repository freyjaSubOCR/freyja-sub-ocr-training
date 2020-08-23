import string


class BasicChars():
    '''
    Chars containing letters, digits, kana and punctuation
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


class SC3500Chars(BasicChars):
    '''
    Chars containing BaseChars and frequently used 3500 simplified chinese chars
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/SC3500Chars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()


class SC7000Chars(BasicChars):
    '''
    Chars containing BaseChars and frequently used 7000 simplified chinese chars
    '''

    def __init__(self):
        super().__init__()
        with open('data/chars/SC7000Chars.txt', 'r', encoding='utf-8') as f:
            self.chars = self.chars + f.read()


class CJKChars(BasicChars):
    '''
    Chars containing BaseChars and Unicode CJK Unified Ideographs (No extensions)
    '''

    def __init__(self):
        super().__init__()
        self.chars = self.chars + ''.join([chr(c) for c in range(0x4e00, 0x9fa6)])

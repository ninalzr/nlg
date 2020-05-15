# TODO: Adjust the class for other tokenizers
class Lookup():
    def __init__(self, model_class, file_prefix=None):

        self.model_class = model_class

        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.sep_token = None
        self.pad_token = None
        self.cls_token = None
        self.mask_token = None

        if model_class == 'gpt2':
            from transformers import GPT2Tokenizer
            self._tokenizer = GPT2Tokenizer.from_pretrained(model_class)

        if model_class == 'bert':
            from transformers import BertTokenizer
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self._tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        if self._tokenizer._bos_token:
            self.bos_token = self._tokenizer.bos_token
        if self._tokenizer._eos_token:
            self.eos_token = self._tokenizer.eos_token
        if self._tokenizer._unk_token:
            self.unk_token = self._tokenizer.unk_token
        if self._tokenizer._sep_token:
            self.sep_token = self._tokenizer.sep_token
        if self._tokenizer._pad_token:
            self.pad_token = self._tokenizer.pad_token
        if self._tokenizer._cls_token:
            self.cls_token = self._tokenizer.cls_token
        if self._tokenizer._mask_token:
            self.mask_token = self._tokenizer.mask_token

        if file_prefix:
            self.load(file_prefix)

        def save_special_tokens(self, file_prefix):
            if self.model_class == "gpt2" or self.model_class == 'bert':
                special_tokens = {}
            if self.bos_token:
                special_tokens['bos_token'] = self.bos_token
            if self.eos_token:
                special_tokens['eos_token'] = self.eos_token
            if self.unk_token:
                special_tokens['unk_token'] = self.unk_token
            if self.sep_token:
                special_tokens['sep_token'] = self.sep_token
            if self.pad_token:
                special_tokens['pad_token'] = self.pad_token
            if self.cls_token:
                special_tokens['cls_token'] = self.cls_token
            if self.mask_token:
                special_tokens['mask_token'] = self.mask_token
            json.dump(special_tokens, open(file_prefix + ".special_tokens", "w", encoding="utf8"), indent=4,
                      sort_keys=True)
            self._tokenizer.add_special_tokens(special_tokens)

        def load(self, file_prefix):
            if os.path.exists(file_prefix + ".special_tokens"):
                special_tokens = json.load(open(file_prefix + ".special_tokens", "r", encoding="utf8"))
            if 'bos_token' in special_tokens:
                self.bos_token = special_tokens['bos_token']
            if 'eos_token' in special_tokens:
                self.eos_token = special_tokens['eos_token']
            if 'unk_token' in special_tokens:
                self.unk_token = special_tokens['unk_token']
            if 'sep_token' in special_tokens:
                self.sep_token = special_tokens['sep_token']
            if 'pad_token' in special_tokens:
                self.pad_token = special_tokens['pad_token']
            if 'cls_token' in special_tokens:
                self.cls_token = special_tokens['cls_token']
            if 'mask_token' in special_tokens:
                self.mask_token = special_tokens['mask_token']
            self._tokenizer.add_special_tokens(special_tokens)

    def tokenize(self, text):
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, token_ids):
        return self._tokenizer.convert_ids_to_tokens(token_ids)

    def convert_tokens_to_string(self, tokens):
        return self._tokenizer.convert_tokens_to_string(tokens)

    def encode(self, text, add_bos_eos_tokens=False):
        tokens = self.tokenize(text)

        if add_bos_eos_tokens:
            if self.model_class == 'bert':
                if not self.cls_token or not self.sep_token:
                    raise Exception("Lookup encode error: {} model does not have CLS or SEP tokens set!")
                return [self.convert_tokens_to_ids(self.cls_token)] + self.convert_tokens_to_ids(tokens) + [
                    self.convert_tokens_to_ids(self.sep_token)]
            else:
                if not self.bos_token or not self.eos_token:
                    raise Exception("Lookup encode error: {} model does not have BOS or EOS tokens set!")
                return [self.convert_tokens_to_ids(self.bos_token)] + self.convert_tokens_to_ids(tokens) + [
                    self.convert_tokens_to_ids(self.eos_token)]
        else:
            return self.convert_tokens_to_ids(tokens)

    def decode(self, token_ids, skip_bos_eos_tokens=False):
        if skip_bos_eos_tokens:
            if self.model_class == "bert":
                if len(token_ids) > 0:
                    if token_ids[0] == self.convert_tokens_to_ids(self.cls_token):
                        token_ids = token_ids[1:]
                if len(token_ids) > 0:
                    if token_ids[-1] == self.convert_tokens_to_ids(self.sep_token):
                        token_ids = token_ids[:-1]
            else:
                if not self.bos_token or not self.eos_token:
                    raise Exception("Lookup decode error: {} model does not have BOS or EOS tokens set!")
                if len(token_ids) > 0:
                    if token_ids[0] == self.convert_tokens_to_ids(self.bos_token):
                        token_ids = token_ids[1:]
                if len(token_ids) > 0:
                    if token_ids[-1] == self.convert_tokens_to_ids(self.eos_token):
                        token_ids = token_ids[:-1]
        if len(token_ids) > 0:
            tokens = self.convert_ids_to_tokens(token_ids)
            return self.convert_tokens_to_string(tokens)
        return ""

    def __len__(self):
        return len(self._tokenizer)
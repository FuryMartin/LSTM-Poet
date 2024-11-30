import json
import torch

class CharacterTokenizer:
	def __init__(self, char2id):
		self.char2id = char2id
		self.id2char = {idx: char for char, idx in char2id.items()}

		self.vocab_size = len(char2id)

		self.pad_token_id = self.char2id["<PAD>"]
		self.unk_token_id = self.char2id["<UNK>"]
		self.bos_token_id = self.char2id["<BOS>"]
		self.eos_token_id = self.char2id["<EOS>"]

	def __call__(self, texts, **kwargs):
		return self.batch_encode(texts, **kwargs)

	def encode(self, text, max_length=None, add_special_tokens=True, padding=True, return_tensors="pt"):
		"""Convert text to token IDs, with optional padding/truncation."""
		token_ids = []
		if add_special_tokens:
			token_ids.append(self.bos_token_id)
			
		token_ids.extend([self.char2id.get(char, self.unk_token_id) for char in text])
		
		if add_special_tokens:
			token_ids.append(self.eos_token_id)

		if max_length is not None:
			if len(token_ids) > max_length:
				token_ids = token_ids[:max_length]
			else:
				if padding:
					token_ids = [self.pad_token_id] * (max_length - len(token_ids)) + token_ids

		if return_tensors == "pt":
			token_ids = torch.tensor(token_ids, dtype=torch.long)

		return token_ids
	
	def batch_encode(self, texts, **kwargs):
		input_ids = [self.encode(text,  max_length=kwargs.get("max_length", None), add_special_tokens=kwargs.get("add_special_tokens", True), padding=kwargs.get("padding", True), return_tensors=None) for text in texts]

		if kwargs.get("return_tensors") == "pt":
			return torch.tensor(input_ids, dtype=torch.long)

		return input_ids


	def decode(self, token_ids, skip_special_tokens=True):
		"""Convert token IDs back to text."""

		if isinstance(token_ids, torch.Tensor):
			token_ids = token_ids.tolist()

		text = []
		for id_ in token_ids:
			if skip_special_tokens and id_ in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
				continue
			char = self.id2char.get(id_, self.unk_token_id)
			text.append(char)
			
		return "".join(text)
	
	def batch_decode(self, batch_token_ids, **kwargs):
		"""Convert batch of token IDs back to text."""
		return [self.decode(token_ids, **kwargs) for token_ids in batch_token_ids]

	@classmethod
	def from_pretrained(cls, path):
		"""Load pre-trained tokenizer from file."""
		with open(path, "r") as file:
			char2id = json.load(file)

		return cls(char2id=char2id)

	def save_pretrained(self, path):
		"""Save tokenizer to file."""
		with open(path, "w") as file:
			json.dump(self.char2id, file, ensure_ascii=False)

	@classmethod
	def from_vocab_list(cls, vocab_list):
		"""Create tokenizer from vocabulary."""
		char2id = {char: idx for idx, char in enumerate(vocab_list)}
		return cls(char2id=char2id)
	
if __name__ == '__main__':
	from datasets import load_dataset

	dataset = load_dataset("FuryMartin/poetry_dataset")

	vocab = sorted(list({char for item in dataset["train"] for char in item["text"]})) + ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]

	tokenizer = CharacterTokenizer.from_vocab_list(vocab)

	tokenizer.save_pretrained("vocab.json")
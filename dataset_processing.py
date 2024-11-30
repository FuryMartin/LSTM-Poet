import os
import json
import re
import numpy as np
from opencc import OpenCC
from tokenizer import CharacterTokenizer
from datasets import load_dataset, concatenate_datasets, load_from_disk

def _parseRawData(author_limit=None, length_limit=None, dataset_path="data/", class_limit="poet.tang", to_simplified=False):
	converter = OpenCC('t2s')

	def sentenceParse(para):
		result = re.sub(r"（.*）", "", para)
		result = re.sub(r"{.*}", "", result)
		result = re.sub(r"《.*》", "", result)
		result = re.sub(r"[\]\[]", "", result)
		final_result = ""
		for char_word in result:
			if char_word not in set("0123456789-"): final_result += char_word
		final_result = re.sub(u"。。", u"。", final_result)
		return final_result

	def handleJson(filein_path):
		final_result = []
		if not os.path.exists(filein_path):
			raise ValueError("error! not found the filein path: {}".format(filein_path))
		data = json.loads(open(filein_path, "r", encoding="utf-8").read())
		for poetry_contains in data:
			poetry_data = ""
			if author_limit is not None and poetry_contains.get("author") != author_limit:
				continue
			poetry = poetry_contains.get("paragraphs")
			flag = False
			for sentence in poetry:
				sample_sentences = re.split(u"[，！。]", sentence)
				for sample_sentence in sample_sentences:
					# 几言诗
					if length_limit is not None and len(sample_sentence) != length_limit and len(sample_sentence) != 0:
						flag = True
						break
				if flag: break
			if flag: continue
			for sentence in poetry:
				poetry_data += sentence
			poetry_data = sentenceParse(poetry_data)

			author = poetry_contains.get("author")
			title = poetry_contains.get("title")
			if length_limit == 5:
				category = "五言诗"
			elif length_limit == 7:
				category = "七言诗"
			else:
				category = "其他"
			
			item = {
				"title": title if not to_simplified else converter.convert(title),
				"author": author if not to_simplified else converter.convert(author),
				"text": poetry_data if not to_simplified else converter.convert(poetry_data),
				"id": poetry_contains.get("id"),
				"category": category
			}
			if poetry_data != "": final_result.append(item)

		return final_result

	final_data = list()
	print("loading source data file...")
	for filein_name in os.listdir(dataset_path):
		if filein_name.startswith(class_limit):
			final_data.extend(handleJson(dataset_path + filein_name))
			print("[ loading file: {} ]  OK!".format(filein_name))
		else:
			continue
	return final_data

if __name__ == '__main__':
	dataset_path = "data/全唐诗/"
	author_limit = None
	class_limit = "poet.tang"

	five_word_poetry = _parseRawData(
		dataset_path=dataset_path,
		author_limit=author_limit,
		length_limit=5,
		class_limit=class_limit,
		to_simplified=False
	)

	# save training data in jsonl
	with open("data/five_word.jsonl", "w") as f:
		for i, line in enumerate(five_word_poetry):
			f.write(json.dumps(line, ensure_ascii=False) + "\n")

	seven_word_poetry = _parseRawData(
		dataset_path=dataset_path,
		author_limit=author_limit,
		length_limit=7,
		class_limit=class_limit,
		to_simplified=False
	)

	# save training data in jsonl
	with open("data/seven_word.jsonl", "w") as f:
		for i, line in enumerate(seven_word_poetry):
			f.write(json.dumps(line, ensure_ascii=False) + "\n")

	five = load_dataset("json", data_files="data/five_word.jsonl")
	seven = load_dataset("json", data_files="data/seven_word.jsonl")

	dataset = concatenate_datasets([five["train"], seven["train"]]).save_to_disk("data/poetry_dataset")

	# dataset.push_to_hub("FuryMartin/poetry_dataset")
'''

tokenize PG19, store input_ids

'''

import os
import pdb
import torch
import random
import pickle
import torch
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
random.seed(42)
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input-path", type=str, default=None)
	parser.add_argument("--output-path", type=str, default=None)
	parser.add_argument("--shard-id", type=int, default=None)
	parser.add_argument("--shard-size", type=int, default=None)
	args = parser.parse_args()
	return args

def tokenize(in_path, book_name, chunk_size=256):
	'''
		store a list of tokenized sentence for each book
	'''
	with open(os.path.join(in_path, book_name), "r") as f:
		data = ' '.join([l.strip() for l in f.readlines()])

	sent_tok_data = sent_tokenize(data)

	this_book_sents = []
	for sent in tqdm(sent_tok_data):
		sent_ids = tokenizer(sent)['input_ids'][1:-1] # get rid of the <eos>, added later when adding to this_book_ids
		this_book_sents.append(sent_ids)
	return this_book_sents 


def main():
	args = parse_args()
	books = sorted(os.listdir(args.input_path))

	shard_id = args.shard_id
	shard_size = args.shard_size
	this_books = books[shard_size*shard_id:shard_size*(shard_id+1)]
	encoded_all = {}
	for book in this_books:
		book_id = book.strip(".txt")
		input_ids = tokenize(args.input_path, book)
		encoded_all[book_id] = {'input_ids': input_ids}

	with open(os.path.join(args.output_path, f"{shard_id:04}.pkl"), 'wb') as f:
		pickle.dump(encoded_all, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	main()



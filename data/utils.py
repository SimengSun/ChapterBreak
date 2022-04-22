
import os
import pdb
import pickle
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
logger = logging.getLogger(__name__)
random.seed(42)

from transformers import RobertaTokenizer, GPT2Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

class SLMEvalDataset(Dataset):
	"""
		Dataset for ChapterBreak, need to preprocess 
		the chapterbreak data with `tokenize_eval_data.py` first
	"""
	def __init__(self, args, data_path, batch_size):

		with open(data_path, "rb") as f:
			data = pickle.load(f)

		assert batch_size == 1, "Need to have one example per evaluation batch"
		self.args = args
		self.data = [example for book_id in data for example in data[book_id]]
		
	def __len__(self):
		return len(self.data)

	def _get_cls_item(self, idx):

		this_item = self.data[idx]

		ctx_vec = this_item['ctx_vec']
		pos_vec = this_item['pos_vec']

		# tokenized context could be longer, need to cut it down
		# to the maximize chunk length allowed by the trained suffixLM
		ctx_vec = ctx_vec[-(self.args.max_tokens_per_batch // self.args.chunk_size_list[0]-1):]

		negs_vec = this_item['negs_vec']

		pos_data = torch.cat([ctx_vec, pos_vec], dim=0)
		negs_data = [torch.cat([ctx_vec, neg_vec]) for neg_vec in negs_vec]

		batch_data = torch.stack([pos_data] + negs_data)
		padding_mask = torch.zeros(batch_data.shape[0], batch_data.shape[1]).float()
		
		return {
			'data': batch_data,
			'padding_mask': padding_mask
		}

	def __getitem__(self, idx):
		
		return self._get_cls_item(idx)

		
class TextDataset(Dataset):
	"""
		Dataset for loading data for training SuffixLM
	"""
	def __init__(self, args, data_path, max_num_chunks, split, max_books=500, chunk_size=256, fileids=None):
		'''
			load tokenized and binarized training data data,
			divide to chunks of max_tokens_per_batch

			chunk_size: chunk_size can be different for each dataset
			num_chunks: the seq_len for segment-level LM, depends on
								max_token_per_batch and chunk_size
		'''
		self.max_books = max_books
		self.max_num_chunks = max_num_chunks
		self.args = args
		self.chunk_size_list = args.chunk_size_list
		self.max_tokens_per_batch = args.max_tokens_per_batch

		# load roberta tokenized input ids
		book_data, book_fid = self.load_tokenized(os.path.join(data_path, split), chunk_size, fileids) # {'book_id': [#segments, chunk_size]} {'book_id': 'file_id'}
		print(f'number of books {len(book_data)}')

		self.data = self.group_segment(book_data, max_num_chunks) # [ [<=max_num_chunks, chunk_size],  ]
		print(f'number of examples {len(self.data)}')

	def _segment_book(self, book_input_ids, chunk_size=256):
		'''
			1. read data and do sentence tokenization
			2. keep adding sentence until after tokenization > chunk_size tokens, step back and pad
		'''
		this_book_ids = []
		segment_ids = []
		for sent_ids in book_input_ids:

			if len(segment_ids) + len(sent_ids) < chunk_size - 1:
				segment_ids.extend(sent_ids)

			else:
				if len(segment_ids) != 0 and len(segment_ids) < chunk_size - 1:
					segment_ids += [eos_id]
					segment_ids.extend([pad_id] * (chunk_size - len(segment_ids) - 1))
					segment_ids = [cls_id] + segment_ids 
					assert len(segment_ids) == chunk_size, pdb.set_trace()
					this_book_ids.append(segment_ids)
					segment_ids = sent_ids
				
				if len(segment_ids) == 0 or len(segment_ids) >= chunk_size - 1:
					this_ids = sent_ids if len(segment_ids) == 0 else segment_ids
					num_sent_chunks = len(this_ids) // (chunk_size - 1) + 1
					for chunk_id in range(num_sent_chunks):
						sid = chunk_id * (chunk_size - 1)
						eid = min((chunk_id+1) * (chunk_size - 1), len(this_ids))
						if eid == (chunk_id+1) * (chunk_size - 1):
							this_book_ids.append([cls_id] + this_ids[sid:eid])
						else:
							this_book_ids.append([cls_id] + this_ids[sid:eid] + [eos_id] + [pad_id] * (chunk_size - (eid-sid) - 2))
					segment_ids = []

		assert all(len(x) == chunk_size for x in this_book_ids), pdb.set_trace()
		return torch.tensor(this_book_ids)
		
	def group_segment(self, book_data, num_chunks):
		ret = []
		for book_id in book_data:
			this_book_data = book_data[book_id]
			this_book_data = torch.split(this_book_data, num_chunks)
			this_book_data = [bd for bd in this_book_data if bd.shape[0] == num_chunks]
			ret.extend(this_book_data)
		return ret

	def load_tokenized(self, data_path, chunk_size, fileids):
		'''
			load tokenized training data
				each .pkl file contain _n_ books,
				each book contains field 'input_ids',
					which is a list of list of roberta token_ids, 
					each list is a sentence, no `bos` and `eos`, need to insert later

			output format: {'book_id': [#segments, chunk_size]}
			}
		'''
		files = sorted([fn for fn in os.listdir(data_path) if fn.endswith(".pkl")])
		fileids = set(fileids)

		ret = {}
		book_fid_map = {}
		for fname in tqdm(files):

			with open(os.path.join(data_path, fname), "rb") as f:
				data = pickle.load(f)

			for book_id in data:
				book_data = data[book_id]['input_ids']
				book_data = self._segment_book(book_data, chunk_size=chunk_size)
				ret[book_id]= book_data
				book_fid_map[book_id] = fname
				if len(ret) >= self.max_books:
						break

			if len(ret) >= self.max_books:
				break

		return ret, book_fid_map

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		
		this_item = torch.tensor(self.data[idx])
		padding_mask = torch.zeros(self.max_num_chunks).float()
		length = this_item.shape[0]
		if length < self.max_num_chunks:
			padding_mask[-(self.max_num_chunks-length):] = float('-inf')

		ret = {
				'data': F.pad(this_item, (0, 0, 0, self.max_num_chunks-length)),
				'padding_mask': padding_mask
			}

		return ret 



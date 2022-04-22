
import os
import pdb
import pickle
import argparse
from tqdm import tqdm
import torch
import random
import torch
from nltk.tokenize import sent_tokenize
torch.set_num_threads(1)
random.seed(42)
from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoConfig
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input-path", type=str, default=None)
	parser.add_argument("--output-path", type=str, default=None)
	parser.add_argument("--action", type=str, default=None)
	parser.add_argument("--chunk-size", type=int, default=256)
	parser.add_argument("--suffix-size", type=int, default=256)
	parser.add_argument("--tokenize-only", action="store_true", default=False)
	args = parser.parse_args()
	return args

args = parse_args()

model = RobertaModel.from_pretrained('roberta-base')
mode = model.cuda()
for param in model.parameters():
	param.requires_grad = False

def tokenize(args, data, chunk_size=256, single_chunk=False):
	'''
		tokenize the data with roberta tokenizer and chunk to 256,
		respect sentence boundaries
	'''
	sent_tok_data = sent_tokenize(data)
	segment_ids = []
	this_book_ids = []
	sent_chunk_map = {}
	for si, sent in enumerate(sent_tok_data):
		sent_ids = tokenizer(sent)['input_ids'][1:-1] # get rid of the <eos>, added later when adding to this_book_ids
		if len(segment_ids) + len(sent_ids) < chunk_size - 1:
			segment_ids.extend(sent_ids)

		else:
			# if adding new sentence leads to >256 tokens
			# back-off a sentence
			# pdb.set_trace()
			if len(segment_ids) != 0 and len(segment_ids) < chunk_size - 1:
				segment_ids += [eos_id]
				segment_ids.extend([pad_id] * (chunk_size - len(segment_ids) - 1))
				segment_ids = [cls_id] + segment_ids 
				assert len(segment_ids) == chunk_size, pdb.set_trace()
				this_book_ids.append(segment_ids)
				segment_ids = sent_ids
			
			if len(segment_ids) == 0 or len(segment_ids) >= chunk_size - 1:
				# split the sentence into multiple chunks of chunk_size
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

		sent_chunk_map[si] = len(this_book_ids)

	if len(segment_ids) != 0:
		segment_ids += [eos_id]
		segment_ids.extend([pad_id] * (chunk_size - len(segment_ids) - 1))
		segment_ids = [cls_id] + segment_ids 
		assert len(segment_ids) == chunk_size, pdb.set_trace()
		this_book_ids.append(segment_ids)

	sent_chunk_map[si] = len(this_book_ids)

	if chunk_size != args.chunk_size:
		if chunk_size > args.chunk_size:
			for i in range(len(this_book_ids)):
				this_book_ids[i] = this_book_ids[i][:args.chunk_size]
		else:
			for i in range(len(this_book_ids)):
				this_book_ids[i] = this_book_ids[i] + [pad_id] * (args.chunk_size -  len(this_book_ids[i]))
				assert len(this_book_ids[i]) == args.chunk_size

	if single_chunk:
		if len(this_book_ids) != 1:
			try:
				this_book_ids = [this_book_ids[0]]
			except:
				pdb.set_trace()
	else:
		assert all(len(x) == chunk_size for x in this_book_ids), pdb.set_trace()

	return torch.tensor(this_book_ids)


def encode(input_ids, batch_size=64):
	'''
		input_ids: #chunks x chunk_size(256)
		return
			#chunks x #hidden_dim
	'''
	# create attention mask
	input_ids = input_ids.cuda()
	attention_mask = (input_ids != pad_id).long().cuda()
	sent_vecs = []
	for batch_id in range(input_ids.shape[0] // batch_size + 1):
		batch_sid = batch_id * batch_size
		batch_eid = min((batch_id+1) * batch_size, input_ids.shape[0])
		try:
			outputs = model(input_ids=input_ids[batch_sid:batch_eid], 
						attention_mask=attention_mask[batch_sid:batch_eid],
						output_attentions=False,
						output_hidden_states=False)
		except:
			pdb.set_trace()
		this_batch_sent_vecs = outputs.last_hidden_state[:,0,:].detach() # batch_size x hidden_dim
		sent_vecs.append(this_batch_sent_vecs.cpu())

	sent_vecs = torch.cat(sent_vecs) # #chunks x #hidden_dim
	return sent_vecs

def encode_file(args, tokenize_only=False):

	try:
		with open(os.path.join(args.input_path), "rb") as f:
			data = pickle.load(f)
	except:
		raise FileNotFoundError, "invalid input file path"

	all_data = {}
	for book_id in tqdm(data):
		all_data[book_id] = []
		for example in data[book_id]:
			ctx = example['ctx']
			if args.suffix_size != args.chunk_size:
				pos = tokenizer.decode(tokenizer.encode(example['pos'])[1:-1][:args.suffix_size])
				negs = [tokenizer.decode(tokenizer.encode(neg)[1:-1][:args.suffix_size]) \
						for neg in example['negs']]
			else:
				pos = example['pos']
				negs = example['negs']

			ctx_ids = tokenize(args, ctx, chunk_size=args.chunk_size)
			pos_ids = tokenize(args, pos, chunk_size=args.suffix_size, single_chunk=True)
			negs_ids = [tokenize(args, neg, chunk_size=args.suffix_size, single_chunk=True) for neg in negs]
			all_data[book_id].append(
				{
					'ctx_vec': encode(ctx_ids) if not tokenize_only else ctx_ids,
					'pos_vec': encode(pos_ids) if not tokenize_only else pos_ids,
					'negs_vec': [encode(neg_ids) for neg_ids in negs_ids] if not tokenize_only else negs_ids
				}
			)

	with open(os.path.join(args.output_path), "wb") as f:
		pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
	
	
def main():
	# if tokenize_only, only run tokenizer, else extract [cls] vector from roberta-base output
	encode_file(args, tokenize_only=args.tokenize_only)

if __name__ == "__main__":
	main()

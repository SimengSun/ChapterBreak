import pdb
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Transformer
from transformers import (
	RobertaTokenizer, 
	RobertaModel,
)

from data.utils import pad_id


class LRLM(nn.Module):

	def __init__(self, args):
		super(LRLM, self).__init__()

		self.args = args

		roberta = RobertaModel.from_pretrained('roberta-base')
		
		if 'eval' in self.args.action:
			roberta.eval()
		del roberta.encoder.layer[-10:]
		self.encoder = roberta
		if 'eval' in self.args.action:
			self.encoder.eval()
			print(self.encoder)

		self.sentence_lm = Transformer(args, encoder=False)

	def _encode_pretrained_lm(self, batch):
		'''
			In:
				batch['data'] -> tensor([bsz, num_chunks, segment_size])
				batch['padding_mask'] -> tensor([bsz, num_chunks])
			Out:
				encoded batch -> tensor([bsz, num_chunks, d_model])
		'''

		bsz, num_chunks, segment_size = batch['data'].shape
		data = batch['data'].contiguous().view(-1, segment_size)
		attention_mask = (data != pad_id).long()

		

		out = self.encoder(input_ids=data, 
						   attention_mask=attention_mask,
						   output_attentions=False,
						   output_hidden_states=False).last_hidden_state[:, 0, :] # bsz*num_chunks x 768

		if 'eval' in self.args.action:
			outputs = out.detach()
		else:
			outputs = out
		
		return outputs.contiguous().view(bsz, num_chunks, outputs.shape[-1])

	def sentence_lm_forward(self, batch):
		'''
		In:
			batch -> tensor([bsz, num_chunks+1, 768])
		Out:
			loss -> tensor([bsz, num_chunks])
		'''
		# bsz x (num_chunks) x d_model
		slm_out = self.sentence_lm(batch) # shift happens in the transformer

		# right shift batch by one [bsz, num_chunks-1, d_model]
		target = batch['data'][:, 1:, :]
		padding_mask = batch['padding_mask'][:, 1:]
		padding_mask[padding_mask == 0] = 1.0
		padding_mask[padding_mask < 0] = 0

		if self.args.action == "eval-seg-lm":
			slm_out = slm_out[0, -1] # should all be the same during eval-suffix, since context is the same
			target = target[:, -1]
			dr = torch.matmul(slm_out, target.transpose(1, 0))
			return dr

		# a: predicted next segment representation
		# b: encoded gold next segment representation
		# model the density ratio p(b | a) / p(b) to maximize MI(encoded context, next segment)
		bsz, L, embed_dim = slm_out.shape
		slm_out_reshape = slm_out.contiguous().view(-1, embed_dim)
		target_reshape = target.contiguous().view(-1, embed_dim).transpose(1,0)
		dr_out = torch.matmul(slm_out_reshape, target_reshape)
		dr = dr_out.contiguous().view(bsz, L, -1)

		# contrastive loss
		bsz, n, _ = dr.shape
		loss = - F.log_softmax(dr * padding_mask[:, :, None], dim=2).reshape(bsz, n, bsz, n)
		loss = loss.transpose(2,1)[range(bsz), range(bsz)][:, range(n), range(n)]

		return loss, padding_mask

	def forward(self, raw_batch):
		'''
		IN:
			raw_batch:
				raw_batch['data'] -> tensor([bsz, num_chunks, segment_size])
				raw_batch['padding_mask'] -> tensor([bsz, num_chunks])
		
		OUT:
			loss
		'''

		batch = {}
		
		batch['data'] = self._encode_pretrained_lm(raw_batch)
		batch['data'] = F.pad(batch['data'], (0,0,1,0), "constant", 0) # pad left (segment-level padding)
		batch['padding_mask'] = F.pad(raw_batch['padding_mask'], (1, 0), "constant", 0)
		
		return self.sentence_lm_forward(batch)




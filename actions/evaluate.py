import os
import sys
import pdb
import wandb
import shutil
import pickle
import logging
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from apex import amp
from tqdm import tqdm
from models.utils import checkpoint


logger = logging.getLogger(__name__)

class Evaluator(object):

	def __init__(self, args, model, dataloaders, eval_type="segment"):

		self.args = args
		self.model = model
		self.model = model.cuda()
		self.model = nn.parallel.DistributedDataParallel(self.model, 
					device_ids=[args.local_rank], 
					output_device=args.local_rank, broadcast_buffers=True)
		self.eval_type = eval_type
		self.dataloaders = dataloaders

		self.modules = {'model': self.model}

	def eval_segment(self):
		self.model.eval()

		total_loss, total_num_examples = 0, 0
		for dl in self.dataloaders:
			for batch in tqdm(dl):
				loss, pad_mask = self.model(batch)

				num_examples = pad_mask.sum()
				loss = loss.sum()
				
				total_loss += loss
				total_num_examples += num_examples

		self.model.train()

		return total_loss/total_num_examples

	def eval_suffix_identification(self):
		self.model.eval()

		total_num = 0
		crrc_num = 0
		res = []
		for i, batch in enumerate(tqdm(self.dataloaders)):
			batch = {
				'data': batch['data'].squeeze(0),
				'padding_mask': batch['padding_mask'].squeeze(0)
			} 
			
			dr = self.model(batch)
			correct = ((dr - dr[0]) > 0).sum() == 0
			crrc_num += (1 * correct).item()
			total_num += 1

			res.append(((1 * correct).item(), dr.cpu().tolist()))
		print(f"suffix identification accuray: {crrc_num / total_num}")

	def __call__(self, ):

		with torch.no_grad():

			if self.eval_type == "segment":
				return self.eval_segment()

			else:
				return self.eval_suffix_identification()


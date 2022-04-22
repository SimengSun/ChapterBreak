import os
import sys
import pdb
import wandb
import shutil
import random
import logging
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import  CosineAnnealingLR
import numpy as np
from apex import amp
from tqdm import tqdm
from actions.evaluate import Evaluator
from models.utils import checkpoint

random.seed(42)
logger = logging.getLogger(__name__)
torch.set_num_threads(1)
class Trainer(object):

	def __init__(self, args, model, dataloader_lst, train_type="segment", validation_dataloader_lst=None, clip=0.25):

		self.args = args
		self.model = model
		self.train_type = train_type
		self.dataloaders = dataloader_lst
		self.validation_dataloaders = validation_dataloader_lst
		self.clip = clip
		self.epoch = 0
		if self.args.wandb:
			wandb.init(project=self.args.project_name, config=vars(args))
			print(f"setting run name to {args.wandb_run_name}")
			wandb.run.name = args.wandb_run_name
			wandb.run.save()
		self.steps = 0

		self.optimizer = optim.Adam(model.parameters(), 
									args.learning_rate, 
									betas=(0.9, 0.999), eps=1e-08)

		self.lr_scheduler = CosineAnnealingLR(self.optimizer, args.max_steps//args.accumulate_steps, eta_min=args.final_lr)


		self.model = model.cuda()
		if args.fp16:
			self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

		self.model = nn.parallel.DistributedDataParallel(self.model, 
					device_ids=[args.local_rank], 
					output_device=args.local_rank, broadcast_buffers=True,
					find_unused_parameters=True)

		if self.args.wandb:
			wandb.watch(self.model, log='all')

		self.modules = {
			'model': self.model,
			'optimizer': self.optimizer,
			'lr_scheduler': self.lr_scheduler
		}

		self.best_eval_loss = sys.maxsize

		logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
		logger.warning("Process rank: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
					   args.local_rank, args.n_gpu_per_node, bool(args.local_rank != -1), args.fp16)

	def optimize(self, step_i):

		if self.args.fp16:
			torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.clip)
		else:
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

		self.optimizer.step()
		self.optimizer.zero_grad()
		self.model.zero_grad()

		if step_i < self.args.warmup_steps:
			self.optimizer.param_groups[0]['lr'] = self.args.learning_rate * step_i / self.args.warmup_steps
			return self.optimizer.param_groups[0]['lr']
		else:
			self.lr_scheduler.step()
			return self.lr_scheduler.get_lr()[0]

	def compute_gradient(self, batch):
		self.model.train()

		loss, pad_mask = self.model(batch)

		num_examples = pad_mask.sum()
		loss = loss.sum()
		
		if self.args.fp16:
			with amp.scale_loss(loss, self.optimizer) as scaled_loss:
				scaled_loss.backward()
		else:
			loss.backward()

		return loss, num_examples

	def save_checkpoint(self, step, best=False):

		checkpoint_path = checkpoint(
			self.epoch, step, self.modules,
			self.args.checkpoint_path,
			max_checkpoints=self.args.max_checkpoints
		)

		if best:
			dirname = os.path.dirname(checkpoint_path)
			basename = os.path.basename(checkpoint_path)
			best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
			shutil.copy2(checkpoint_path, best_checkpoint_path)

	def train_epoch_suffix_lm(self, ):

		total_loss = 0
		total_num_examples = 0
		def eval_and_save(optimize_steps):
			vdl_lst = [iter(vdl) for vdl in self.validation_dataloaders]
			evaluator = Evaluator(self.args, self.model, vdl_lst)
			eval_loss = evaluator()
			
			if self.args.wandb:
				wandb.log({'eval_loss': eval_loss}, step=self.steps//self.args.accumulate_steps)
				logger.info(f"eval_loss {eval_loss}")
			else:
				logger.info(f"eval_loss {eval_loss}")

			self.save_checkpoint(optimize_steps, best=eval_loss < self.best_eval_loss)
			self.best_eval_loss = eval_loss if eval_loss < self.best_eval_loss else self.best_eval_loss

			if self.args.wandb:
				wandb.log({'best_eval_loss': self.best_eval_loss}, step=self.steps//self.args.accumulate_steps)
				logger.info(f"best_eval_loss {self.best_eval_loss}")
			else:
				logger.info(f"best_eval_loss {self.best_eval_loss}")
		
		def sample_batch(dataloaders):

			if len(dataloaders) == 0:
				return None

			dl_idx = random.choice(range(len(dataloaders)))
			try:
				batch = next(dataloaders[dl_idx])
				return batch

			except:
				dataloaders.pop(dl_idx)
				return sample_batch(dataloaders)

		def optimize_batch(batch, total_loss, total_num_examples):
			loss, num_examples = self.compute_gradient(batch)
			self.steps += 1

			total_loss += loss
			total_num_examples += num_examples
			optimize_steps = self.steps // self.args.accumulate_steps

			if self.steps % self.args.accumulate_steps == 0:
				lr = self.optimize(optimize_steps)
				if self.args.wandb:
					wandb.log({'train_loss': total_loss / total_num_examples, 
						   'steps': self.steps,
						   'optimize_steps': self.steps // self.args.accumulate_steps,
						   'lr': lr}, step=self.steps//self.args.accumulate_steps)
					logger.info(f"train_loss {total_loss.item() / total_num_examples} optimize_steps {self.steps // self.args.accumulate_steps} lr {lr}")
				else:
					logger.info(f"train_loss {total_loss.item() / total_num_examples} optimize_steps {self.steps // self.args.accumulate_steps} lr {lr}")

				
				if optimize_steps % self.args.eval_every == 0:
					eval_and_save(optimize_steps)

				if optimize_steps % self.args.ckpt_every == 0:
					self.save_checkpoint(optimize_steps)

			return optimize_steps, total_loss, total_num_examples
		
		dataloaders = [iter(dl) for dl in self.dataloaders]

		while True:
			batch = sample_batch(dataloaders)
			if batch is None:
				break

			optimize_steps, total_loss, total_num_examples = optimize_batch(batch, 
																		total_loss, 
																		total_num_examples)
			if optimize_steps >= self.args.max_steps:
				break


		lr = self.optimize(self.steps)
		if self.args.wandb:
			wandb.log({'train_loss': total_loss / total_num_examples, 
				   'steps': self.steps,
				   'lr': lr}, step=self.steps//self.args.accumulate_steps)
			logger.info(f"train_loss {total_loss.item() / total_num_examples} optimize_steps {self.steps // self.args.accumulate_steps} lr {lr}")
		eval_and_save(optimize_steps)


	def train_suffix_lm(self):
		optimize_steps = self.steps // self.args.accumulate_steps
		while (optimize_steps < self.args.max_steps) and (self.epoch < self.args.max_epochs):
			self.epoch += 1
			self.train_epoch_suffix_lm()


	def __call__(self, ):

		if self.train_type == "segment":
			self.train_suffix_lm()

		else:
			raise NotImplementedError









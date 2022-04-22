
import pdb
from args import parse_args
from data.utils import SLMEvalDataset
from actions.train import Trainer
from actions.evaluate import Evaluator
from models.long_range_lm import LRLM
from slurm import init_distributed_mode
from models.utils import restore
from utils import prepare_data

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

def main():
	args = parse_args()

	print('='*42)
	print('All configs:')
	v_configs = vars(args)
	for k in v_configs:
		print('\t{:20s} {:50s}'.format(k, str(v_configs[k])))
	print('='*42)

	init_distributed_mode(args)

	if args.action == "preprocess-data":
		dl_lst, vdl_lst = prepare_data(args)
		exit(0)

	elif args.action == "train-seg-lm":
		
		dl_lst, vdl_lst = prepare_data(args)

		model = LRLM(args)
		print(model)

		actioner = Trainer(args, model, dl_lst, validation_dataloader_lst=vdl_lst, train_type="segment")

		if args.restore:

			restore_modules = {
				module_name: module
				for module_name, module in actioner.modules.items()
			}

			epoch, step = restore(
				args.restore,
				restore_modules,
				num_checkpoints=1,
				map_location=torch.device('cuda'),
				strict=False
			)

			actioner.steps = 0 if not args.load_optimizer else step * args.accumulate_steps
			actioner.epoch = 0 

	elif args.action == "eval-seg-lm":
		
		ds = SLMEvalDataset(args, args.data_path, batch_size=1)
		sampler = SequentialSampler(ds)
		dl = DataLoader(ds, sampler=sampler, batch_size=args.batch_size)

		args.dropout_p = 0.0
		model = LRLM(args)
		actioner = Evaluator(args, model, dl, eval_type="suffix_identification")

		restore_modules = {
			module_name: module
			for module_name, module in actioner.modules.items()
		}

		epoch, step = restore(
			args.restore,
			restore_modules,
			num_checkpoints=1,
			map_location=torch.device('cuda'),
			strict=True
		)

	else:
		raise NotImplementedError

	actioner()

if __name__ == "__main__":
	main()







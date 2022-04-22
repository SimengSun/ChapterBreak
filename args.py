import argparse

ACTIONS = ['train-seg-lm', 'train-lm', 'eval-seg-lm', 'eval-lm', 'preprocess-data']

def parse_args():
	parser = argparse.ArgumentParser()

	# general args
	parser.add_argument("--action", type=str, default="train", choices=ACTIONS)
	parser.add_argument("--checkpoint-path", type=str, default=None)
	parser.add_argument("--max-checkpoints", type=int, default=5)
	parser.add_argument("--debug", action="store_true", default=False)
	parser.add_argument("--wandb", action="store_true", default=False)
	parser.add_argument("--project-name", type=str, default="lrlm")
	parser.add_argument("--wandb-run-name", type=str, default=None)
	parser.add_argument("--restore", type=str, default=None, help="reload checkpoint path")
	parser.add_argument("--load-optimizer", action="store_true", default=False)

	# data args
	parser.add_argument("--data-path", type=str, default=None)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--split", type=str, default="train")
	parser.add_argument("--max-books", type=int, default=30000)
	parser.add_argument("--max-chunk-per-seq", type=int, default=64)
	parser.add_argument("--max-tokens-per-batch", type=int, default=10240)
	parser.add_argument("--chunk-size-list", type=int, default=[128], nargs='+')
	parser.add_argument("--preprocess-data", action="store_true", default=False)

	# eval args
	parser.add_argument("--eval-out-path", type=str, default=None)

	# decoder(suffixLM) args
	parser.add_argument("--embedding-size", type=int, default=768)
	parser.add_argument("--num-heads", type=int, default=8)
	parser.add_argument("--model-size", type=int, default=768)
	parser.add_argument("--num-layers", type=int, default=6)
	parser.add_argument("--hidden-dim", type=int, default=2048)

	# LRLM general
	parser.add_argument("--train-small-roberta-from-scratch", action="store_true", default=False)
	parser.add_argument("--init-std", type=float, default=0.01)

	# train args
	parser.add_argument("--fp16", action="store_true", default=False)
	parser.add_argument("--learning-rate", type=float, default=7e-5)
	parser.add_argument("--final-lr", type=float, default=1e-7)
	parser.add_argument("--max-steps", type=int, default=200000)
	parser.add_argument("--warmup-steps", type=int, default=400)
	parser.add_argument("--max-epochs", type=int, default=150)
	parser.add_argument("--accumulate-steps", type=int, default=4)
	parser.add_argument("--optimizer", type=str, default="adam")
	parser.add_argument("--dropout-p", type=float, default=0.1)
	parser.add_argument("--local_rank", type=int, default=-1)
	parser.add_argument("--master_port", type=int, default=-1)
	parser.add_argument("--eval-every", type=int, default=500)
	parser.add_argument("--ckpt-every", type=int, default=1000)
	args = parser.parse_args()
	return args


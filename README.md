# SuffixLM

This is the repository for our NAACL 2022 paper [CHAPTERBREAK: A Challenge Dataset for Long-Range Language Models](http://arxiv/).


# Setup

This repository requires `Python 3.8` and  `cuda11/11.1.0`. Install all dependencies by running the following command:
```
  pip install -r requirements.txt
```


# Data

## Data download

Download [ChapterBreak](https://drive.google.com/drive/folders/1JkFHspT56_yRWwXVj47Fw0PzHtitODt5?usp=sharing) with various prefix lengths. Download the long-fanfic data (13,682 filtered long fanfics posted on Archive of Our Own) [here](https://drive.google.com/drive/folders/1Wb5dG6PABOleYGDG9rARVy1nSQ2WhHK2?usp=sharing). Please refer to the Appendix A for more details about the filtering of this dataset.

## Data format

Our data contains two split `pg19` and `ao3`. Each split contains a dictionary where the key is the id of a work, and value is a list of examples extracted from that work
```
{
	(workid): [{
      					'ctx': 	(the context preceding the gold suffix)
      					'pos': 	(the gold suffix)
      					'negs': (list of 5 negatives)
      				}, 

			   ...]
}
```
For the `ao3_longfic` dataset, the data is stored as following format:
```
{
  (workid): {
    'title': (title of this work)
    'fandom': (fandom this work belongs to, separated by ',')
    'genre': (genres this work belongs to, separated by ',')
    'Summary': (Summary written by the author)
    'num_chaps': (number of chapters)
    'num_words': (number of words)
    'text': (the work content)
  }
}
```


## Notes

While evaluating each example of ChapterBreak, if the total length of (prefix+suffix) is greater than the maximum input length of a model, truncate the prefix from left and the suffix from right. In our experiments, we truncate the suffix to be maximum 128 as the segment size of our SuffixLM is 128. We show in Appendix G. the variation in suffix lengths does not explain the large performance gap between SuffixLM and other LMs evaluated in our work.



# Model
## Model download

We provide a pre-trained suffixLM [here](https://drive.google.com/file/d/1eQJABPau-rbeag2_aZI-nrtZaD_HgT0t/view?usp=sharing). This model is trained on PG19 with segment size 128 and max input sequence of 10K tokens. To train your own suffixLM, please read the following sections.

## Evaluate SuffixLM

To run the SuffixLM, first run the following code to encode the chapterbreak dataset to binary data:

```
input_path=path/to/downloaded/chapterbreak/chapterbreak_ctx_512.json
output_path=/path/to/output/encoded/chapterbreak
mkdir -p $output_path
python tokenize_eval_data.py \
       --input-path $input_path \
       --output-path $output_path \
       --tokenize-only
```
After running the above command, you should see two files are created in your `output_path`: `pg19_ctx512.pkl` and `ao3_ctx512.pkl`. To run the ctx with other lengths, replace the `512` in the above example with the corresponding sequence length. The length of original suffixes is around 150 words, we truncate them to 128 tokens as the segment size of our SuffixLM is set to be 128.

Next, make sure you have a trained or downloaded suffixLM model with name `best_checkpoint.pt` in your experiment folder.

Run the following command to evaluate suffixLM on `PG19` split with prefix length 512, replace `[port_number]` with a random port number

```
data_path=/path/to/encoded/chapterbreak/pg19_ctx512.pkl
experiment_path=/path/to/trained/suffixlm/model
python -m torch.distributed.run --master_port [port_number] main.py \
  --data-path $data_path  \
  --fp16  \
  --action eval-seg-lm \
  --batch-size 1 \
  --restore $experiment_path/best_checkpoint.pt 

```

## Train suffixLM

To train suffixLM, first download PG19 data from [here](https://github.com/deepmind/pg19), then encode the PG19 data for training SuffixLM. You can encode the sharded data in parallel using commands such as the following:
```
IN_PATH=/path/to/raw/text/train
OUT_PATH=/path/to/encoded/text/train-tok
for (( SHARD_ID={s_id}; SHARD_ID<{e_id}; SHARD_ID++ )); do
python encode_pg19_train.py \
       --input-path $IN_PATH --output-path $OUT_PATH --shard-id $SHARD_ID --shard-size 100 \
       --chunk-size 128 --batch-size 64
done
```

You also need to encode the `test` and `eval` set similarly. After encoding the data, you should have a folder named `train-tok` containing the encoded training files and a folder `valid-tok` containing encoded validation file.  Train suffixLM with the following command.

```
data_path=/path/to/data
experiment_path=/path/to/save/checkpoints
mkdir -p $experiment_path
export NGPU=1
python -m torch.distributed.launch --master_port=[port_number]  main.py \
  --data-path $data_path    \
  --fp16    \
  --split train-tok   \
  --action train-seg-lm   \
  --checkpoint-path $experiment_path 
```



# Citation

```
@inproceedings{long21,
author={Simeng Sun and Katherine Thai and Mohit Iyyer},
Booktitle = {North American Association for Computational Linguistics},
Year = "2022",
Title={ChapterBreak: A Challenge Dataset for Long-Range Language Models},
}
```

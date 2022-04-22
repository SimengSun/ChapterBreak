data_path=/mnt/nfs/work1/miyyer/simengsun/data/PG19/valid_hard_suffix/chapter_breaks_new/rawmaxlen-4096-suffix-168.pkl
out_path=/mnt/nfs/work1/miyyer/simengsun/data/PG19/valid_hard_tok/chapter_breaks_new/rawmaxlen-4096-suffix-168.pkl

mkdir -p $out_path
python encode_eval_Data.py \
       --input-path $data_path \
       --output-path $out_path \
       --chunk-size 128 \
       --suffix-size 128 \
       --tokenize-only

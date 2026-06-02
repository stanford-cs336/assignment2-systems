#!/usr/bin/env bash

# benchmark
# uv run nsys profile \
#    -o ../result/train_profile \
#    --force-overwrite true \
#    -t cuda,cudnn,cublas,nvtx,osrt \
#    -- \
#    python train.py \
#    --vocab_size 10000 \
#    --d_model 256 \
#    --num_layers 4 \
#    --num_heads 8 \
#    --d_ff 768 \
#    --context_length 256 \
#    --batch_size 16 \
#    --num_steps 15 \
#    --vocab_path "../result/tinystories_bpe_vocab.json" \
#    --merges "../result/tinystories_bpe_merge.json" \
#    --bm_mode \
#    --warmup 5\
#    --num_bm 10\
#    --backward \
   # --backward \
   # --optim \

python train.py \
   --vocab_path "../result/tinystories_bpe_vocab.json" \
   --merges "../result/tinystories_bpe_merge.json" \
   --vocab_size 10000 \
   --d_model 768 \
   --num_layers 16 \
   --num_heads 16 \
   --d_ff 3072 \
   --context_length 256 \
   --batch_size 16 \
   --num_steps 15 \
   --bench "forward"\
   --warmup 5\
   --bench_step 10\
   --use_amp \


# python train.py \
#     --train_data ../result/TinyStoriesV2-GPT4-train_tokens.npy \
#     --val_data ../result/TinyStoriesV2-GPT4-valid_tokens.npy \
#     --vocab_size 10000 \
#     --d_model 256 \
#     --num_layers 4 \
#     --num_heads 8 \
#     --d_ff 768 \
#     --context_length 256 \
#     --batch_size 1 \
#     --num_steps 10000 \
#     --vocab_path "../result/tinystories_bpe_vocab.json" \
#     --merges "../result/tinystories_bpe_merge.json" \
#     --generate "what are you doing?"

# python train.py \
#     --train_data ../result/TinyStoriesV2-GPT4-train_tokens.npy \
#     --val_data ../result/TinyStoriesV2-GPT4-valid_tokens.npy \
#     --vocab_size 10000 \
#     --d_model 512 \
#     --d_ff 1344 \
#     --num_layers 4 \
#     --num_heads 16 \
#     --context_length 256 \
#     --batch_size 32 \
#     --num_steps 10000 \
#     --vocab_path "../result/tinystories_bpe_vocab.json" \
#     --merges "../result/tinystories_bpe_merge.json" \
#     --generate "what are you doing?"

    

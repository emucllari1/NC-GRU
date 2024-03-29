python denoise_task.py \
--model NCGRU \
--T 200 \
--iters 10000 \
--batch_size 128 \
--hidden_size 118 \
--capacity 4 \
--in_out_optimizer adam \
--in_out_lr 1e-3 \
--A_optimizer adam \
--A_lr 1e-3 \

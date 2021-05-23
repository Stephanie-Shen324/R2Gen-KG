python main.py \
--image_dir data/mimic_cxr/images/ \
--ann_path data/mimic_cxr/annotation.json \

--max_seq_length 100 \
--threshold 10 \
--batch_size 16 \
--epochs 30 \
--save_dir results/mimic_cxr \
--step_size 1 \
--gamma 0.8 \
--pretrained models/mimic_gcnclassifier_v1_ones3_t0v1t2_lr1e-6_e10.pth \
--d_vf 1024 \
--num_classes 36 \
--dataset_name mimic_cxr \
--seed 456789

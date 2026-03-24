accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path /data/model/stable-diffusion-v1-5 \
  --crop_root /data/Sketch/manga_line/crop \
  --index_file panel_index.jsonl \
  --train_stage main \
  --output_dir /data/Sketch/output \
  --train_batch_size 4 \
  --num_train_epochs 20 \
  --caption_dropout_prob 0.25 \
  --sketch_dropout_prob 0.05 \
  --sketch_gate_warmup_ratio 0.12 \
  --num_sketch_sem_tokens 8 \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 5 \
  --validation_every_n_steps 1000 \
  --num_validation_samples 4 \
  --validation_num_inference_steps 30 \
  --mixed_precision fp16 


# accelerate launch train_text_to_image_lora.py \
#   --pretrained_model_name_or_path /data/model/stable-diffusion-v1-5 \
#   --crop_root /data/Sketch/manga_line/crop \
#   --index_file panel_index.jsonl \
#   --train_stage ref \
#   --ref_image_encoder_name_or_path openai/clip-vit-large-patch14 \
#   --num_ref_tokens 8 \
#   --max_refs_per_panel 3 \
#   --output_dir /data/Sketch/experiments/panel_line_ref \
#   --train_batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --num_train_epochs 10 \
#   --caption_dropout_prob 0.25 \
#   --sketch_dropout_prob 0.05 \
#   --sketch_gate_warmup_ratio 0.12 \
#   --num_sketch_sem_tokens 8 \
#   --dataloader_num_workers 4 \
#   --checkpointing_steps 1000 \
#   --checkpoints_total_limit 5 \
#   --validation_every_n_steps 1000 \
#   --num_validation_samples 4 \
#   --validation_num_inference_steps 30 \
#   --mixed_precision fp16 \
#   --resume_from_checkpoint /data/Sketch/experiments/panel_line_main
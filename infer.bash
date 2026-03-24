
# 1. 只有 sketch
python infer_panel_line.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --checkpoint_dir /path/to/your_checkpoint_or_output_dir \
  --sketch_path /path/to/sketch.png \
  --output_path /path/to/out.png

# # 2. sketch + caption
# python infer_panel_line.py \
#   --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
#   --checkpoint_dir /path/to/your_checkpoint_or_output_dir \
#   --sketch_path /path/to/sketch.png \
#   --caption "Two children are sitting at a table reading together." \
#   --output_path /path/to/out.png


# # 3. sketch + caption + ref

# 先准备一个 refs.json：

# [
#   {
#     "image": "/path/to/ref_char0.png",
#     "bbox_norm": [0.12, 0.18, 0.42, 0.91]
#   },
#   {
#     "image": "/path/to/ref_char1.png",
#     "bbox_norm": [0.55, 0.15, 0.84, 0.93]
#   }
# ]

# 再运行：

# python infer_panel_line.py \
#   --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
#   --checkpoint_dir /path/to/your_checkpoint_or_output_dir \
#   --sketch_path /path/to/sketch.png \
#   --caption "Two children are sitting at a table reading together." \
#   --ref_json /path/to/refs.json \
#   --output_path /path/to/out.png
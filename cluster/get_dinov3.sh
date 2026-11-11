#!/usr/bin/env bash
# Run ON the cluster login node, yourself: downloads gated DINOv3 into ~/hf_dl/dinov3-vits16.
# Asks for your HF token (hidden); it is only passed to this one download and not saved.
set -e
read -rsp 'Paste your Hugging Face token (hidden), then Enter: ' HF_TOKEN; echo
mkdir -p ~/hf_dl
HF_TOKEN="$HF_TOKEN" apptainer exec -B ~/hf_dl:/dl /staging/s/smittal39/k1-follow/containers/evt-habitat.sif /opt/hab/bin/python -c "
from huggingface_hub import snapshot_download as d
d('facebook/dinov3-vits16-pretrain-lvd1689m', local_dir='/dl/dinov3-vits16', ignore_patterns=['.gitattributes'])"
rm -rf ~/hf_dl/dinov3-vits16/.cache
ls -l ~/hf_dl/dinov3-vits16 && echo "OK - tell Claude it's done"

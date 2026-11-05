#!/usr/bin/env python3
"""Run one isolated VGGT inference size and write GPU allocator measurements."""
import argparse
import csv
import json
from pathlib import Path

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

parser = argparse.ArgumentParser()
parser.add_argument("--frame-list", type=Path, required=True)
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = list(csv.DictReader(args.frame_list.open(newline="")))
paths = [Path(row["rgb_path"]) for row in rows]
paths = [paths[i % len(paths)] for i in range(args.n)]
if not torch.cuda.is_available():
    raise RuntimeError("Host CUDA is unavailable; run this probe outside the sandbox")
device = torch.device("cuda:0")
dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
model = VGGT()
state = torch.load(args.checkpoint_dir / "model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval().to(device)
torch.cuda.reset_peak_memory_stats(device)
images = load_and_preprocess_images([str(p) for p in paths], mode="pad").to(device)
with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
    prediction = model(images)
torch.cuda.synchronize(device)
result = {
    "status": "completed", "n_frames": args.n,
    "gpu": torch.cuda.get_device_name(device), "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
    "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
    "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
    "free_after_inference_bytes": torch.cuda.mem_get_info(device)[0],
    "input_shape": list(images.shape), "dtype": str(dtype),
    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

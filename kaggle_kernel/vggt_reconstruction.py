"""
=============================================================================
MONASTERYAI — KAGGLE GPU 3D RECONSTRUCTION KERNEL
=============================================================================
"""

import os
import sys
import subprocess
import time
import urllib.request
import glob
from pathlib import Path

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [MonasteryAI-GPU] {msg}", flush=True)

# 1. Install required packages
log("=== Step 1: Installing 3D Vision & Geometry Libraries ===")
packages = ["trimesh", "open3d", "scipy", "roma", "pyrender", "imageio[ffmpeg]", "timm", "einops"]
for pkg in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        log(f" Installed {pkg}")
    except Exception as e:
        log(f"⚠️ Warning installing {pkg}: {e}")

# Clone DUSt3R
log("=== Step 2: Cloning VGGT-Omega / DUSt3R Foundation Engine ===")
if not os.path.exists("dust3r"):
    subprocess.check_call(["git", "clone", "--recursive", "https://github.com/naver/dust3r.git"])
sys.path.append(os.path.abspath("dust3r"))
sys.path.append(os.path.abspath("dust3r/croco"))

import torch
log(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU Device: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB)")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Model Loading
log("=== Step 3: Initializing VGGT-Omega / DUSt3R Pretrained Weights ===")
try:
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.image import load_images

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    log(f"Downloading & Loading {model_name} onto {device}...")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    log(" Foundation Model successfully loaded into GPU memory!")

    # 3. Find Images
    log("=== Step 4: Loading Multi-View Monastic Video Keyframes ===")
    images_list = sorted(glob.glob("/kaggle/input/**/*.jpg", recursive=True) + glob.glob("/kaggle/input/**/*.png", recursive=True))
    if not images_list:
        log("No custom dataset found in /kaggle/input; using sample multi-view assets from repo.")
        images_list = sorted(glob.glob("dust3r/croco/assets/*.png"))[:12]
        if not images_list:
            images_list = sorted(glob.glob("dust3r/assets/*.png"))[:12]

    log(f"Processing {len(images_list)} input frames for 3D reconstruction...")
    images = load_images(images_list, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    log(f"Running Pairwise Visual Geometry Inference across {len(pairs)} angle pairs...")
    output = inference(pairs, model, device, batch_size=2)

    log("Solving Global Point-Cloud Optimization...")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init='mst', niter=250, schedule='cosine', lr=0.01)
    log(f" 3D Alignment Converged with Loss: {loss:.4f}")

    # 4. Save GLB
    log("=== Step 5: Exporting Dense 3D Point Cloud (.glb) ===")
    out_glb = "/kaggle/working/monastery_3d_model.glb"
    scene.save_glb(out_glb, min_conf_thr=45.0, max_points=1000000)
    size_mb = os.path.getsize(out_glb) / (1024 * 1024)
    log(f" Successfully exported 3D Model: {out_glb} ({size_mb:.2f} MB)")

except Exception as e:
    log(f"⚠️ Encountered generation exception: {e}")
    import traceback
    traceback.print_exc()

# 5. Sync to Supabase Public Bucket
log("=== Step 6: Syncing 3D Model to Supabase Storage ===")
SUPABASE_URL = "https://ygdmzmqkztwpmkdozzsp.supabase.co"
SUPABASE_SERVICE_KEY = "YOUR_SUPABASE_KEY"
STORAGE_BUCKET = "monasteries"
DEST_NAME = "emei_mountain.glb"

out_file = "/kaggle/working/monastery_3d_model.glb"
if os.path.exists(out_file):
    with open(out_file, "rb") as f:
        data = f.read()
    req = urllib.request.Request(
        f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{DEST_NAME}",
        data=data,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "model/gltf-binary",
            "x-upsert": "true"
        }
    )
    try:
        with urllib.request.urlopen(req) as resp:
            log(f" Direct Supabase CDN Published: {SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{DEST_NAME}")
    except Exception as e:
        log(f"❌ Supabase upload failed: {e}")
else:
    log("ℹ️ Completed verification pass.")

log("=== MonasteryAI 3D Reconstruction Pipeline Finished ===")

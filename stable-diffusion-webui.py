from colorama import Fore
from pathlib import Path

import modal
import subprocess
import sys
import shlex
import os

# modal系の変数を定義
stub = modal.Stub("stable-diffusion-webui-automatic1111")
volume_main = modal.SharedVolume().persist("stable-diffusion-webui-automatic1111-main")

# webui内のパスを定義
webui_dir = "/content/stable-diffusion-webui"
webui_model_dir = webui_dir + "/models/Stable-diffusion/"
webui_lora_dir = webui_dir + "/models/Lora/"
webui_texture_dir = webui_dir + "/embeddings/"

# Modelファイル
model_files = [
    {
        "url": "https://civitai.com/api/download/models/11745",
        "file_name": "Chilloutmix-Ni-pruned-fp32-fix.safetensors",
        "info": "https://civitai.com/models/6424/chilloutmix",
    },
]

# Loraファイル
lora_files = [
    # {
    #     "url": "https://civitai.com/api/download/models/xxxxx",
    #     "file_name": "xxxxx..safetensors",
    #     "info": "",
    # }
]

# Textureファイル
texture_files = [
    # {
    #     "url": "https://civitai.com/api/download/models/xxxxx",
    #     "file_name": "xxxxx..safetensors",
    #     "info": "",
    # }
]

@stub.function(
    image=modal.Image.from_dockerhub("python:3.10")
    .apt_install(
        "git", "libgl1-mesa-dev", "libglib2.0-0", "libsm6", "libxrender1", "libxext6"
    )
    .run_commands(
        "pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
    )
    .pip_install(
        "astunparse",
        "blendmodes",
        "accelerate",
        "basicsr",
        "fonts",
        "font-roboto",
        "facexlib",
        "gfpgan==1.3.8",
        "gradio==3.29.0",
        "numpy",
        "omegaconf",
        "opencv-contrib-python",
        "requests",
        "piexif",
        "Pillow",
        "pytorch_lightning==1.7.7",
        "realesrgan",
        "scikit-image>=0.19",
        "timm==0.4.12",
        "transformers==4.25.1",
        "torch",
        "einops",
        "jsonmerge",
        "clean-fid",
        "resize-right",
        "torchdiffeq",
        "kornia",
        "lark",
        "inflection",
        "GitPython",
        "torchsde",
        "safetensors",
        "psutil",
        "rich",
        "colorama",
        "xformers",
        "clip",
        "gdown",
        "httpcore",
        "tensorboard",
        "taming-transformers",
        "test-tube",
        "diffusers",
        "invisible-watermark",
        "pyngrok",
        "huggingface_hub"
    )
    .pip_install("git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    shared_volumes={webui_dir: volume_main},
    gpu="a10g",
    timeout=6000,
)
async def run_stable_diffusion_webui():
    print(Fore.CYAN + "\n---------- Download Start ----------\n")

    webui_dir_path = Path(webui_model_dir)
    if not webui_dir_path.exists():
        subprocess.run(f"git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui {webui_dir}", shell=True)

    # Modelファイルをダウンロードし、保存する
    for model_file in model_files:
        download_file(model_file["url"], webui_model_dir + model_file["file_name"])

    # Loraファイルをダウンロードし、保存する
    for lora_file in lora_files:
        download_file(lora_file["url"], webui_lora_dir + lora_file["file_name"])

    # Textureファイルをダウンロードし、保存する
    for texture_file in texture_files:
        download_file(texture_file["url"], webui_texture_dir + texture_file["file_name"])

    print(Fore.CYAN + "\n---------- Download Complete ----------\n")

    # WebUIを起動
    sys.path.append(webui_dir)
    sys.argv += shlex.split("--skip-install --xformers")
    os.chdir(webui_dir)
    from launch import start, prepare_environment

    prepare_environment()
    sys.argv = shlex.split("--a --gradio-debug --share --xformers --enable-insecure-extension-access")
    start()

# 指定のURLからファイルをダウンロードし、指定したパスに保存する
@stub.function()
def download_file(url, path):
    # ファイルのダウンロード
    subprocess.run(
        shlex.split(f"wget {url} -O {path}"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # ファイルの存在確認
    if not os.path.exists(path):
        print(Fore.RED + f"File not found: {path}")
        sys.exit(1)

    print(Fore.GREEN + f"Downloaded: {path}")

@stub.local_entrypoint()
def main():
    run_stable_diffusion_webui.call()


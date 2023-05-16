from colorama import Fore
from pathlib import Path

import modal
import shutil
import subprocess
import sys
import shlex
import os

# modal系の変数の定義
stub = modal.Stub("stable-diffusion-webui-automatic1111")
volume_main = modal.SharedVolume().persist("stable-diffusion-webui-automatic1111-main")

# 色んなパスの定義
webui_dir = "/content/stable-diffusion-webui"
webui_model_dir = webui_dir + "/models/Stable-diffusion/"

# モデルのID
model_ids = [
    {
        "repo_id": "hakurei/waifu-diffusion-v1-4",
        "model_path": "wd-1-4-anime_e1.ckpt",
        "config_file_path": "wd-1-4-anime_e1.yaml",
    },
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
    print(Fore.CYAN + "\n---------- セットアップ開始 ----------\n")

    webui_dir_path = Path(webui_model_dir)
    if not webui_dir_path.exists():
        subprocess.run(f"git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui {webui_dir}", shell=True)

    # Hugging faceからファイルをダウンロードしてくる関数
    def download_hf_file(repo_id, filename):
        from huggingface_hub import hf_hub_download

        download_dir = hf_hub_download(repo_id=repo_id, filename=filename)
        return download_dir


    for model_id in model_ids:
        print(Fore.GREEN + model_id["repo_id"] + "のセットアップを開始します...")

        if not Path(webui_model_dir + model_id["model_path"]).exists():
            # モデルのダウンロード＆コピー
            model_downloaded_dir = download_hf_file(
                model_id["repo_id"],
                model_id["model_path"],
            )
            shutil.copy(model_downloaded_dir, webui_model_dir + os.path.basename(model_id["model_path"]))


        if "config_file_path" not in model_id:
            continue

        if not Path(webui_model_dir + model_id["config_file_path"]).exists():
            # コンフィグのダウンロード＆コピー
            config_downloaded_dir = download_hf_file(
                model_id["repo_id"], model_id["config_file_path"]
            )
            shutil.copy(config_downloaded_dir, webui_model_dir + os.path.basename(model_id["config_file_path"]))


        print(Fore.GREEN + model_id["repo_id"] + "のセットアップが完了しました！")

    print(Fore.CYAN + "\n---------- セットアップ完了 ----------\n")

    # WebUIを起動
    sys.path.append(webui_dir)
    sys.argv += shlex.split("--skip-install --xformers")
    os.chdir(webui_dir)
    from launch import start, prepare_environment

    prepare_environment()
    # 最初のargumentは無視されるので注意
    sys.argv = shlex.split("--a --gradio-debug --share --xformers")
    start()


@stub.local_entrypoint()
def main():
    run_stable_diffusion_webui.call()


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
webui_lora_dir = webui_dir + "/models/Lora/"
webui_texture_dir = webui_dir + "/embeddings/"

# HuggingFaceのモデルのID
# hugging_face_model_ids = [
#     {
#         "repo_id": "hakurei/waifu-diffusion-v1-4",
#         "model_path": "wd-1-4-anime_e1.ckpt",
#         "config_file_path": "wd-1-4-anime_e1.yaml",
#     },
# ]

# Modelファイル
model_files = [
    {
        "url": "https://civitai.com/api/download/models/11745",
        "file_name": "Chilloutmix-Ni-pruned-fp32-fix.safetensors",
        "info": "https://civitai.com/models/6424/chilloutmix",
    },
    {
        "url": "https://civitai.com/api/download/models/63786",
        "file_name": "bra_v5.safetensors",
        "info": "https://civitai.com/models/25494/brabeautiful-realistic-asians-v2",
    },
    {
        "url": "https://civitai.com/api/download/models/29460",
        "file_name": "realisticVisionV20_v20.safetensors",
        "info": "https://civitai.com/models/4201/realistic-vision-v20",
    },
    {
        "url": "https://civitai.com/api/download/models/15640",
        "file_name": "uberRealisticPornMerge_urpmv13.safetensors",
        "info": "https://civitai.com/models/2661/uber-realistic-porn-merge-urpm",
    },
    {
        "url": "https://civitai.com/api/download/models/56401",
        "file_name": "lazymixRealAmateur_v20.safetensors",
        "info": "https://civitai.com/models/10961?modelVersionId=56401",
    },
]

# Loraファイル
lora_files = [
    {
        "url": "https://civitai.com/api/download/models/16677",
        "file_name": "cuteGirlMix4_v10.safetensors",
        "info": "https://civitai.com/models/14171/cutegirlmix4",
    },
    {
        "url": "https://civitai.com/api/download/models/34562",
        "file_name": "JapaneseDollLikeness_v15.safetensors",
        "info": "https://civitai.com/models/28811/japanesedolllikeness-v15",
    },
    {
        "url": "https://civitai.com/api/download/models/26413",
        "file_name": "public_v1.0-000005.safetensors",
        "info": "https://civitai.com/models/22123/in-public-photographers?modelVersionId=26413",
    },
    {
        "url": "https://civitai.com/api/download/models/48983",
        "file_name": "blacked_v1.5.safetensors",
        "info": "https://civitai.com/models/44353/blacked",
    },
    {
        "url": "https://civitai.com/api/download/models/10187",
        "file_name": "skirtlift-v1.safetensors",
        "info": "https://civitai.com/models/8631/skirtlift-the-astonishing-sequel-to-shirtlift",
    },
    {
        "url": "https://civitai.com/api/download/models/19492",
        "file_name": "GodPussy2_Innie.safetensors",
        "info": "https://civitai.com/models/10332/realistic-vaginas-god-pussy-2-innie",
    },
    {
        "url": "https://civitai.com/api/download/models/18077",
        "file_name": "Creampie_v11.safetensors",
        "info": "https://civitai.com/models/14557/creampie-and-hairy-pussy",
    },
]

# Textureファイル
texture_files = [
    # {
    #     "url": "",
    #     "file_name": "",
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
    print(Fore.CYAN + "\n---------- Setup Start ----------\n")

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

    # Hugging faceからファイルをダウンロードしてくる関数
    # def download_hf_file(repo_id, filename):
    #     from huggingface_hub import hf_hub_download

    #     download_dir = hf_hub_download(repo_id=repo_id, filename=filename)
    #     return download_dir


    # for model_id in hugging_face_model_ids:
    #     print(Fore.GREEN + model_id["repo_id"] + "のセットアップを開始します...")

    #     if not Path(webui_model_dir + model_id["model_path"]).exists():
    #         # モデルのダウンロード＆コピー
    #         model_downloaded_dir = download_hf_file(
    #             model_id["repo_id"],
    #             model_id["model_path"],
    #         )
    #         shutil.copy(model_downloaded_dir, webui_model_dir + os.path.basename(model_id["model_path"]))


    #     if "config_file_path" not in model_id:
    #         continue

    #     if not Path(webui_model_dir + model_id["config_file_path"]).exists():
    #         # コンフィグのダウンロード＆コピー
    #         config_downloaded_dir = download_hf_file(
    #             model_id["repo_id"], model_id["config_file_path"]
    #         )
    #         shutil.copy(config_downloaded_dir, webui_model_dir + os.path.basename(model_id["config_file_path"]))

        # print(Fore.GREEN + model_id["repo_id"] + "のセットアップが完了しました！")

    print(Fore.CYAN + "\n---------- Setup Complete ----------\n")

    # WebUIを起動
    sys.path.append(webui_dir)
    sys.argv += shlex.split("--skip-install --xformers")
    os.chdir(webui_dir)
    from launch import start, prepare_environment

    prepare_environment()
    # 最初のargumentは無視されるので注意
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


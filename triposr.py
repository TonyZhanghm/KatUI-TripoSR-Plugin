import os
import torch
import torchvision
import numpy as np
import pathlib
import rembg
from PIL import Image
import urllib

from torch import Tensor
from typing import TypedDict

from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode

import sys
sys.path.append("nodes/KatUITripoSRPlugin")

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video


class TripoSRModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.triposr_model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        TSR_model: TSR

    def execute(self, 
                pretrained_model_name_or_path: str="stabilityai/TripoSR",
                device: torch.device = torch.device("cuda"),
                chunk_size: int = 8192,
                ) -> ReturnDict:
        model = TSR.from_pretrained(
            pretrained_model_name_or_path,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)
        return self.ReturnDict(TSR_model=model)


class LoadImage(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.load_image")
    def __init__(self) -> None:
        pass

    def execute(self, image_path_or_url: str) -> Tensor:
        if image_path_or_url.startswith("http"):
            image = Image.open(urllib.request.urlopen(image_path_or_url))
        else:
            image = Image.open(image_path_or_url)
        return image

class RemoveBackground(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.remove_background")
    def __init__(self) -> None:
        pass

    def execute(self, image: Image.Image, no_remove_bg: bool = False, foreground_ratio: float = 0.85) -> Tensor:
        if no_remove_bg:
            return image.convert("RGB")
        rembg_session = rembg.new_session()
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image


class RunTripoSR(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.run_triposr")
    def __init__(self) -> None:
        pass

    def execute(self, model: TSR, image: Image.Image, device: torch.device = torch.device("cuda")) -> Tensor:
        scene_codes = model([[np.array(image)]], device=device)
        return scene_codes


class ExportMesh(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.export_mesh")
    def __init__(self) -> None:
        pass

    def execute(self, model: TSR, scene_codes: Tensor, path: str="mesh.obj", mc_resolution: int=256) -> pathlib.Path:
        assert path.endswith(".obj") or path.endswith(".glb")
        meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)
        pathlib_path = os.path.join(self.OUTPUT_PATH, path)
        meshes[0].export(pathlib_path)
        return pathlib_path
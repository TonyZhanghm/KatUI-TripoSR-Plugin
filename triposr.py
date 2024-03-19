import io
import torch
import numpy as np
import pathlib
import rembg
import PIL.Image
import urllib.request
import trimesh

from torch import Tensor
from typing import TypedDict, List, Literal

from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video


class TripoSRModelLoader(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.triposr_model_loader")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        TSR_model: TSR

    def execute(
            self,
            pretrained_model_name_or_path: str = "stabilityai/TripoSR",
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


# class RemoveBackground(BaseNode):

#     @KatzukiNode(node_type="diffusion.triposr.remove_background")
#     def __init__(self) -> None:
#         pass

#     def execute(self, image: PIL.Image.Image, no_remove_bg: bool = False, foreground_ratio: float = 0.85) -> PIL.Image.Image:
#         if no_remove_bg:
#             return image.convert("RGB")
#         rembg_session = rembg.new_session()
#         image = remove_background(image, rembg_session)
#         image = resize_foreground(image, foreground_ratio)
#         image_np = np.array(image).astype(np.float32) / 255.0
#         image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
#         image_pil = PIL.Image.fromarray((image_np * 255.0).astype(np.uint8))
#         return image_pil


class RunTripoSR(BaseNode):

    @KatzukiNode(
        node_type="diffusion.triposr.run_triposr",
        input_description={
            "image": "The input image tensor of shape [1, 3, H, W] with values in range [-1, 1] and grey background",
        },
    )
    def __init__(self) -> None:
        pass

    def execute(self, model: TSR, image: Tensor, device: torch.device = torch.device("cuda")) -> Tensor:
        with torch.no_grad():
            image = (image + 1) / 2 # [-1, 1] -> [0, 1]
            scene_codes = model(image, device=device)
        return scene_codes


class SceneCodesToMesh(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.scene_codes_to_mesh")
    def __init__(self) -> None:
        pass

    def execute(self, model: TSR, scene_codes: Tensor, mc_resolution: int = 256) -> List[trimesh.Trimesh]:
        with torch.no_grad():
            meshes: List[trimesh.Trimesh] = model.extract_mesh(scene_codes, resolution=mc_resolution)
            return meshes


class RenderTrimeshAsObj(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.render_trimesh_as_obj")
    def __init__(self) -> None:
        pass

    class ReturnDict(TypedDict):
        obj: bytes

    def execute(self, meshes: List[trimesh.Trimesh]) -> ReturnDict:
        binary_stream = io.BytesIO()
        meshes[0].export(file_obj=binary_stream, file_type="obj")
        binary_data = binary_stream.getvalue()
        return self.ReturnDict(obj=binary_data)


class SaveMesh(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.save_mesh")
    def __init__(self) -> None:
        pass

    def execute(self, meshes: List[trimesh.Trimesh], path: str = "mesh.obj", file_type: Literal["stl", "dict", "glb", "obj", "gltf", "dict64", "stl_ascii"] = "obj") -> pathlib.Path:
        assert path.split(".")[-1] == file_type, f"File type {file_type} does not match file extension {path.split('.')[-1]}"
        pathlib_path = self.OUTPUT_PATH / path
        # QUESTION: what is [0] for?
        meshes[0].export(file_obj=pathlib_path, file_type=file_type)
        return pathlib_path

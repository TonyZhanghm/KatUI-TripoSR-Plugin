import io
import time
import torch
import pathlib
import trimesh
import numpy as np
import asyncio
import base64
import PIL.Image

from torch import Tensor
from typing import TypedDict, List, Literal, Dict, Any, Optional

from kokikit.dataset import NeRFDataset as _NeRFDataset
from backend.loader.decorator import KatzukiNode
from backend.nodes.builtin import BaseNode
from backend.sio import sio
from backend import variable
from KatUIDiffusionBasics.nerf import NeRFDataset
from KatUIDiffusionBasics.util import should_update

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
        return scene_codes # [B, 3, 40, 64, 64]


class RenderTriplane(BaseNode):

    @KatzukiNode(
        node_type="diffusion.triposr.render_triplane",
        signal_to_default_data={
            "break": "break",
        },
    )
    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        # clean up socket.io handlers
        if f"{self._node_id}_camera" in sio.handlers['/']:
            sio.handlers['/'].pop(f"{self._node_id}_camera")
        return super().__del__()

    def execute(
            self,
            model: TSR,
            scene_code: Tensor, # [3, 40, 64, 64]
            h_img: int = 256,
            w_img: int = 256,
            dataset: Optional[_NeRFDataset] = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ) -> None:

        render_variable = {
            "data": {
                "camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1],
                "fov": 75.0,
                "focal": 15.204296016480733,
                "near": 0.001,
                "far": 1000,
            }
        }

        def camera(render_variable: Dict):

            async def f(sid: str, data: int):
                if sid in variable.sid2uuid and variable.sid2uuid[sid] == self._uuid:
                    render_variable["data"] = data

            return f

        sio.on(f"{self._node_id}_camera", camera(render_variable=render_variable))

        with torch.no_grad():
            if dataset is None:
                dataset: _NeRFDataset = NeRFDataset().execute() # type: ignore (wait for https://peps.python.org/pep-0671/)

            render_state = {
                "start_timestamp": time.time(),
                "start_render_variable": None,
                "updated": False,
            }
            sid: str = self._sid

            def update(render_variables: Dict[str, Any], timestamp: float):
                if should_update(
                        target_fps=24,
                        start_timestamp=render_state["start_timestamp"],
                        start_render_variable=render_state["start_render_variable"],
                        sid=sid,
                        current_timestamp=timestamp,
                        current_render_variable=render_variables,
                ):
                    ray_bundle = dataset.get_eval_ray_bundle(
                        h_latent=h_img,
                        w_latent=w_img,
                        c2w=torch.tensor(render_variables["camera"], dtype=dtype, device=device).reshape(4, 4).t(),
                        fov_y=render_variables["fov"] / 180 * np.pi,
                    )
                    ray_bundle = ray_bundle.to_device(device=device, dtype=dtype)
                    rays_o = ray_bundle.origins # [B, H, W, 3]
                    rays_d = ray_bundle.directions # [B, H, W, 3]
                    assert rays_d is not None

                    image = model.render_one(
                        scene_code=scene_code,
                        rays_o=rays_o[0],
                        rays_d=rays_d[0],
                    )

                    image_buffer = io.BytesIO()
                    pil_image = PIL.Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
                    pil_image.save(image_buffer, format='PNG')

                    render_state["start_render_variable"] = render_variables
                    render_state["start_timestamp"] = timestamp

                    async def send():
                        await sio.emit(f"{self._node_id}_render", {
                            "image": f"data:image/png;base64,{base64.b64encode(image_buffer.getvalue()).decode('utf-8')}",
                        }, room=variable.uuid2sid[self._uuid])

                    asyncio.run_coroutine_threadsafe(send(), self._loop)
                else:
                    render_state["updated"] = False
                    time.sleep(1 / (24 * 2))

            step = 0
            while True:
                signal = self.check_execution_state_change()
                if signal == "break":
                    break

                self.set_output("log", f"{step}/inf")
                self.set_output("progress", int(100))
                step = step + 1
                update(render_variables=render_variable["data"], timestamp=time.time())

                if render_state["updated"]:
                    self.send_update()
                    render_state["updated"] = False
            return None


class SceneCodesToMesh(BaseNode):

    @KatzukiNode(node_type="diffusion.triposr.scene_codes_to_mesh")
    def __init__(self) -> None:
        pass

    def execute(self, model: TSR, scene_codes: Tensor, mc_resolution: int = 256) -> List[trimesh.Trimesh]:

        def correct_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
            x = mesh.vertices[:, 0]
            y = mesh.vertices[:, 1]
            z = mesh.vertices[:, 2]

            # Correct orientation
            mesh.vertices = np.stack([y, z, x], axis=1)

        with torch.no_grad():
            meshes: List[trimesh.Trimesh] = model.extract_mesh(scene_codes, resolution=mc_resolution)
            for mesh in meshes:
                correct_orientation(mesh)
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

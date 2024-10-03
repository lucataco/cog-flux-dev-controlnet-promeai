# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

MODEL_CACHE = "FLUX.1-dev"
CONTROL_CACHE = "FLUX.1-controlnet-lineart-promeai"
# BASE_MODEL = 'black-forest-labs/FLUX.1-dev'
# CONTROLNET_MODEL = 'promeai/FLUX.1-controlnet-lineart-promeai'
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
CONTROL_URL = "https://weights.replicate.delivery/default/promeai/FLUX.1-controlnet-lineart-promeai/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading Flux ControlNet")
        if not os.path.exists(CONTROL_CACHE):
            download_weights(CONTROL_URL, '.')
        self.controlnet = FluxControlNetModel.from_pretrained(
            CONTROL_CACHE,
            torch_dtype=torch.bfloat16
        )
        print("Loading Flux Pipeline")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, '.')
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=self.controlnet,
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit"),
        control_image: Path = Input(description="Grayscale input image"),
        conditioning_scale: float = Input(description="Conditioning scale", default=0.6, ge=0.1, le=1.0),
        num_inference_steps: int = Input(description="Number of inference steps", default=28, ge=1, le=50),
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0.1, le=10.0),
        seed: int = Input(description="Seed", default=None)
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        control_image = Image.open(control_image).convert("RGB").resize((1024, 1024))
        image = self.pipe(
            prompt, 
            control_image=control_image,
            controlnet_conditioning_scale=conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)

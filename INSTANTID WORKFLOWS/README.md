POINT controlnet to full path controlnet-openpose-sdxl-1.0

make sure you download controlnet into appropriate folder.
/ComfyUI/custom_nodes/ComfyUI-InstantID/checkpoints/ControlNetModel
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")
```

```shell
cd custom_nodes
gh repo clone ZHO-ZHO-ZHO/ComfyUI-ArtGallery
ComfyUI-ArtGallery
pip install -r requirements.txt
cd..
gh repo clone ZHO-ZHO-ZHO/ComfyUI-Layout-Zh-Chinese
cd ComfyUI-Layout-Zh-Chinese
pip install -r requirements.txt
cd..
gh repo clone ZHO-ZHO-ZHO/ComfyUI-Gemini
cd ComfyUI-Gemini
pip install -r requirements.txt
```

POINT controlnet to full path controlnet-openpose-sdxl-1.0


```
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
cd ..
cd models/controlnet
git lfs install
git clone https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0
```

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from .style_template import styles

import os
import cv2
import torch
import numpy as np
from PIL import Image
import folder_paths

from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from .pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Neon"


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + ' ' + negative


def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    image_np = (255. * input_image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
    input_image = Image.fromarray(image_np)

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


class InsightFaceLoader_Node_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CUDA", "CPU"], ),
            },
        }

    RETURN_TYPES = ("INSIGHTFACEMODEL",)
    FUNCTION = "load_insight_face_antelopev2"
    CATEGORY = "📷InstantID"

    def load_insight_face_antelopev2(self, provider):
            
        model = FaceAnalysis(name="antelopev2", root=current_directory, providers=[provider + 'ExecutionProvider',])
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)


class IDControlNetLoaderNode_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_path": ("STRING", {"default": "enter your path"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_idcontrolnet"
    CATEGORY = "📷InstantID"
    
    def load_idcontrolnet(self, controlnet_path):

        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

        return [controlnet]


class IDBaseModelLoader_fromhub_Node_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "wangqixun/YamerMIX_v8"}),
                "controlnet": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "📷InstantID"
  
    def load_model(self, base_model_path, controlnet):
        # Code to load the base model
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            local_dir="./checkpoints"
        ).to(device)
        return [pipe]


class IDBaseModelLoader_local_Node_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "controlnet": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "📷InstantID"
  
    def load_model(self, ckpt_name, controlnet):
        # Code to load the base model
        if not ckpt_name:
            raise ValueError("Please provide the ckpt_name parameter with the name of the checkpoint file.")

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
                
        pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
            pretrained_model_link_or_path=ckpt_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        return [pipe]


class Ipadapter_instantidLoader_Node_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Ipadapter_instantid_path": ("STRING", {"default": "enter your path"}),
                "filename": ("STRING", {"default": "ip-adapter.bin"}),
                "pipe": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_ip_adapter_instantid"
    CATEGORY = "📷InstantID"

    def load_ip_adapter_instantid(self, pipe, Ipadapter_instantid_path, filename):
        # 使用hf_hub_download方法获取PhotoMaker文件的路径
        face_adapter = os.path.join(Ipadapter_instantid_path, filename)

        # load adapter
        pipe.load_ip_adapter_instantid(face_adapter)

        return [pipe]


class ID_Prompt_Style_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "analog film photo of a woman. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured", "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME})
            }
        }

    RETURN_TYPES = ('STRING','STRING',)
    RETURN_NAMES = ('positive_prompt','negative_prompt',)
    FUNCTION = "prompt_style"
    CATEGORY = "📷InstantID"

    def prompt_style(self, style_name, prompt, negative_prompt):
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
        
        return prompt, negative_prompt


class IDGenerationNode_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_image": ("IMAGE",),
                "pipe": ("MODEL",),
                "insightface": ("INSIGHTFACEMODEL",),
                "positive": ("STRING", {"multiline": True, "forceInput": True}),
                "negative": ("STRING", {"multiline": True, "forceInput": True}),
                "ip_adapter_scale": ("FLOAT", {"default": 0.8, "min": 0, "max": 1.0, "display": "slider"}),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 0.8, "min": 0, "max": 1.0, "display": "slider"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0, "max": 10, "display": "slider"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "📷InstantID"
                       
    def generate_image(self, insightface, positive, negative, face_image, pipe, ip_adapter_scale, controlnet_conditioning_scale, steps, guidance_scale, width, height, seed):

        face_image = resize_img(face_image)
        
        # prepare face emb
        face_info = insightface.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if not face_info:
            return "No face detected"

        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])

        generator = torch.Generator(device=device).manual_seed(seed)

        pipe.set_ip_adapter_scale(ip_adapter_scale)

        output = pipe(
            prompt=positive,
            negative_prompt=negative,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            return_dict=False
            )

        # 检查输出类型并相应处理
        if isinstance(output, tuple):
            # 当返回的是元组时，第一个元素是图像列表
            images_list = output[0]
        else:
            # 如果返回的是 StableDiffusionXLPipelineOutput，需要从中提取图像
            images_list = output.images

        # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        images_tensors = []
        for img in images_list:
            # 将 PIL.Image 转换为 numpy.ndarray
            img_array = np.array(img)
            # 转换 numpy.ndarray 为 torch.Tensor
            img_tensor = torch.from_numpy(img_array).float() / 255.
            # 转换图像格式为 CHW (如果需要)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            # 添加批次维度并转换为 NHWC
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]

        return (output_image,)



NODE_CLASS_MAPPINGS = {
    "InsightFaceLoader": InsightFaceLoader_Node_Zho,
    "IDControlNetLoader": IDControlNetLoaderNode_Zho,
    "IDBaseModelLoader_fromhub": IDBaseModelLoader_fromhub_Node_Zho,
    "IDBaseModelLoader_local": IDBaseModelLoader_local_Node_Zho,
    "Ipadapter_instantidLoader": Ipadapter_instantidLoader_Node_Zho,
    "ID_Prompt_Styler": ID_Prompt_Style_Zho,
    "IDGenerationNode": IDGenerationNode_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsightFaceLoader": "📷InsightFace Loader",
    "IDControlNetLoader": "📷ID ControlNet Loader",
    "IDBaseModelLoader_fromhub": "📷ID Base Model Loader from hub 🤗",
    "IDBaseModelLoader_local": "📷ID Base Model Loader locally",
    "Ipadapter_instantidLoader": "📷Ipadapter_instantid Loader",
    "ID_Prompt_Styler": "📷ID Prompt_Styler",
    "IDGenerationNode": "📷InstantID Generation"
}


![ISID_](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/01393483-3145-4691-9daa-7ce9035c9bd0)


# ComfyUI InstantID

Unofficial implementation of [InstantID](https://github.com/InstantID/InstantID) for ComfyUI

![Dingtalk_20240123182131](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/7a99b32c-b4a2-4c46-acb0-f796fc46f9ee)

+ pose_ref

![Dingtalk_20240124232946](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/caa60456-f2d8-4315-864b-659a9e7cea89)


## 项目介绍 | Info

- 来自对[InstantID](https://github.com/InstantID/InstantID)的非官方实现
  
- 版本：V2.0 支持姿势参考图

<!---
  同时支持本地、huggingface hub模型，支持通用styler（也与 PhotoMaker Styler 通用）
--->

## 视频演示

V2.0


https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/083c9e5e-06a0-4623-b5ac-05f7e85a74f2


V1.0

https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/7295c0d7-1d1b-4044-aea3-8efa67047362



## 节点说明 | Features

- 基础模型加载 | base model loader
    - 📷ID Base Model Loader from hub 🤗：支持从 huggingface hub 自动下载模型，输入模型名称（如：wangqixun/YamerMIX_v8）即可
    - 📷ID Base Model Loader locally：支持加载本地模型（需 SDXL 系列模型）

- InsightFace 模型加载 | 📷InsightFace Loader
    - ：支持 CUDA 和 CPU

- ID ControlNet 模型加载 | 📷ID ControlNet Loader
    - controlnet_path：ID ControlNet 模型地址

- Ipadapter_instantid 模型加载 | 📷Ipadapter_instantid Loader
    - Ipadapter_instantid_path：模型路径
    - filename：模型名称

 - 提示词 + 风格 | 📷ID Prompt_Styler
    - 与各种提示词（文本）输入（如肖像大师等）、styler、 Photomaker Prompt_Styler 兼容
    - prompt、negative：正负提示词
    - style_name：支持官方提供的8种风格
        - (No style)
        - Watercolor
        - Film Noir
        - Neon
        - Jungle
        - Mars
        - Vibrant Color
        - Snow
        - Line art

- InstantID 生成 | 📷InstantID Generation 🆕
    - face_image：接入脸部参考图像
    - pipe：接入模型
    - insightface：接入 insightface 模型 🆕
    - pose_image_optional（非必要）：接入姿势参考图像（注意：仅对面部周围姿势起效，与通常的 openpose 不同）
    - positivet、negative：正负提示词
    - ip_adapter_scale：IPA 强度
    - controlnet_conditioning_scale：ID Controlnet 强度
    - step：步数，官方默认30步
    - guidance_scale：提示词相关度，一般默认为5
    - enhance_face_region：脸部增强选项 🆕
    - seed：种子


## 风格 | Styles

![ISID_STYLE](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/142bda7a-798b-46b3-aa69-1b88701c8311)



## 安装 | Install


- 推荐使用管理器 ComfyUI Manager 安装（On the Way）


- 手动安装：
    1. `cd custom_nodes`
    2. `git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID.git`
    3. `cd custom_nodes/ComfyUI-InstantID`
    4. `pip install -r requirements.txt`
    5. 重启 ComfyUI


## 使用方法 | How to Use

- 下载 [InstantID/ControlNetModel](https://huggingface.co/InstantX/InstantID/tree/main/ControlNetModel) 中的 config.json 和 diffusion_pytorch_model.safetensors ，将模型地址填入 📷ID ControlNet Loader 节点中（例如：ComfyUI/custom_nodes/ComfyUI-InstantID/checkpoints/controlnet）

- 下载 [InstantID/ip-adapter](https://huggingface.co/InstantX/InstantID/tree/main) 中的 ip-adapter.bin ，将其地址填入 📷Ipadapter_instantid Loader 节点中（例如：ComfyUI/custom_nodes/ComfyUI-InstantID/checkpoints）

- 下载 [DIAMONIK7777/antelopev2](https://huggingface.co/DIAMONIK7777/antelopev2/tree/main) 中的所有模型，将其放入 ComfyUI//custom_nodes/ComfyUI-InstantID/models/antelopev2 中

- 兼容性： CUDA11 支持默认安装的 onnxruntime-gpu（1.16.0），如果是 CUDA12 则需手动安装 onnxruntime-gpu==1.17.0 [地址](https://dev.azure.com/onnxruntime/onnxruntime/_artifacts/feed/onnxruntime-cuda-12/PyPI/onnxruntime-gpu/overview/1.17.0)

  
## 工作流 | Workflows

V2.0

- [V2.0 InstantID_pose_ref + ArtGallery](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/blob/main/INSTANTID%20WORKFLOWS/V2.0%20InstantID_pose_ref%20%2B%20ArtGallery%20%E3%80%90Zho%E3%80%91.json)

  ![Dingtalk_20240124232833](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/99be9592-775d-4c33-bafc-5bd5c95a7222)


- [V2.0 自动下载 huggingface hub](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/blob/main/INSTANTID%20WORKFLOWS/V2.0%20InstantID_fromhub_pose_ref%E3%80%90Zho%E3%80%91.json)

  ![Dingtalk_20240124230145](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/95c4a1dd-864d-4a46-8c45-a48866aef29f)


- [V2.0 InstantID_locally_pose_ref](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/blob/main/INSTANTID%20WORKFLOWS/V2.0%20InstantID_locally_pose_ref%E3%80%90Zho%E3%80%91.json)

  ![Dingtalk_20240124230609](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/d4c22389-f853-44bd-9ea2-568b2ac7ed06)


V1.0 工作流仅适用于V1.0 版本

- [V1.0  InstantID + ArtGallery](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/blob/main/INSTANTID%20WORKFLOWS/V1.0%20InstantID%20%2B%20ArtGallery%E3%80%90Zho%E3%80%91.json)


  ![Dingtalk_20240123182440](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/c6ee25bf-a528-4d78-9b35-f5b0d0303601)


- [V1.0 本地模型 locally](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/blob/main/INSTANTID%20WORKFLOWS/V1.0%20InstantID_locally%E3%80%90Zho%E3%80%91.json)

  ![Dingtalk_20240123175624](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/459bfede-59e8-4d8d-941c-a950c4827c49)


- [V1.0 自动下载 huggingface hub](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/blob/main/INSTANTID%20WORKFLOWS/V1.0%20InstantID_fromhub%E3%80%90Zho%E3%80%91.json)

  ![Dingtalk_20240123174950](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/50133961-1752-4ec8-ac0b-068d998b8534)




## 更新日志

- 20240124

  更新为 V2.0 ：新增姿势参考图、优化代码

  修复 insightfaceloader 冲突问题

  修复 onnxruntime-gpu 版本兼容性的问题

- 20240123

  V1.0 上线：同时支持本地、huggingface hub托管模型，支持8种风格

- 20240122

  创建项目


## 速度实测 | Speed

- V1.0 

    - A100 50步 14s

    ![image](https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID/assets/140084057/dc535e67-3f56-4faf-be81-621b84bb6ee2)



## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/ComfyUI-InstantID&type=Date)](https://star-history.com/#ZHO-ZHO-ZHO/ComfyUI-InstantID&Date)


## 关于我 | About me

📬 **联系我**：
- 邮箱：zhozho3965@gmail.com
- QQ 群：839821928

🔗 **社交媒体**：
- 个人页：[-Zho-](https://jike.city/zho)
- Bilibili：[我的B站主页](https://space.bilibili.com/484366804)
- X（Twitter）：[我的Twitter](https://twitter.com/ZHOZHO672070)
- 小红书：[我的小红书主页](https://www.xiaohongshu.com/user/profile/63f11530000000001001e0c8?xhsshare=CopyLink&appuid=63f11530000000001001e0c8&apptime=1690528872)

💡 **支持我**：
- B站：[B站充电](https://space.bilibili.com/484366804)
- 爱发电：[为我充电](https://afdian.net/a/ZHOZHO)


## Credits

[InstantID](https://github.com/InstantID/InstantID)

📷InsightFace Loader 代码修改自 [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)，感谢 [@cubiq](https://github.com/cubiq)！

感谢 [@hidecloud](https://twitter.com/hidecloud) 对 onnxruntime 版本兼容性的测试与反馈！

感谢 [esheep](https://www.esheep.com/) 技术人员对节点冲突问题的反馈！

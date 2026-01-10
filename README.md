# Some-Changes-on-GeCo

GeCo（[原仓库](https://github.com/jerpelhan/GeCo)）是一个强大的少样本目标计数与检测模型。  
本仓库在 GeCo 基础上进行以下改动：

1. 新增**跨图搜索**能力  
2. 将 backbone 替换为 [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)，性能提升  
3. 未完待续…

---

## 环境
- Python 3.10  
- CUDA 12.1  
- torch、torchvision、torchaudio  
- matplotlib  

---

## 权重准备
| 文件 | 放置路径 | 下载地址 |
|---|---|---|
| `GeCo.pth` | 仓库根目录 | [Google Drive](https://drive.google.com/file/d/1wjOF9MWkrVJVo5uG3gVqZEW9pwRq_aIk/view) |
| `sam_vit_h_4b8939.pth` | 仓库根目录 | [SAM 官方](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `efficientvit_sam_l1.pt` | `Some-Changes-on-GeCo/third_party/efficientvit/assets/checkpoints/efficientvit_sam/` | [HuggingFace](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l1.pt) |

---

## 使用说明

### 跨图搜索
1. 确保已下载 `GeCo.pth` 与 `sam_vit_h_4b8939.pth`  
2. 运行根目录 `demo_cross.py`  
3. 在弹出窗口中选择**支持图** → 框选**示例框** → 关闭窗口 → 查看结果

### EfficientViT-SAM 改进
1. 确保已下载 `GeCo.pth` 与 `efficientvit_sam_l1.pt`  
2. 运行根目录 `efficientvitsam_demo.py`  
3. 选择图片 → 框选示例 → 关闭窗口 → 查看结果

<!-- 第一组 -->
<p align="center">
  <img src="demo_pic/efficientvitgeco_result/1.png" width="35%">
&nbsp; &nbsp; &nbsp;
  <img src="demo_pic/geco_result/1.png" width="35%">
</p>
<p align="center"><em>图 1  左=eff_vit_sam(3717.88 ms模型加载+2102.95模型推理)，右=sam（4961.88 ms+2659.98 ms） </em></p>

<!-- 第一组 -->
<p align="center">
  <img src="demo_pic/efficientvitgeco_result/2.png" width="35%">
&nbsp; &nbsp; &nbsp;
  <img src="demo_pic/geco_result/2.png" width="35%">
</p>
<p align="center"><em>图 1  左=eff_vit_sam（4044.56 ms模型加载+2075.46 ms模型推理），右=sam（4696.03 ms+3055.34 ms） </em></p>

## 运行环境  
设备：i9-13900HX + RTX 4060 Laptop + Windows 11  

### 关于 GeCo 跨图搜索  
对 support_img、query_img、support_box 均使用 backbone 提取特征，再将 Prototype Embeddings 注入查询图特征。  
代码参考：demo_cross.py、geco_infer.py（新增 forward_cross 函数）。  
![cross_img](pics/crossmodel.jpg)  

### 关于将 efficient_vit_sam 替换 SAM  
新增 eefficientvitsam_geco_infer.py，结构基本照抄原 geco_infer.py，仅替换 import 以使用新 backbone。  
新增 efficientvitsam_demo.py，逻辑参考原 demo.py，但 build_model 改为从 models/eefficientvitsam_geco_infer 导入。  

因工程最初不含 EfficientViT-SAM，需将官方仓库置为第三方依赖：  
官方仓库 mit-han-lab/efficientvit → 放置路径 third_party/efficientvit  

backbone 侧：EfficientViT-SAM 的 image_encoder 使用 512 输入及专用 mean/std，而 GeCo 原默认 1024+ImageNet 参数。  
在 models/eeefficientvitsam_geco_infer.py 的 EfficientViTSAMBackbone.forward() 中做输入适配：  
先去 ImageNet 归一化回 [0,1]，resize 到 512，再用 EfficientViT-SAM 的 mean/std 重新归一化后送入 model.image_encoder。  

原 GeCo 结构里除 backbone 外，还有用 SAM 的 prompt_encoder+mask_decoder 进行 refine_bounding_boxes 的后处理阶段，该阶段原代码硬编码加载 sam_vit_h_4b8939.pth。  
在 models/eefficientvitsam_geco_infer.py 中将 refine 设为可选：若找不到该权重则自动跳过 refine（输出更粗但不报错），并在 efficientvitsam_demo.py 提供参数 --disable_sam_refine 与 --sam_refine_ckpt 供选择。  
代码参考：models/eefficientvitsam_geco_infer.py、efficientvitsam_demo.py  

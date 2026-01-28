# GeCo_CrossSearch_EfficientVitSam

GeCo（[原仓库](https://github.com/jerpelhan/GeCo)）是一个强大的少样本目标计数与检测模型。  
本仓库在 GeCo 基础上进行以下改动：

新增**跨图搜索**能力  
将 backbone 替换为 [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)，速度提升  

---

## 环境
- Python 3.10  
- CUDA 12.1  
- torch 2.5.1、torchvision、torchaudio  
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
![cross_img](demo_pic/crossmodel.jpg)  

### ⚡️ 关于将 SAM 替换为 EfficientViT-SAM

#### 1. 性能变更 (Performance)

* **精度/速度权衡**：修改后的模型推理速度显著上升，但精度略有下降。
* **后续计划**：后续将在其他类型的数据集以及 EfficientViT-SAM 的其他权重版本上进行进一步的对比测试。

#### 2. 代码变更 (Code Changes)

* **新增文件**：
* `efficientvitsam_geco_infer.py`：结构源自原 `geco_infer.py`，主要修改了 import 部分以适配新 Backbone。
* `efficientvitsam_demo.py`：逻辑参考原 `demo.py`，将 `build_model` 指向新的 `models/efficientvitsam_geco_infer`。


* **第三方依赖**：
* 因工程初始不包含 EfficientViT-SAM，需引入官方仓库作为依赖：
* `mit-han-lab/efficientvit` → `third_party/efficientvit`



#### 3. 核心实现细节 (Implementation Details)

**Backbone 输入适配**
GeCo 默认使用 1024 分辨率及 ImageNet 归一化参数，而 EfficientViT-SAM 的 `image_encoder` 需要 512 输入及专用 mean/std。

* **适配逻辑**：在 `models/efficientvitsam_geco_infer.py` 的 `EfficientViTSAMBackbone.forward()` 中：
1. 先将输入反归一化回 `[0,1]` 并 Resize 到 512；
2. 使用 EfficientViT-SAM 专用的 mean/std 重新归一化后送入模型。



**Refine 阶段优化**
原 GeCo 包含基于 SAM (`prompt_encoder` + `mask_decoder`) 的 `refine_bounding_boxes` 后处理阶段，且原代码硬编码加载 `sam_vit_h_4b8939.pth`。

* **改进**：在 `efficientvitsam_geco_infer.py` 中将 Refine 设为可选模块。
* **自动跳过**：若找不到权重文件，则自动跳过 Refine 阶段（仅输出粗略结果，不报错）。
* **参数控制**：`efficientvitsam_demo.py` 新增参数 `--disable_sam_refine` 与 `--sam_refine_ckpt` 以支持灵活配置。

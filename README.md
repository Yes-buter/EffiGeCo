# EffiGeCo

这是一个基于 [GeCo](https://github.com/jerpelhan/GeCo) (Generative Count) 的改进版本，主要针对**跨图检索能力**与**推理速度**进行了优化。

### 🚀 主要特性

1. **新增跨图搜索 (Cross-Image Search)**：支持在一张图片中框选目标，在另一张完全不同的图片中搜索并计数同类目标。
2. **引入 EfficientViT-SAM**：将原版的 SAM Backbone 替换为 [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit)，在保持可用精度的前提下显著提升推理速度。

---

## 🛠️ 环境依赖 (Environment)

* **OS**: Windows 11 (Tested) / Linux
* **Python**: 3.10
* **CUDA**: 12.1
* **Core Libraries**:
* `torch==2.5.1`, `torchvision`, `torchaudio`
* `matplotlib`



> **测试硬件**: Intel i9-13900HX + NVIDIA RTX 4060 Laptop

---

## 📥 权重准备 (Model Zoo)

请下载以下权重文件并放置在指定目录：

| 模型文件 | 存放路径 | 下载地址 | 说明 |
| --- | --- | --- | --- |
| `GeCo.pth` | `./` (项目根目录) | [Google Drive](https://drive.google.com/file/d/1wjOF9MWkrVJVo5uG3gVqZEW9pwRq_aIk/view) | GeCo 原始权重 |
| `sam_vit_h_4b8939.pth` | `./` (项目根目录) | [SAM 官方](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | 用于 Refine 阶段 (可选) |
| `efficientvit_sam_l1.pt` | `third_party/efficientvit/assets/checkpoints/efficientvit_sam/` | [HuggingFace](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l1.pt) | 新 Backbone 权重 |

---

## 💻 快速开始 (Quick Start)

### 1. 跨图搜索 (Cross-Image Search)

利用 GeCo 的特征提取能力进行跨图像的目标定位。

```bash
python demo_cross.py

```

**操作流程**: 运行脚本 → 在弹窗中选择**支持图 (Query)** → 框选目标物体 → 关闭窗口 → 查看结果。

### 2. EfficientViT-SAM 加速推理

体验替换 Backbone 后的高速推理版本。

```bash
python efficientvitsam_demo.py
# 可选参数：禁用 SAM Refine 以进一步提速
# python efficientvitsam_demo.py --disable_sam_refine

```

**操作流程**: 运行脚本 → 选择图片 → 框选示例 → 关闭窗口 → 查看结果。

---

## 📊 效果与性能对比 (Benchmark)

我们在 RTX 4060 Laptop 上进行了对比测试，EfficientViT-SAM 版本在模型加载和推理阶段均有显著提速。

| 测试样本 | 模型版本 | 加载耗时 (ms) | 推理耗时 (ms) | 总耗时 (ms) | 速度提升 (推理) |
| --- | --- | --- | --- | --- | --- |
| **Sample 1** | Original SAM | 4961.88 | 2659.98 | 7621.86 | - |
|  | **EfficientViT** | **3717.88** | **2102.95** | **5820.83** | **+21%** 🚀 |
| **Sample 2** | Original SAM | 4696.03 | 3055.34 | 7751.37 | - |
|  | **EfficientViT** | **4044.56** | **2075.46** | **6120.02** | **+32%** 🚀 |

### 可视化结果

<p align="center">
<img src="demo_pic/efficientvitgeco_result/1.png" width="45%">
&nbsp;
<img src="demo_pic/geco_result/1.png" width="45%">





<em>图 1: Sample 1 效果对比 (左: EfficientViT-SAM, 右: Original SAM)</em>
</p>

<p align="center">
<img src="demo_pic/efficientvitgeco_result/2.png" width="45%">
&nbsp;
<img src="demo_pic/geco_result/2.png" width="45%">





<em>图 2: Sample 2 效果对比 (左: EfficientViT-SAM, 右: Original SAM)</em>
</p>

---

## 📝 技术细节 (Implementation Details)

### 1. 跨图搜索实现 (Cross-Image Search)

原理是对 `support_img` (查询图)、`query_img` (被搜图) 和 `support_box` 均使用 Backbone 提取特征，然后将 Prototype Embeddings 注入到查询图特征中。

* **核心代码**: `models/geco_infer.py` (新增 `forward_cross` 函数)
* **架构示意**:

### 2. EfficientViT-SAM 替换方案

我们移除了沉重的 ViT-H Backbone，改用轻量级的 EfficientViT-SAM。

#### A. 性能与策略

* **精度/速度权衡**：推理速度显著上升，精度略有下降。
* **第三方依赖**：引入 `mit-han-lab/efficientvit` 到 `third_party/efficientvit` 目录。

#### B. 核心代码变更

* **`efficientvitsam_geco_infer.py`**: 复用原 `geco_infer.py` 结构，修改 Import 与 Backbone 调用。
* **`efficientvitsam_demo.py`**: 适配新的模型构建逻辑。

#### C. 关键技术点：Backbone 输入适配

GeCo 默认使用 1024 分辨率及 ImageNet 归一化，而 EfficientViT-SAM 需要 512 分辨率及专用 Mean/Std。我们在 `EfficientViTSAMBackbone.forward()` 中实现了动态适配：

1. **反归一化**: 将输入 Tensor 还原回 `[0,1]`。
2. **Resize**: 调整分辨率至 `512x512`。
3. **重归一化**: 使用 EfficientViT 专用的 Mean/Std 进行标准化，最后送入 `image_encoder`。

#### D. Refine 阶段的优化

原 GeCo 强制加载 SAM 权重进行 Refine。为了解耦，我们将此阶段设为**可选**：

* 若未检测到 `sam_vit_h_4b8939.pth`，代码会自动跳过 Refine 阶段（不报错）。
* 提供 `--disable_sam_refine` 参数供用户手动关闭该阶段以追求极致速度。

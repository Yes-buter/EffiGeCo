# Some-Changes-on-GeCo
GeCo(https://github.com/jerpelhan/GeCo)是一个强大的少样本目标计数与检测模型
对GeCo进行修改，使其具备
1.跨图搜索的能力
2.将backbone改为efficientvitsam (https://github.com/mit-han-lab/efficientvit) 性能提升 
3.未完待续

python 环境:
cuda12.1
torch torchvision torchaudio 
matplotlib

权重文件准备
根目录下放置GeCo.pth：
https://drive.google.com/file/d/1wjOF9MWkrVJVo5uG3gVqZEW9pwRq_aIk/view
根目录下放置sam_vit_h_4b8939.pth
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
Some-Changes-on-GeCo\third_party\efficientvit\assets\checkpoints\efficientvit_sam目录下放置efficientvit_sam_l1.pt：
https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l1.pt

跨图搜索使用：
运行根目录下demo_cross.py,选择support_img（支持图），在图上框选出support_box（示例），关闭窗口，弹出结果
跨图只需要权重GeCo.pth和sam_vit_h_4b8939.pth

efficientvitsma改进：
运行根目录下efficientvitsam_demo.py，选择图片，在图上框选出示例，关闭窗口，弹出结果
跨图只需要权重GeCo.pth和efficientvit_sam_l1.pt


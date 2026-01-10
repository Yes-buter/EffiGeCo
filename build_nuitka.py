import os
import subprocess
import sys

def build():
    # 检查是否安装了 Nuitka 和 pefile
    try:
        import nuitka
        import pefile
    except ImportError:
        print("Required packages not found. Installing Nuitka, zstandard, and pefile...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka", "zstandard", "pefile"])
        except subprocess.CalledProcessError:
            print("Failed to install packages. Please install: pip install nuitka zstandard pefile")
            return

    main_script = "demo_cross.py"
    
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found!")
        return

    output_dir = "dist"

    # 设置 matplotlib 后端环境变量，防止 Nuitka 插件检测失败
    os.environ["MPLBACKEND"] = "TkAgg"

    # Nuitka 命令构建
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--show-progress",
        "--show-memory",
        f"--output-dir={output_dir}",
        
        # 包含项目自身的包
        "--include-package=models",
        "--include-package=utils",
        "--include-package=segment_anything",
        
        # 包含必要的第三方包
        "--include-package=torch",
        "--include-package=torchvision",
        "--include-package=PIL",
        "--include-package=matplotlib",
        "--include-package=numpy",

        # 启用插件
        "--enable-plugin=numpy",
        "--enable-plugin=torch",
        # 启用 tk-inter 插件以支持 matplotlib 的 GUI 显示 (Windows 默认通常是 TkAgg)
        "--enable-plugin=tk-inter", 
        
        # 尝试强制使用 MSVC (Visual Studio)
        "--msvc=latest",
        #"--assume-yes-for-downloads",
        # 使用 pefile 进行更精确的 Windows 依赖项分析 (修复 DLL 入口点丢失问题)
        "--windows-dependency-tool=pefile",        
        # 禁用不需要的插件以减少干扰 (可选)
        "--disable-plugin=anti-bloat",
        
        # Windows 特定选项
        "--windows-console-mode=force", # 强制显示控制台，方便查看报错和日志
        
        main_script
    ]

    print("=" * 60)
    print("Starting Nuitka build process...")
    print(f"Target Script: {main_script}")
    print(f"Output Directory: {os.path.abspath(output_dir)}")
    print("=" * 60)
    print("Running command:")
    print(" ".join(cmd))
    print("-" * 60)
    
    try:
        subprocess.check_call(cmd)
        print("=" * 60)
        print("Build completed successfully!")
        exe_path = os.path.join(output_dir, "demo_cross.exe")
        print(f"Executable generated at: {os.path.abspath(exe_path)}")
        print("\nUsage example:")
        print(f'{exe_path} --support_image_path "1.png" --query_image_path "2.png" --model_path GeCo.pth')
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"Build failed with error code {e.returncode}")
        print("Please check the error messages above.")
        print("=" * 60)

if __name__ == "__main__":
    build()

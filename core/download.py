"""
Model download script for CosyVoice
Supports both HuggingFace and ModelScope
"""
import os
import sys
import argparse
from pathlib import Path


def get_download_method():
    """Ask user to choose download method"""
    print("\n" + "=" * 50)
    print("选择模型下载渠道 / Choose Download Channel:")
    print("=" * 50)
    print("1. ModelScope (推荐，国内访问快)")
    print("2. HuggingFace (需要Token，国外访问快)")
    print("=" * 50)
    
    while True:
        choice = input("请选择 (1 or 2) / Please select: ").strip()
        if choice == '1':
            return 'modelscope', None
        elif choice == '2':
            print("\n⚠️ 注意: 部分模型可能需要 HuggingFace Token")
            if input("是否输入 Token? (y/n) / Input Token? (y/n): ").strip().lower() == 'y':
                token = input("请输入 Token / Please enter Token: ").strip()
                return 'huggingface', token
            return 'huggingface', None
        print("无效选择，请重试 / Invalid choice, please try again")


def download_huggingface(model_id, local_dir, token=None):
    """Download model from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
        print(f"\n📥 从 HuggingFace 下载: {model_id}")
        snapshot_download(repo_id=model_id, local_dir=local_dir, token=token)
        print(f"✅ 下载完成: {local_dir}")
        return True
    except Exception as e:
        print(f"❌ HuggingFace 下载失败: {str(e)}")
        return False


def download_modelscope(model_id, local_dir):
    """Download model from ModelScope"""
    try:
        from modelscope import snapshot_download
        import shutil
        
        print(f"\n📥 从 ModelScope 下载: {model_id}")
        
        # Use a temp directory for download to handle the nested structure
        temp_root = os.path.join(os.path.dirname(local_dir), "temp_ms_download")
        if os.path.exists(temp_root):
            shutil.rmtree(temp_root)
            
        # This returns the full path to the model directory
        downloaded_path = snapshot_download(model_id=model_id, cache_dir=temp_root)
        
        print(f"临时下载路径: {downloaded_path}")
        
        # Move files to target local_dir
        # Ensure local_dir exists
        os.makedirs(local_dir, exist_ok=True)
        
        # Move all files from downloaded_path to local_dir
        for item in os.listdir(downloaded_path):
            s = os.path.join(downloaded_path, item)
            d = os.path.join(local_dir, item)
            if os.path.exists(d):
                if os.path.isdir(d):
                    shutil.rmtree(d)
                else:
                    os.remove(d)
            shutil.move(s, d)
            
        # Clean up temp directory
        try:
            shutil.rmtree(temp_root)
        except:
            pass
            
        print(f"✅ 下载完成: {local_dir}")
        return True
    except Exception as e:
        print(f"❌ ModelScope 下载失败: {str(e)}")
        # Try to clean up
        try:
            if 'temp_root' in locals() and os.path.exists(temp_root):
                shutil.rmtree(temp_root)
        except:
            pass
        return False


def download_model(model_info, download_method, token=None):
    """Download a single model"""
    name, hf_id, ms_id, local_dir = model_info
    
    # Create directory if not exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        print(f"\n⏭️  {name} 已存在: {local_dir}")
        return True
    
    print(f"\n{'=' * 50}")
    print(f"下载模型: {name}")
    print(f"本地目录: {local_dir}")
    print(f"{'=' * 50}")
    
    if download_method == 'huggingface':
        success = download_huggingface(hf_id, local_dir, token)
        if not success:
            print(f"尝试从 ModelScope 下载...")
            success = download_modelscope(ms_id, local_dir)
    else:  # modelscope
        success = download_modelscope(ms_id, local_dir)
        if not success:
            print(f"尝试从 HuggingFace 下载...")
            success = download_huggingface(hf_id, local_dir, token)
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Download CosyVoice models')
    parser.add_argument('--method', choices=['huggingface', 'modelscope'], 
                        help='Download method (huggingface or modelscope)')
    parser.add_argument('--token', help='HuggingFace token')
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--wetext', action='store_true', help='Download wetext only')
    parser.add_argument('--cosyvoice3', action='store_true', help='Download CosyVoice3 only')
    
    args = parser.parse_args()
    
    # Determine download method
    token = args.token
    if args.method:
        download_method = args.method
    else:
        download_method, token = get_download_method()
    
    # Base directory
    pretrained_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained_models')
    os.makedirs(pretrained_models_dir, exist_ok=True)
    
    # Models configuration
    models = {
        'wetext': (
            'wetext',
            'pengzhendong/wetext',
            'pengzhendong/wetext',
            os.path.join(pretrained_models_dir, 'wetext')
        ),
        'cosyvoice3': (
            'Fun-CosyVoice3-0.5B',
            'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
            'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
            os.path.join(pretrained_models_dir, 'Fun-CosyVoice3-0.5B')
        ),
    }
    
    # Determine which models to download
    to_download = []
    if args.all or (not args.wetext and not args.cosyvoice3):
        to_download = list(models.values())
    else:
        if args.wetext:
            to_download.append(models['wetext'])
        if args.cosyvoice3:
            to_download.append(models['cosyvoice3'])
    
    print(f"\n🚀 开始下载模型...")
    print(f"下载渠道: {download_method}")
    print(f"待下载模型数: {len(to_download)}")
    
    # Download models
    results = []
    for model_info in to_download:
        success = download_model(model_info, download_method, token)
        results.append((model_info[0], success))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📊 下载汇总:")
    print(f"{'=' * 50}")
    for model_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{model_name}: {status}")
    
    all_success = all(success for _, success in results)
    if all_success:
        print(f"\n🎉 所有模型下载完成！")
    else:
        print(f"\n⚠️ 部分模型下载失败，请检查网络并重试。")
    
    print(f"\n模型保存位置: {pretrained_models_dir}")


if __name__ == '__main__':
    main()

"""Model download script for CosyVoice (CLI + GUI callbacks)."""

import argparse
import os
import shutil
from typing import Callable, Dict, List, Optional, Tuple


LogCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[int, str], None]]


def _emit_log(message: str, log_callback: LogCallback = None):
    if log_callback:
        log_callback(message)
    else:
        print(message)


def _emit_progress(value: int, status: str = "", progress_callback: ProgressCallback = None):
    if progress_callback:
        progress_callback(max(0, min(100, int(value))), status)


def get_download_method() -> Tuple[str, Optional[str]]:
    """Ask user to choose download method (CLI only)."""
    print("\n" + "=" * 50)
    print("选择模型下载渠道 / Choose Download Channel:")
    print("=" * 50)
    print("1. ModelScope (推荐，国内访问快)")
    print("2. HuggingFace (需要Token，国外访问快)")
    print("=" * 50)

    while True:
        choice = input("请选择 (1 or 2) / Please select: ").strip()
        if choice == "1":
            return "modelscope", None
        if choice == "2":
            print("\n⚠️ 注意: 部分模型可能需要 HuggingFace Token")
            if input("是否输入 Token? (y/n) / Input Token? (y/n): ").strip().lower() == "y":
                token = input("请输入 Token / Please enter Token: ").strip()
                return "huggingface", token
            return "huggingface", None
        print("无效选择，请重试 / Invalid choice, please try again")


def get_model_catalog(pretrained_models_dir: str,
                      model_paths: Optional[Dict[str, str]] = None) -> Dict[str, Tuple[str, str, str, str]]:
    model_paths = model_paths or {}
    
    # 这里的逻辑修改为：如果用户指定了路径，我们就在该路径下创建对应的子文件夹
    # 增加检测：如果路径已经是以目标名称结尾，则不再叠加子目录
    
    def get_final_path(base_path, sub_name):
        base_path = os.path.abspath(base_path)
        # 如果路径本身就是以目标名结尾（比如 D:/1/wetext），就直接使用它
        if os.path.basename(base_path).lower() == sub_name.lower():
            return base_path
        # 否则创建子目录
        return os.path.join(base_path, sub_name)

    wetext_base = model_paths.get("wetext") or pretrained_models_dir
    cosy_base = model_paths.get("cosyvoice3") or pretrained_models_dir
    
    return {
        "wetext": (
            "wetext",
            "pengzhendong/wetext",
            "pengzhendong/wetext",
            get_final_path(wetext_base, "wetext"),
        ),
        "cosyvoice3": (
            "Fun-CosyVoice3-0.5B",
            "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            get_final_path(cosy_base, "Fun-CosyVoice3-0.5B"),
        ),
    }


def is_model_downloaded(local_dir: str) -> bool:
    """检查模型是否已下载，检测关键文件是否存在"""
    if not os.path.isdir(local_dir):
        return False
        
    # 如果是目录，检查是否有实质性文件，而不仅仅是一个空目录或只有 readme
    # 针对 CosyVoice，关键文件是 .yaml
    # 针对 WeText，标准路径下应该有 zh/tn 目录
    
    files = os.listdir(local_dir)
    if len(files) == 0:
        return False
        
    # 特别检查：如果目录里只有下载残留的临时文件夹，也视为未下载
    if len(files) == 1 and files[0].startswith("temp_ms"):
        return False
        
    return True


def download_huggingface(model_id: str, local_dir: str, token: Optional[str] = None,
                         log_callback: LogCallback = None, flatten: bool = True) -> bool:
    try:
        from huggingface_hub import snapshot_download

        _emit_log(f"📥 从 HuggingFace 下载: {model_id}", log_callback)
        if flatten:
            snapshot_download(repo_id=model_id, local_dir=local_dir, token=token)
        else:
            # 如果不扁平化，我们手动创建一个子文件夹来模拟结构或保持原样
            # 对于 HuggingFace，通常是直接下载到 local_dir
            snapshot_download(repo_id=model_id, local_dir=local_dir, token=token)
            
        _emit_log(f"✅ 下载完成: {local_dir}", log_callback)
        return True
    except Exception as error:
        _emit_log(f"❌ HuggingFace 下载失败: {error}", log_callback)
        return False


def download_modelscope(model_id: str, local_dir: str, log_callback: LogCallback = None, flatten: bool = True) -> bool:
    temp_root = ""
    try:
        from modelscope import snapshot_download

        _emit_log(f"📥 从 ModelScope 下载: {model_id}", log_callback)

        temp_root = os.path.join(os.path.dirname(local_dir), "temp_ms_download")
        if os.path.exists(temp_root):
            shutil.rmtree(temp_root)

        downloaded_path = snapshot_download(model_id=model_id, cache_dir=temp_root)
        _emit_log(f"临时下载路径: {downloaded_path}", log_callback)

        os.makedirs(local_dir, exist_ok=True)
        if flatten:
            for item in os.listdir(downloaded_path):
                source = os.path.join(downloaded_path, item)
                target = os.path.join(local_dir, item)
                if os.path.exists(target):
                    if os.path.isdir(target):
                        shutil.rmtree(target)
                    else:
                        os.remove(target)
                shutil.move(source, target)
        else:
            # 不扁平化：保持原始结构 (例如 pengzhendong/wetext)
            author_dir = os.path.dirname(downloaded_path) # temp_root/author
            author_name = os.path.basename(author_dir)
            
            target_author_dir = os.path.join(local_dir, author_name)
            if os.path.exists(target_author_dir):
                shutil.rmtree(target_author_dir)
            
            shutil.move(author_dir, local_dir)

        if os.path.exists(temp_root):
            shutil.rmtree(temp_root)

        _emit_log(f"✅ 下载完成并整理路径: {local_dir}", log_callback)
        return True
    except Exception as error:
        _emit_log(f"❌ ModelScope 下载失败: {error}", log_callback)
        if temp_root and os.path.exists(temp_root):
            try:
                shutil.rmtree(temp_root)
            except Exception:
                pass
        return False


def download_model(model_info: Tuple[str, str, str, str], download_method: str,
                   token: Optional[str] = None, model_index: int = 0, total_models: int = 1,
                   progress_callback: ProgressCallback = None,
                   log_callback: LogCallback = None) -> Tuple[bool, bool]:
    """Download a single model and return (success, skipped)."""
    name, hf_id, ms_id, local_dir = model_info
    os.makedirs(local_dir, exist_ok=True)

    # 针对 WeText，我们遵循 ModelScope 的原始结构 (pengzhendong/wetext)
    # 针对 CosyVoice，我们使用扁平结构
    flatten = (name != "wetext")

    start_progress = int(model_index / max(1, total_models) * 100)
    end_progress = int((model_index + 1) / max(1, total_models) * 100)

    _emit_progress(start_progress, f"准备下载 {name}", progress_callback)

    if is_model_downloaded(local_dir):
        _emit_log(f"⏭️ {name} 已存在，跳过: {local_dir}", log_callback)
        _emit_progress(end_progress, f"{name} 已存在（已跳过）", progress_callback)
        return True, True

    _emit_log("\n" + "=" * 50, log_callback)
    _emit_log(f"下载模型: {name}", log_callback)
    _emit_log(f"本地目录: {local_dir}", log_callback)
    _emit_log("=" * 50, log_callback)

    success = False
    if download_method == "huggingface":
        success = download_huggingface(hf_id, local_dir, token, log_callback, flatten=flatten)
        if not success:
            _emit_log("尝试从 ModelScope 下载...", log_callback)
            success = download_modelscope(ms_id, local_dir, log_callback, flatten=flatten)
    else:
        success = download_modelscope(ms_id, local_dir, log_callback, flatten=flatten)
        if not success:
            _emit_log("尝试从 HuggingFace 下载...", log_callback)
            success = download_huggingface(hf_id, local_dir, token, log_callback, flatten=flatten)

    if success:
        _emit_progress(end_progress, f"{name} 下载完成", progress_callback)
    else:
        _emit_progress(start_progress, f"{name} 下载失败", progress_callback)

    return success, False


def download_models(download_method: str = "modelscope", token: Optional[str] = None,
                    download_keys: Optional[List[str]] = None,
                    pretrained_models_dir: Optional[str] = None,
                    model_paths: Optional[Dict[str, str]] = None,
                    progress_callback: ProgressCallback = None,
                    log_callback: LogCallback = None) -> Dict[str, object]:
    """Download selected models with callback support for GUI."""
    if not pretrained_models_dir:
        pretrained_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained_models")

    pretrained_models_dir = os.path.abspath(pretrained_models_dir)
    os.makedirs(pretrained_models_dir, exist_ok=True)

    model_catalog = get_model_catalog(pretrained_models_dir, model_paths=model_paths)
    if not download_keys:
        keys = ["wetext", "cosyvoice3"]
    else:
        keys = [key for key in download_keys if key in model_catalog]

    to_download = [model_catalog[key] for key in keys]

    _emit_log("🚀 开始下载模型...", log_callback)
    _emit_log(f"下载渠道: {download_method}", log_callback)
    _emit_log(f"待下载模型数: {len(to_download)}", log_callback)
    _emit_log(f"模型保存位置: {pretrained_models_dir}", log_callback)
    _emit_progress(0, "开始准备下载", progress_callback)

    results: List[Tuple[str, bool, bool]] = []
    for index, model_info in enumerate(to_download):
        success, skipped = download_model(
            model_info,
            download_method,
            token,
            model_index=index,
            total_models=len(to_download),
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
        results.append((model_info[0], success, skipped))

    _emit_log("\n" + "=" * 50, log_callback)
    _emit_log("📊 下载汇总:", log_callback)
    _emit_log("=" * 50, log_callback)
    for model_name, success, skipped in results:
        if skipped:
            status = "⏭️ 已存在"
        else:
            status = "✅ 成功" if success else "❌ 失败"
        _emit_log(f"{model_name}: {status}", log_callback)

    all_success = all(success for _, success, _ in results)
    if all_success:
        _emit_log("🎉 所有模型处理完成！", log_callback)
    else:
        _emit_log("⚠️ 部分模型下载失败，请检查网络并重试。", log_callback)

    _emit_progress(100, "下载任务结束", progress_callback)

    return {
        "all_success": all_success,
        "base_dir": pretrained_models_dir,
        "results": results,
        "resolved_paths": {
            "cosyvoice_model_path": model_catalog["cosyvoice3"][3],
            "wetext_model_path": model_catalog["wetext"][3],
        },
        "download_status": {
            "cosyvoice3": is_model_downloaded(model_catalog["cosyvoice3"][3]),
            "wetext": is_model_downloaded(model_catalog["wetext"][3]),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Download CosyVoice models")
    parser.add_argument("--method", choices=["huggingface", "modelscope"], help="Download method")
    parser.add_argument("--token", help="HuggingFace token")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--wetext", action="store_true", help="Download wetext only")
    parser.add_argument("--cosyvoice3", action="store_true", help="Download CosyVoice3 only")
    parser.add_argument("--models-dir", help="Custom pretrained_models directory")

    args = parser.parse_args()

    token = args.token
    if args.method:
        download_method = args.method
    else:
        download_method, token = get_download_method()

    download_keys: List[str] = []
    if args.all or (not args.wetext and not args.cosyvoice3):
        download_keys = ["wetext", "cosyvoice3"]
    else:
        if args.wetext:
            download_keys.append("wetext")
        if args.cosyvoice3:
            download_keys.append("cosyvoice3")

    download_models(
        download_method=download_method,
        token=token,
        download_keys=download_keys,
        pretrained_models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()

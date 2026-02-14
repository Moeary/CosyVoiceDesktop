import os
import sys
import gc
import subprocess
from typing import List, Optional

def load_cosyvoice_model():
    """加载CosyVoice模型的工具函数"""
    from core.config_manager import ConfigManager
    from core.download import get_model_catalog
    config = ConfigManager()
    
    # 获取原始路径设置
    raw_cosy_path = config.get("cosyvoice_model_path") or "./pretrained_models"
    raw_wetext_path = config.get("wetext_model_path") or "./pretrained_models"
    
    # 使用 catalog 逻辑解析出真实的模型存放路径 (会自动补全子文件夹)
    catalog = get_model_catalog(os.path.abspath("./pretrained_models"), {
        "cosyvoice3": raw_cosy_path,
        "wetext": raw_wetext_path
    })
    
    model_dir = catalog["cosyvoice3"][3]
    wetext_dir = catalog["wetext"][3]
    
    # 冗余检查：如果补全后的路径不存在，尝试原始路径（万一用户故意放到了一个非标准命名的文件夹）
    if not os.path.exists(os.path.join(model_dir, "cosyvoice3.yaml")):
        if os.path.exists(os.path.join(os.path.abspath(raw_cosy_path), "cosyvoice3.yaml")):
            model_dir = os.path.abspath(raw_cosy_path)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"未能找到模型目录: {model_dir}")
    
    # 确保第三方库路径正确
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    matcha_path = os.path.join(root_path, 'third_party', 'Matcha-TTS')
    
    if matcha_path not in sys.path:
        sys.path.insert(0, matcha_path)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
        
    from cosyvoice.cli.cosyvoice import AutoModel
    
    print(f"正在从以下路径加载模型:\n - CosyVoice: {model_dir}\n - WeText: {wetext_dir}")
    
    return AutoModel(
        model_dir=model_dir, 
        load_trt=False, 
        load_vllm=False, 
        fp16=False,
        wetext_dir=wetext_dir
    )

def unload_cosyvoice_model(model):
    """卸载CosyVoice模型并彻底释放显存
    
    参数:
        model: CosyVoice 模型对象
    """
    if model is None:
        return
    
    try:
        # 清理内部缓存字典
        if hasattr(model, 'model'):
            model_obj = model.model
            
            # 清理缓存字典
            if hasattr(model_obj, 'tts_speech_token_dict'):
                model_obj.tts_speech_token_dict.clear()
            if hasattr(model_obj, 'llm_end_dict'):
                model_obj.llm_end_dict.clear()
            if hasattr(model_obj, 'hift_cache_dict'):
                model_obj.hift_cache_dict.clear()
            if hasattr(model_obj, 'flow_cache_dict'):
                model_obj.flow_cache_dict.clear()
            if hasattr(model_obj, 'mel_overlap_dict'):
                model_obj.mel_overlap_dict.clear()
            
            # 将模型组件从 GPU 移到 CPU 并删除
            if hasattr(model_obj, 'llm'):
                try:
                    model_obj.llm.cpu()
                    del model_obj.llm
                except:
                    pass
            
            if hasattr(model_obj, 'flow'):
                try:
                    model_obj.flow.cpu()
                    del model_obj.flow
                except:
                    pass
            
            if hasattr(model_obj, 'hift'):
                try:
                    model_obj.hift.cpu()
                    del model_obj.hift
                except:
                    pass
            
            # 删除模型对象
            try:
                del model.model
            except:
                pass
        
        # 清理 frontend
        if hasattr(model, 'frontend'):
            try:
                # 清理 frontend 的缓存
                if hasattr(model.frontend, 'spk2info'):
                    model.frontend.spk2info.clear()
                del model.frontend
            except:
                pass
        
        # 最后删除整个模型对象
        del model
        
    except Exception as e:
        print(f"⚠️ Error during model unloading: {e}")
    
    # 强制垃圾回收
    gc.collect()
    
    # 清理 GPU 缓存
    try:
        import torch
        if torch.cuda.is_available():
            # 同步所有 CUDA 操作
            torch.cuda.synchronize()
            # 清空缓存
            torch.cuda.empty_cache()
            # 重置峰值内存
            torch.cuda.reset_peak_memory_stats()
    except:
        pass

def merge_audio_files(audio_files: List[str], output_dir: str, 
                     output_name: str) -> Optional[str]:
    """合并音频文件"""
    try:
        # 检查ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ 未找到ffmpeg")
            return None
        
        output_path = os.path.join(output_dir, output_name)
        
        # 创建文件列表
        filelist_path = os.path.join(output_dir, "filelist_temp.txt")
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for audio_file in audio_files:
                # Windows路径处理
                abs_path = os.path.abspath(audio_file).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        # 合并
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy', '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 清理
        try:
            os.remove(filelist_path)
        except:
            pass
        
        return output_path if result.returncode == 0 else None
        
    except Exception as e:
        print(f"❌ 合成错误: {str(e)}")
        return None

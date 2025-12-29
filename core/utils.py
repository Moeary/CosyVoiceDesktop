import os
import sys
import gc
import subprocess
from typing import List, Optional

def load_cosyvoice_model():
    """加载CosyVoice模型的工具函数
    
    返回:
        CosyVoice 模型对象
        
    抛出异常:
        FileNotFoundError: 模型文件不存在
        Exception: 模型加载失败
    """
    # 优先尝试CosyVoice3
    model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'
    if not os.path.exists(model_dir):
        # 尝试另一个可能的目录名
        model_dir = 'pretrained_models/CosyVoice3-0.5B'
        
    if not os.path.exists(model_dir):
        # 回退到CosyVoice2
        model_dir = 'pretrained_models/CosyVoice2-0.5B'
        
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    sys.path.insert(0, 'third_party/Matcha-TTS')
    from cosyvoice.cli.cosyvoice import AutoModel
    
    # AutoModel会自动根据yaml选择正确的模型类
    # 注意：CosyVoice3不支持load_jit参数
    return AutoModel(
        model_dir=model_dir, 
        load_trt=False, 
        load_vllm=False, 
        fp16=False
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

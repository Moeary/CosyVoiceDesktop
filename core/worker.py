import sys
import os
import torch
import random
import torchaudio
from typing import List, Optional
from PyQt5.QtCore import QThread, pyqtSignal

from .models import TaskSegment

class AudioGenerationWorker(QThread):
    """音频生成工作线程"""
    progress = pyqtSignal(str)  # 日志消息
    finished = pyqtSignal(list)  # 生成的文件列表
    error = pyqtSignal(str)  # 错误消息
    segment_finished = pyqtSignal(int, list)  # 段落索引, 生成的文件列表
    
    def __init__(self, segments: List[TaskSegment], output_dir: str, 
                 project_name: str, cosyvoice_model=None):
        super().__init__()
        self.segments = segments
        self.output_dir = output_dir
        self.project_name = project_name
        self.cosyvoice = cosyvoice_model
        self.is_running = True
    
    def stop(self):
        self.is_running = False
    
    def run(self):
        try:
            # 如果没有模型，先加载
            if self.cosyvoice is None:
                self.progress.emit("📦 正在加载CosyVoice模型...")
                self.cosyvoice = self.load_model()
                self.progress.emit("✅ 模型加载成功")
            
            # 导入必要的模块
            from cosyvoice.utils.file_utils import load_wav
            
            # 创建输出目录
            # 修改：输出目录包含项目名
            project_output_dir = os.path.join(self.output_dir, self.project_name)
            os.makedirs(project_output_dir, exist_ok=True)
            
            all_generated_files = []
            
            # 按段落生成
            for segment in self.segments:
                if not self.is_running:
                    break
                
                # 设置随机种子
                torch.manual_seed(segment.seed)
                random.seed(segment.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(segment.seed)
                    torch.cuda.manual_seed_all(segment.seed)
                
                self.progress.emit(f"🎵 正在生成第 {segment.index} 段...")
                self.progress.emit(f"   文本: {segment.text}")
                self.progress.emit(f"   配置: {segment.voice_config.name} ({segment.mode})")
                self.progress.emit(f"   种子: {segment.seed}")
                
                # 加载参考音频
                if not segment.voice_config.prompt_audio or not os.path.exists(segment.voice_config.prompt_audio):
                    self.progress.emit(f"⚠️ 参考音频不存在，跳过")
                    continue
                
                # 修改：直接传递音频路径，而不是加载后的tensor
                # CosyVoice内部会处理音频加载
                prompt_audio_path = segment.voice_config.prompt_audio
                
                # 生成音频 - 同一次运行的所有片段作为一个版本
                segment_files = []
                
                inference_func = self.get_inference_function(segment)
                
                for sub_idx, result in enumerate(inference_func(segment, prompt_audio_path)):
                    if not self.is_running:
                        break
                    
                    # 生成文件名：使用run_count+1作为版本号
                    filename = self.generate_filename(segment, sub_idx, segment.run_count + 1)
                    filepath = os.path.join(project_output_dir, filename)
                    
                    # 保存音频
                    torchaudio.save(filepath, result['tts_speech'], self.cosyvoice.sample_rate)
                    segment_files.append(filepath)
                    all_generated_files.append(filepath)
                    
                    self.progress.emit(f"✅ 保存: {filename}")
                
                # 将这一批文件作为新版本添加
                if segment_files:
                    segment.add_version(segment_files)
                    self.progress.emit(f"📦 版本 v{segment.run_count} 包含 {len(segment_files)} 个片段")
                
                # 发送段落完成信号
                self.segment_finished.emit(segment.index, segment_files)
            
            if self.is_running:
                self.finished.emit(all_generated_files)
            
        except Exception as e:
            self.error.emit(f"生成失败: {str(e)}")
    
    def load_model(self):
        """加载CosyVoice模型"""
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
        
        sys.path.append('third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import AutoModel
        
        # AutoModel会自动根据yaml选择正确的模型类
        # 注意：CosyVoice3不支持load_jit参数
        return AutoModel(
            model_dir=model_dir, 
            load_trt=False, 
            load_vllm=False, 
            fp16=False
        )
    
    def get_inference_function(self, segment: TaskSegment):
        """获取推理函数"""
        # 检查是否为CosyVoice3模型
        is_v3 = 'CosyVoice3' in getattr(self.cosyvoice, 'model_dir', '')
        
        if segment.mode == '零样本复制':
            def inference(seg, prompt_audio):
                prompt_text = seg.voice_config.prompt_text
                # CosyVoice3需要特定的prompt格式
                if is_v3 and '<|endofprompt|>' not in prompt_text:
                    prompt_text = f'You are a helpful assistant.<|endofprompt|>{prompt_text}'
                
                return self.cosyvoice.inference_zero_shot(
                    seg.text, prompt_text, 
                    prompt_audio, stream=False
                )
            return inference
        
        elif segment.mode == '精细控制':
            def inference(seg, prompt_audio):
                text = seg.text
                # CosyVoice3精细控制需要在文本前加指令
                if is_v3 and '<|endofprompt|>' not in text:
                    text = f'You are a helpful assistant.<|endofprompt|>{text}'
                
                return self.cosyvoice.inference_cross_lingual(
                    text, prompt_audio, stream=False
                )
            return inference
        
        elif segment.mode == '指令控制':
            def inference(seg, prompt_audio):
                instruct_text = seg.instruct_text
                # CosyVoice3指令需要以<|endofprompt|>结尾，且通常需要"You are a helpful assistant."前缀
                if is_v3:
                    # 确保指令在中间：You are a helpful assistant. {instruct_text}<|endofprompt|>
                    if '<|endofprompt|>' not in instruct_text:
                        instruct_text = f'{instruct_text}<|endofprompt|>'
                    if 'You are a helpful assistant.' not in instruct_text:
                        instruct_text = f'You are a helpful assistant. {instruct_text}'
                
                # 使用 inference_instruct2
                return self.cosyvoice.inference_instruct2(
                    seg.text, instruct_text, 
                    prompt_audio, stream=False
                )
            return inference
        
        elif segment.mode == '语音修补':
            def inference(seg, prompt_audio):
                prompt_text = seg.voice_config.prompt_text
                # 语音修补本质上是Zero-Shot，但文本可能包含发音修正标记
                if is_v3 and '<|endofprompt|>' not in prompt_text:
                    prompt_text = f'You are a helpful assistant.<|endofprompt|>{prompt_text}'
                
                return self.cosyvoice.inference_zero_shot(
                    seg.text, prompt_text, 
                    prompt_audio, stream=False
                )
            return inference
        
        else:  # 默认回退到零样本
            def inference(seg, prompt_audio):
                prompt_text = seg.voice_config.prompt_text
                if is_v3 and '<|endofprompt|>' not in prompt_text:
                    prompt_text = f'You are a helpful assistant.<|endofprompt|>{prompt_text}'
                
                return self.cosyvoice.inference_zero_shot(
                    seg.text, prompt_text, 
                    prompt_audio, stream=False
                )
            return inference
    
    def generate_filename(self, segment: TaskSegment, sub_index: int, version: int) -> str:
        """生成文件名: 段落序号_版本号_文本预览_片段序号.wav"""
        # 文本预览（10个字符）
        text_preview = self.sanitize_filename(segment.text[:10])
        
        # 格式：段落_版本_文本_片段.wav
        # 只有一个片段时不显示片段号
        return f"{segment.index}_{version}_{text_preview}_{sub_index+1}.wav"
    
    def sanitize_filename(self, text: str) -> str:
        """处理文件名，符合Windows规则"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, '')
        text = ''.join(char for char in text if ord(char) >= 32)
        text = text.replace(' ', '_').replace('\n', '_').replace('\t', '_')
        while '__' in text:
            text = text.replace('__', '_')
        text = text.strip('_')
        return text or 'audio'

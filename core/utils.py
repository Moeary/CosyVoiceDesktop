import os
import subprocess
from typing import List, Optional

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

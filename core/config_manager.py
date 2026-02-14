import json
import os
from typing import Dict, Any

class ConfigManager:
    """配置管理器，用于持久化保存应用设置"""
    
    def __init__(self, config_path: str = "app_config.json"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {
            "theme": "Light",
            "voice_config_path": "",
            "project_name": "project",
            "output_dir": "./output",
            "cosyvoice_model_path": "./pretrained_models",
            "wetext_model_path": "./pretrained_models"
        }
        self.load_config()

    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except Exception as e:
                print(f"Error loading config: {e}")

    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self.save_config()

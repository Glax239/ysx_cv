# -*- coding: utf-8 -*-
"""
AI分析器配置文件
Configuration file for AI Analyzer
"""

import os
from typing import List, Optional

class AIConfig:
    """AI分析器配置类"""
    
    # =============================================================================
    # API密钥配置 - 请在这里设置您的API密钥
    # =============================================================================
    
    # 方式1：直接在这里设置API密钥（不推荐用于生产环境）
    GEMINI_API_KEY = "AIzaSyDKDDRG701TttUwCL7uvEkKJYMzo3lsCPo"
    
    # 方式2：从环境变量读取（推荐）
    # 在系统中设置环境变量 GOOGLE_AI_API_KEY
    # Windows: setx GOOGLE_AI_API_KEY "AIzaSyDKDDRG701TttUwCL7uvEkKJYMzo3lsCPo"
    # Linux/Mac: export GOOGLE_AI_API_KEY="您的API密钥"
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """
        获取API密钥，按优先级顺序：
        1. 环境变量 GOOGLE_AI_API_KEY
        2. 环境变量 GEMINI_API_KEY  
        3. 配置文件中的 GEMINI_API_KEY
        """
        # 优先从环境变量获取
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if api_key:
            return api_key
            
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            return api_key
            
        # 如果环境变量没有，使用配置文件中的密钥
        return cls.GEMINI_API_KEY if cls.GEMINI_API_KEY else None
    
    # =============================================================================
    # 模型配置 - 请在这里选择您要使用的模型
    # =============================================================================
    
    # 可用的Gemini模型列表（按优先级排序）
    AVAILABLE_MODELS = [
        "gemini-2.0-flash-lite", # 轻量版本，速度极快
        "gemini-1.5-flash",      # 最快，适合大多数任务
        "gemini-1.5-pro",        # 更强大，适合复杂任务
        "gemini-pro",            # 经典版本

    ]
    
    # 默认使用的模型（如果为None，将按优先级尝试）
    DEFAULT_MODEL = "gemini-2.0-flash-lite"
    
    # =============================================================================
    # API调用配置
    # =============================================================================
    
    # 超时设置（秒）
    API_TIMEOUT = 5
    
    # 重试配置
    MAX_RETRIES = 2
    RETRY_DELAY = 1
    
    # 生成配置
    GENERATION_CONFIG = {
        "max_output_tokens": 600,
        "temperature": 0.3,
        "top_p": 0.7,
        "top_k": 30
    }
    
    # =============================================================================
    # 模拟模式配置
    # =============================================================================
    
    # 是否启用模拟模式作为备选
    ENABLE_SIMULATION_FALLBACK = True
    
    # 模拟分析延迟范围（秒）
    SIMULATION_DELAY_RANGE = (0.5, 1.5)
    
    # =============================================================================
    # 日志配置
    # =============================================================================
    
    # 日志级别
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # 是否显示详细的API调用信息
    VERBOSE_API_CALLS = True
    
    # =============================================================================
    # 提示词配置
    # =============================================================================
    
    # 健康分析提示词模板
    HEALTH_ANALYSIS_PROMPT_TEMPLATE = """作为营养专家，简要分析以下产品的健康程度（中文回答，300字内）：

{text_content}

请提供：
1. 主要营养成分识别
2. 健康评级（优秀/良好/一般/较差）
3. 简要建议

要求：客观、科学、简洁。"""

    # 自定义提示词（如果不为空，将使用此提示词）
    CUSTOM_PROMPT = ""
    
    @classmethod
    def get_prompt_template(cls) -> str:
        """获取提示词模板"""
        return cls.CUSTOM_PROMPT if cls.CUSTOM_PROMPT else cls.HEALTH_ANALYSIS_PROMPT_TEMPLATE
    
    # =============================================================================
    # 验证方法
    # =============================================================================
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置是否有效"""
        api_key = cls.get_api_key()
        if not api_key:
            print("⚠️  警告：未配置API密钥，将使用模拟模式")
            return False
        
        if not cls.AVAILABLE_MODELS:
            print("❌ 错误：未配置可用模型")
            return False
            
        print("✅ AI配置验证通过")
        return True
    
    @classmethod
    def print_config_info(cls):
        """打印当前配置信息"""
        print("🔧 AI分析器配置信息:")
        print(f"   API密钥: {'已配置' if cls.get_api_key() else '未配置'}")
        print(f"   默认模型: {cls.DEFAULT_MODEL}")
        print(f"   可用模型: {', '.join(cls.AVAILABLE_MODELS)}")
        print(f"   超时时间: {cls.API_TIMEOUT}秒")
        print(f"   最大重试: {cls.MAX_RETRIES}次")
        print(f"   模拟模式: {'启用' if cls.ENABLE_SIMULATION_FALLBACK else '禁用'}")

# 创建全局配置实例
ai_config = AIConfig()

if __name__ == "__main__":
    # 测试配置
    ai_config.print_config_info()
    ai_config.validate_config()

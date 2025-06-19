# -*- coding: utf-8 -*-
"""
Gemini健康分析模块
Gemini Health Analysis Module
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime

class GeminiHealthAnalyzer:
    """使用Google Gemini API进行健康分析的类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Gemini健康分析器
        
        Args:
            api_key:'AIzaSyDKDDRG701TttUwCL7uvEkKJYMzo3lsCPo' Google Gemini API密钥，如果为None则从环境变量获取
        """
        # api_key='AIzaSyDKDDRG701TttUwCL7uvEkKJYMzo3lsCPo'
        self.logger = logging.getLogger(__name__)
        
        # 获取API密钥
        self.api_key ='AIzaSyDKDDRG701TttUwCL7uvEkKJYMzo3lsCPo'
        if not self.api_key:
            self.logger.warning("未找到Google API密钥，请设置GOOGLE_API_KEY环境变量")
            self.enabled = False
        else:
            self.enabled = True
            
        # Gemini API配置
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }
        
        self.logger.info(f"Gemini健康分析器初始化完成，状态: {'启用' if self.enabled else '禁用'}")
    
    def analyze_product_health(self, ocr_results: List[Dict], product_info: Dict = None) -> Dict[str, Any]:
        """
        分析产品健康程度
        
        Args:
            ocr_results: OCR识别结果列表，包含文本和坐标信息
            product_info: 产品基本信息（可选）
            
        Returns:
            健康分析结果字典
        """
        if not self.enabled:
            return {
                'status': 'disabled',
                'message': '未配置Google API密钥，无法进行健康分析',
                'overall_score': None,
                'analysis': None
            }
        
        try:
            # 构建分析提示
            prompt = self._build_health_analysis_prompt(ocr_results, product_info)
            
            # 调用Gemini API
            response = self._call_gemini_api(prompt)
            
            if response:
                # 解析响应
                analysis_result = self._parse_health_analysis(response)
                return {
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'overall_score': analysis_result.get('overall_score'),
                    'analysis': analysis_result.get('analysis'),
                    'recommendations': analysis_result.get('recommendations'),
                    'detected_ingredients': analysis_result.get('detected_ingredients', []),
                    'nutrition_facts': analysis_result.get('nutrition_facts', {}),
                    'ocr_corrections': analysis_result.get('ocr_corrections', []),
                    'health_warnings': analysis_result.get('health_warnings', [])
                }
            else:
                return {
                    'status': 'error',
                    'message': 'API调用失败',
                    'overall_score': None,
                    'analysis': None
                }
                
        except Exception as e:
            self.logger.error(f"健康分析失败: {e}")
            return {
                'status': 'error',
                'message': f'分析过程中出现错误: {str(e)}',
                'overall_score': None,
                'analysis': None
            }
    
    def _build_health_analysis_prompt(self, ocr_results: List[Dict], product_info: Dict = None) -> str:
        """
        构建健康分析提示
        
        Args:
            ocr_results: OCR识别结果
            product_info: 产品信息
            
        Returns:
            分析提示字符串
        """
        # 提取OCR文本和位置信息
        text_info = []
        all_texts = []
        
        for result in ocr_results:
            text = result.get('text', '').strip()
            bbox = result.get('bbox', [])
            confidence = result.get('confidence', 0)
            
            if text and confidence > 0.3:  # 降低置信度阈值以包含更多文本
                text_info.append({
                    'text': text,
                    'position': bbox,
                    'confidence': confidence
                })
                all_texts.append(text)
        
        # 合并所有文本用于分析
        combined_text = ' '.join(all_texts)
        
        # 构建提示
        prompt = f"""
你是一个专业的营养师和健康分析专家。请基于以下从产品包装上OCR识别的文本信息，用中文分析这个产品的健康程度。

识别到的文本信息：
{combined_text}

详细文本数据：
{json.dumps(text_info, ensure_ascii=False, indent=2)}

请执行以下任务：
1. 识别并纠正OCR错误（如拼写错误、字符识别错误）
2. 提取营养成分信息
3. 分析产品健康程度
4. 提供专业建议

请严格按照以下JSON格式返回结果：
{{
  "ocr_corrections": [
    {{"original": "错误文本", "corrected": "正确文本", "reason": "纠正原因"}}
  ],
  "detected_ingredients": ["成分1", "成分2"],
  "nutrition_facts": {{
    "calories": "热量值",
    "protein": "蛋白质含量",
    "fat": "脂肪含量",
    "carbohydrates": "碳水化合物",
    "sodium": "钠含量",
    "sugar": "糖分"
  }},
  "overall_score": 数字评分(1-10),
  "analysis": "详细的健康分析，包括营养价值评估、潜在健康风险等",
  "recommendations": ["具体的健康建议1", "具体的健康建议2"],
  "health_warnings": ["需要注意的健康警告"]
}}

分析要点：
- 仔细检查OCR文本中的拼写错误和字符识别错误
- 识别营养成分表中的关键数值
- 评估添加剂、防腐剂、人工色素等
- 考虑糖分、盐分、反式脂肪等健康风险因素
- 提供实用且具体的健康建议
- 如果信息不足，请明确说明
"""
        
        return prompt
    
    def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """
        调用Gemini API
        
        Args:
            prompt: 分析提示
            
        Returns:
            API响应文本
        """
        try:
            # 构建请求数据
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            # 发送请求
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    self.logger.info("Gemini API调用成功")
                    return content
                else:
                    self.logger.error("API响应格式异常")
                    return None
            else:
                self.logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error("API调用超时")
            return None
        except Exception as e:
            self.logger.error(f"API调用异常: {e}")
            return None
    
    def _parse_health_analysis(self, response_text: str) -> Dict[str, Any]:
        """
        解析健康分析响应
        
        Args:
            response_text: API响应文本
            
        Returns:
            解析后的分析结果
        """
        try:
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
            else:
                # 如果没有找到JSON，尝试解析文本
                return self._parse_text_response(response_text)
                
        except json.JSONDecodeError:
            # JSON解析失败，尝试文本解析
            return self._parse_text_response(response_text)
        except Exception as e:
            self.logger.error(f"响应解析失败: {e}")
            return {
                'overall_score': 5,
                'analysis': response_text,
                'detected_ingredients': [],
                'nutrition_facts': {},
                'ocr_corrections': [],
                'recommendations': [],
                'health_warnings': []
            }
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """
        解析文本响应（当JSON解析失败时的备用方案）
        
        Args:
            text: 响应文本
            
        Returns:
            解析后的结果
        """
        # 简单的文本解析逻辑
        import re
        
        # 尝试提取评分
        score_match = re.search(r'(\d+)\s*分', text)
        overall_score = int(score_match.group(1)) if score_match else 5
        if overall_score > 10:  # 如果是百分制，转换为10分制
            overall_score = overall_score // 10
        
        return {
            'overall_score': overall_score,
            'analysis': text,
            'detected_ingredients': [],
            'nutrition_facts': {},
            'ocr_corrections': [],
            'recommendations': [],
            'health_warnings': []
        }
    
    def is_available(self) -> bool:
        """
        检查Gemini分析器是否可用
        
        Returns:
            是否可用
        """
        return self.enabled
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取分析器状态
        
        Returns:
            状态信息
        """
        return {
            'enabled': self.enabled,
            'api_key_configured': bool(self.api_key),
            'base_url': self.base_url
        }
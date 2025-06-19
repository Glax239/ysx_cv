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
        # 使用与GUI相同的logger配置
        from utils.logger import setup_logger
        self.logger = setup_logger('core.gemini_health_analyzer')
        
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
    
    def analyze_product_health(self, ocr_results: List[Dict], product_info: Dict = None, yolo_text_data: List[Dict] = None) -> Dict[str, Any]:
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
            # 构建分析提示，包含OCR和YOLO数据
            prompt = self._build_health_analysis_prompt(ocr_results, product_info, yolo_text_data or [])
            
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
    
    def _build_health_analysis_prompt(self, ocr_results: List[Dict], product_info: Dict = None, yolo_text_data: List[Dict] = None) -> str:
        """
        构建健康分析提示，结合OCR和YOLO数据进行空间拼凑
        
        Args:
            ocr_results: OCR识别结果
            product_info: 产品信息
            yolo_text_data: YOLO检测的文本区域数据
            
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
                    'confidence': confidence,
                    'source': 'ocr'
                })
                all_texts.append(text)
        
        # 处理YOLO检测的文本区域信息
        yolo_regions = yolo_text_data or []
        
        # 执行空间拼凑：将OCR文本与YOLO区域进行匹配
        spatial_analysis = self._perform_spatial_matching(text_info, yolo_regions)
        
        # 按空间位置排序文本（从上到下，从左到右）
        sorted_text_info = self._sort_text_by_spatial_position(text_info)
        
        # 构建空间感知的文本序列
        spatial_text_sequence = self._build_spatial_text_sequence(sorted_text_info)
        
        # 合并所有文本用于分析
        combined_text = ' '.join(all_texts)
        
        # 构建提示
        prompt = f"""
你是一个专业的营养师和健康分析专家。请基于以下从产品包装上OCR识别的文本信息，结合空间位置分析，用中文分析这个产品的健康程度。

**重要提示：请特别关注相同y轴坐标（同一行）的文本内容，这些文本在物理上位于同一水平线上，通常具有关联性。**

识别到的文本信息：
{combined_text}

**按行排列的空间文本序列（相同y轴坐标的内容已分组）：**
{spatial_text_sequence}

详细文本数据（包含精确位置信息）：
{json.dumps(sorted_text_info, ensure_ascii=False, indent=2)}

YOLO检测的文本区域：
{json.dumps(yolo_regions, ensure_ascii=False, indent=2)}

空间匹配分析：
{spatial_analysis}

请执行以下任务：
1. **优先分析相同y轴坐标的文本内容**：
   - 同一行的文本通常在语义上相关（如"蛋白质 15g"、"脂肪 8g"等）
   - 注意数值与其对应标签的匹配关系
   - 识别营养成分表中的行对应关系
2. 识别并纠正OCR错误（如拼写错误、字符识别错误）
3. 提取营养成分信息（重点关注同行文本的关联性）
4. 分析产品健康程度
5. 提供专业建议

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
                self.logger.info(f"API响应结构: {result.keys() if result else 'Empty'}")
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    self.logger.info(f"Gemini API调用成功，响应长度: {len(content)}")
                    self.logger.info(f"响应内容前500字符: {content[:500]}")
                    return content
                else:
                    self.logger.error(f"API响应格式异常: {result}")
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
        self.logger.info(f"开始解析响应，文本长度: {len(response_text)}")
        self.logger.info(f"响应文本前200字符: {response_text[:200]}")
        
        try:
            # 尝试提取JSON部分 - 改进的正则表达式来匹配完整的JSON对象
            import re
            
            # 首先尝试找到```json代码块
            json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
                self.logger.info(f"从代码块中找到JSON字符串，长度: {len(json_str)}")
                self.logger.info(f"JSON字符串前300字符: {json_str[:300]}")
                result = json.loads(json_str)
                self.logger.info(f"JSON解析成功，包含字段: {list(result.keys())}")
                return result
            
            # 如果没有代码块，尝试找到完整的JSON对象
            # 使用更强大的正则表达式来匹配嵌套的JSON结构
            json_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            # 找到最长的JSON字符串（通常是完整的结果）
            if json_matches:
                json_str = max(json_matches, key=len)
                self.logger.info(f"找到JSON字符串，长度: {len(json_str)}")
                self.logger.info(f"JSON字符串前300字符: {json_str[:300]}")
                result = json.loads(json_str)
                self.logger.info(f"JSON解析成功，包含字段: {list(result.keys())}")
                return result
            else:
                self.logger.warning("未找到JSON格式，尝试文本解析")
                # 如果没有找到JSON，尝试解析文本
                return self._parse_text_response(response_text)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}，尝试文本解析")
            # JSON解析失败，尝试文本解析
            return self._parse_text_response(response_text)
        except Exception as e:
            self.logger.error(f"响应解析失败: {e}")
            return {
                'status': 'error',
                'message': f'解析失败: {str(e)}',
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
    
    def _perform_spatial_matching(self, text_info: List[Dict], yolo_regions: List[Dict]) -> str:
        """
        执行OCR文本与YOLO区域的空间匹配分析，着重分析相同y轴坐标的内容
        
        Args:
            text_info: OCR文本信息列表
            yolo_regions: YOLO检测的文本区域列表
            
        Returns:
            空间匹配分析结果字符串
        """
        if not yolo_regions:
            return "未提供YOLO检测区域数据"
        
        matches = []
        unmatched_ocr = []
        unmatched_yolo = list(yolo_regions)
        y_axis_groups = {}  # 按y轴坐标分组的匹配结果
        
        for ocr_item in text_info:
            ocr_bbox = ocr_item.get('position', [])
            if len(ocr_bbox) < 4:
                unmatched_ocr.append(ocr_item)
                continue
                
            ocr_center_y = (ocr_bbox[1] + ocr_bbox[3]) / 2
            best_match = None
            best_overlap = 0
            
            for i, yolo_item in enumerate(unmatched_yolo):
                yolo_bbox = yolo_item.get('bbox', [])
                if len(yolo_bbox) < 4:
                    continue
                    
                overlap = self._calculate_bbox_overlap(ocr_bbox, yolo_bbox)
                if overlap > best_overlap and overlap > 0.3:  # 30%重叠阈值
                    best_overlap = overlap
                    best_match = i
            
            if best_match is not None:
                yolo_match = unmatched_yolo.pop(best_match)
                yolo_center_y = (yolo_match['bbox'][1] + yolo_match['bbox'][3]) / 2
                
                match_info = {
                    'ocr_text': ocr_item['text'],
                    'ocr_bbox': ocr_bbox,
                    'ocr_center_y': ocr_center_y,
                    'yolo_bbox': yolo_match['bbox'],
                    'yolo_center_y': yolo_center_y,
                    'overlap': best_overlap,
                    'yolo_confidence': yolo_match.get('confidence', 0)
                }
                matches.append(match_info)
                
                # 按y轴坐标分组
                y_key = round(ocr_center_y / 10) * 10  # 以10像素为单位分组
                if y_key not in y_axis_groups:
                    y_axis_groups[y_key] = []
                y_axis_groups[y_key].append(match_info)
            else:
                unmatched_ocr.append(ocr_item)
        
        # 构建详细的分析报告
        analysis = f"**空间匹配结果（着重分析相同y轴坐标内容）：**\n"
        analysis += f"- 成功匹配: {len(matches)} 个文本区域\n"
        analysis += f"- 未匹配OCR文本: {len(unmatched_ocr)} 个\n"
        analysis += f"- 未匹配YOLO区域: {len(unmatched_yolo)} 个\n"
        analysis += f"- 识别到 {len(y_axis_groups)} 个不同的y轴坐标组\n\n"
        
        # 按y轴坐标分组显示匹配结果
        if y_axis_groups:
            analysis += "**按y轴坐标分组的匹配详情：**\n"
            for y_coord in sorted(y_axis_groups.keys()):
                group_matches = y_axis_groups[y_coord]
                analysis += f"\ny坐标组 {y_coord:.0f}px 附近 ({len(group_matches)} 个文本):\n"
                
                # 按x坐标排序同一行的文本
                group_matches.sort(key=lambda m: (m['ocr_bbox'][0] + m['ocr_bbox'][2]) / 2)
                
                row_texts = []
                for match in group_matches:
                    row_texts.append(f"'{match['ocr_text']}'")
                    analysis += f"  - 文本: '{match['ocr_text']}', 重叠度: {match['overlap']:.2f}, "
                    analysis += f"OCR_y: {match['ocr_center_y']:.0f}, YOLO_y: {match['yolo_center_y']:.0f}\n"
                
                analysis += f"  → 同行文本序列: {' | '.join(row_texts)}\n"
        
        # 显示未匹配的OCR文本（可能包含重要信息）
        if unmatched_ocr:
            analysis += "\n**未匹配的OCR文本（需要特别关注）：**\n"
            for item in unmatched_ocr[:5]:  # 只显示前5个
                bbox = item.get('position', [])
                if len(bbox) >= 4:
                    center_y = (bbox[1] + bbox[3]) / 2
                    analysis += f"  - '{item['text']}' (y: {center_y:.0f}px)\n"
        
        return analysis
    
    def _calculate_bbox_overlap(self, bbox1: List, bbox2: List) -> float:
        """
        计算两个边界框的重叠度
        
        Args:
            bbox1: 第一个边界框 [x1, y1, x2, y2]
            bbox2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            重叠度 (0-1)
        """
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        # 计算交集区域
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # 计算面积
        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        if bbox1_area == 0 or bbox2_area == 0:
            return 0.0
        
        # 返回相对于较小框的重叠度
        smaller_area = min(bbox1_area, bbox2_area)
        return intersection_area / smaller_area
    
    def _sort_text_by_spatial_position(self, text_info: List[Dict]) -> List[Dict]:
        """
        按空间位置排序文本（从上到下，从左到右），着重处理相同y轴坐标的内容
        
        Args:
            text_info: 文本信息列表
            
        Returns:
            排序后的文本信息列表
        """
        if not text_info:
            return []
            
        # 首先按y坐标分组，识别同一行的文本
        y_groups = {}
        for item in text_info:
            bbox = item.get('position', [])
            if len(bbox) < 4:
                continue
                
            center_y = (bbox[1] + bbox[3]) / 2
            row_height = bbox[3] - bbox[1]
            
            # 使用更精确的行分组策略
            # 寻找相同或相近y坐标的文本
            found_group = False
            tolerance = max(row_height * 0.3, 8)  # 行高30%或最小8像素的容差
            
            for group_y in y_groups.keys():
                if abs(center_y - group_y) <= tolerance:
                    y_groups[group_y].append(item)
                    found_group = True
                    break
                    
            if not found_group:
                y_groups[center_y] = [item]
        
        # 对每个y坐标组内的文本按x坐标排序
        sorted_groups = []
        for y_coord in sorted(y_groups.keys()):
            group_items = y_groups[y_coord]
            # 按x坐标（从左到右）排序同一行的文本
            group_items.sort(key=lambda item: (item.get('position', [0, 0, 0, 0])[0] + item.get('position', [0, 0, 0, 0])[2]) / 2)
            sorted_groups.extend(group_items)
            
        return sorted_groups
    
    def _build_spatial_text_sequence(self, sorted_text_info: List[Dict]) -> str:
        """
        构建空间感知的文本序列，着重记录相同y轴坐标的内容
        
        Args:
            sorted_text_info: 按空间位置排序的文本信息
            
        Returns:
            空间文本序列字符串，包含详细的行信息
        """
        if not sorted_text_info:
            return "无文本数据"
        
        # 按y坐标重新分组，确保相同行的文本被正确识别
        y_groups = {}
        for item in sorted_text_info:
            bbox = item.get('position', [])
            text = item.get('text', '').strip()
            
            if not text or len(bbox) < 4:
                continue
                
            center_y = (bbox[1] + bbox[3]) / 2
            row_height = bbox[3] - bbox[1]
            tolerance = max(row_height * 0.3, 8)
            
            # 寻找相同y坐标组
            found_group = False
            for group_y in y_groups.keys():
                if abs(center_y - group_y) <= tolerance:
                    y_groups[group_y].append({
                        'text': text,
                        'x_center': (bbox[0] + bbox[2]) / 2,
                        'y_center': center_y,
                        'bbox': bbox,
                        'confidence': item.get('confidence', 0)
                    })
                    found_group = True
                    break
                    
            if not found_group:
                y_groups[center_y] = [{
                    'text': text,
                    'x_center': (bbox[0] + bbox[2]) / 2,
                    'y_center': center_y,
                    'bbox': bbox,
                    'confidence': item.get('confidence', 0)
                }]
        
        # 构建详细的空间文本序列
        sequence_parts = []
        row_number = 1
        
        for y_coord in sorted(y_groups.keys()):
            row_items = y_groups[y_coord]
            # 按x坐标排序同一行的文本
            row_items.sort(key=lambda x: x['x_center'])
            
            # 构建行信息
            row_texts = [item['text'] for item in row_items]
            row_info = f"第{row_number}行(y≈{y_coord:.0f}): {' '.join(row_texts)}"
            
            # 添加详细的坐标信息用于Gemini分析
            coord_details = []
            for item in row_items:
                coord_details.append(f"'{item['text']}'(x:{item['x_center']:.0f},y:{item['y_center']:.0f})")
            
            detailed_info = f"  坐标详情: {' | '.join(coord_details)}"
            
            sequence_parts.append(row_info)
            sequence_parts.append(detailed_info)
            row_number += 1
        
        return '\n'.join(sequence_parts)
    
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
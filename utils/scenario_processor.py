# -*- coding: utf-8 -*-
"""
应用场景处理器
Scenario processor for different application scenarios
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

from config import SCENARIO_CONFIG
from core import SimpleDetectionEngine, ImageProcessor, SimpleInformationExtractor
from .file_utils import FileManager

class ScenarioProcessor:
    """应用场景处理器"""
    
    def __init__(self):
        """初始化场景处理器"""
        self.logger = logging.getLogger(__name__)
        self.detection_engine = SimpleDetectionEngine()
        self.image_processor = ImageProcessor()
        self.info_extractor = SimpleInformationExtractor()
        self.file_manager = FileManager()
    
    def process_personal_shopping(self, image: np.ndarray, image_path: str = None) -> Dict:
        """
        处理个人购物助手场景 - 专注于单个商品的详细分析

        Args:
            image: 输入图像
            image_path: 图像路径（可选）

        Returns:
            处理结果字典
        """
        try:
            self.logger.info("开始处理个人购物助手场景")

            # 执行综合检测
            detection_results = self.detection_engine.comprehensive_detection(image)

            # 个人购物场景特殊处理：专注于最大/最清晰的商品
            main_product = self._identify_main_product(detection_results)

            # 提取详细信息 - 重点关注营养和健康信息
            extraction_results = self.info_extractor.extract_comprehensive_info(
                image, detection_results
            )

            # 增强的营养信息分析
            enhanced_nutrition = self._enhanced_nutrition_analysis(extraction_results)

            # 生成个性化购物建议
            shopping_advice = self._generate_personalized_shopping_advice(
                detection_results, extraction_results, enhanced_nutrition
            )

            # 详细健康分析
            health_analysis = self._detailed_health_analysis(extraction_results, enhanced_nutrition)

            # 价格和性价比分析（基于条形码）
            price_analysis = self._analyze_price_value(extraction_results)

            # 构建结果
            result = {
                'scenario': 'personal_shopping',
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'main_product': main_product,
                'detection_results': detection_results,
                'extraction_results': extraction_results,
                'enhanced_nutrition': enhanced_nutrition,
                'shopping_advice': shopping_advice,
                'health_analysis': health_analysis,
                'price_analysis': price_analysis,
                'recommendations': self._generate_product_recommendations(extraction_results),
                'summary': self._create_personal_shopping_summary(
                    detection_results, extraction_results, shopping_advice, health_analysis
                )
            }

            self.logger.info("个人购物助手场景处理完成")
            return result

        except Exception as e:
            self.logger.error(f"个人购物助手场景处理失败: {e}")
            raise
    
    def process_shelf_audit(self, images: List[np.ndarray],
                          image_paths: List[str] = None) -> Dict:
        """
        处理货架审计场景 - 专注于批量分析和商业智能

        Args:
            images: 输入图像列表
            image_paths: 图像路径列表（可选）

        Returns:
            审计结果字典
        """
        try:
            self.logger.info(f"开始处理货架审计场景，共 {len(images)} 张图像")

            all_results = []
            product_locations = []
            shelf_analytics = {}

            # 处理每张图像
            for i, image in enumerate(images):
                image_path = image_paths[i] if image_paths else f"image_{i+1}"

                self.logger.info(f"处理第 {i+1}/{len(images)} 张图像")

                # 检测商品和品牌
                detection_results = self.detection_engine.comprehensive_detection(image)

                # 货架审计特殊处理：空间分析
                spatial_analysis = self._analyze_shelf_space(detection_results, image.shape)

                # 提取品牌信息（简化，专注于识别而非详细分析）
                extraction_results = self.info_extractor.extract_comprehensive_info(
                    image, detection_results
                )

                # 记录结果
                image_result = {
                    'image_index': i,
                    'image_path': image_path,
                    'detection_results': detection_results,
                    'extraction_results': extraction_results,
                    'spatial_analysis': spatial_analysis
                }
                all_results.append(image_result)



                # 记录商品位置
                self._record_product_locations(
                    product_locations, detection_results, i
                )

                # 货架分析
                self._update_shelf_analytics(shelf_analytics, detection_results, spatial_analysis, i)

            # 生成综合审计报告
            audit_report = self._generate_comprehensive_audit_report(
                all_results, product_locations, shelf_analytics
            )

            # 库存和陈列分析
            inventory_analysis = self._analyze_inventory_display(product_locations, shelf_analytics)

            # 构建最终结果
            result = {
                'scenario': 'shelf_audit',
                'timestamp': datetime.now().isoformat(),
                'total_images': len(images),
                'image_paths': image_paths,
                'individual_results': all_results,
                'product_locations': product_locations,
                'shelf_analytics': shelf_analytics,
                'audit_report': audit_report,
                'inventory_analysis': inventory_analysis,
                'business_insights': self._generate_business_insights(shelf_analytics),
                'summary': self._create_shelf_audit_summary(
                    all_results, audit_report
                )
            }

            self.logger.info("货架审计场景处理完成")
            return result

        except Exception as e:
            self.logger.error(f"货架审计场景处理失败: {e}")
            raise
    
    def _generate_shopping_advice(self, detection_results: Dict, 
                                extraction_results: Dict) -> Dict:
        """生成购物建议"""
        advice = {
            'product_identified': len(detection_results.get('products', [])) > 0,
            'nutrition_alerts': [],
            'barcode_info': [],
            'general_advice': []
        }
        
        # 营养建议
        nutrition = extraction_results.get('nutrition_info', {})
        if nutrition:
            if nutrition.get('sodium') and nutrition['sodium'] > 600:  # 高钠警告
                advice['nutrition_alerts'].append("注意：该产品钠含量较高")
            
            if nutrition.get('sugar') and nutrition['sugar'] > 15:  # 高糖警告
                advice['nutrition_alerts'].append("注意：该产品糖含量较高")
            
            if nutrition.get('energy'):
                advice['nutrition_alerts'].append(f"能量: {nutrition['energy']} kJ")
        
        # 条形码信息
        for barcode in extraction_results.get('barcodes', []):
            advice['barcode_info'].append({
                'type': barcode['type'],
                'data': barcode['data']
            })
        
        # 通用建议
        if not advice['product_identified']:
            advice['general_advice'].append("未能识别到明确的商品，建议重新拍摄")
        else:
            advice['general_advice'].append("商品识别成功，可查看详细信息")
        
        return advice
    
    def _analyze_health_info(self, extraction_results: Dict) -> Dict:
        """分析健康信息"""
        health_analysis = {
            'has_nutrition_info': bool(extraction_results.get('nutrition_info')),
            'health_score': 0,
            'health_warnings': [],
            'health_benefits': [],
            'recommendations': []
        }
        
        nutrition = extraction_results.get('nutrition_info', {})
        if nutrition:
            score = 100  # 基础分数
            
            # 根据营养成分调整健康分数
            if nutrition.get('sodium'):
                if nutrition['sodium'] > 800:
                    score -= 30
                    health_analysis['health_warnings'].append("钠含量过高")
                elif nutrition['sodium'] > 400:
                    score -= 15
                    health_analysis['health_warnings'].append("钠含量偏高")
            
            if nutrition.get('sugar'):
                if nutrition['sugar'] > 20:
                    score -= 25
                    health_analysis['health_warnings'].append("糖含量过高")
                elif nutrition['sugar'] > 10:
                    score -= 10
                    health_analysis['health_warnings'].append("糖含量偏高")
            
            if nutrition.get('protein') and nutrition['protein'] > 10:
                score += 10
                health_analysis['health_benefits'].append("蛋白质含量丰富")
            
            health_analysis['health_score'] = max(0, min(100, score))
            
            # 生成建议
            if health_analysis['health_score'] < 60:
                health_analysis['recommendations'].append("建议选择更健康的替代品")
            elif health_analysis['health_score'] > 80:
                health_analysis['recommendations'].append("这是一个相对健康的选择")
        
        return health_analysis
    

    
    def _record_product_locations(self, product_locations: List, 
                                detection_results: Dict, image_index: int):
        """记录商品位置信息"""
        for product in detection_results.get('products', []):
            location_info = {
                'image_index': image_index,
                'product_class': product['class_name'],
                'bbox': product['bbox'],
                'confidence': product['confidence'],
                'center_point': self._calculate_center_point(product['bbox'])
            }
            product_locations.append(location_info)
    
    def _calculate_center_point(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """计算边界框中心点"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)
    
    def _generate_audit_report(self, all_results: List, product_locations: List) -> Dict:
        """生成审计报告"""
        report = {
            'total_products_detected': len(product_locations),
            'coverage_analysis': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        # 覆盖率分析
        images_with_products = len(set(loc['image_index'] for loc in product_locations))
        total_images = len(all_results)
        coverage_rate = (images_with_products / total_images * 100) if total_images > 0 else 0
        
        report['coverage_analysis'] = {
            'images_with_products': images_with_products,
            'total_images': total_images,
            'coverage_rate': round(coverage_rate, 2)
        }
        
        # 质量指标
        all_confidences = [loc['confidence'] for loc in product_locations]
        if all_confidences:
            report['quality_metrics'] = {
                'average_detection_confidence': round(sum(all_confidences) / len(all_confidences), 3),
                'min_confidence': round(min(all_confidences), 3),
                'max_confidence': round(max(all_confidences), 3)
            }
        
        # 生成建议
        if coverage_rate < 80:
            report['recommendations'].append("建议增加货架覆盖率，确保所有区域都被检测")
        
        if report['quality_metrics'].get('average_detection_confidence', 0) < 0.7:
            report['recommendations'].append("检测置信度偏低，建议改善拍摄条件")
        
        return report

    def _identify_main_product(self, detection_results: Dict) -> Dict:
        """识别主要商品（个人购物场景）"""
        products = detection_results.get('products', [])
        if not products:
            return {}

        # 选择置信度最高且面积最大的商品
        main_product = max(products, key=lambda p: p['confidence'] * self._calculate_area(p['bbox']))

        return {
            'product': main_product,
            'area': self._calculate_area(main_product['bbox']),
            'is_main_focus': True
        }

    def _calculate_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """计算边界框面积"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _enhanced_nutrition_analysis(self, extraction_results: Dict) -> Dict:
        """增强的营养信息分析（个人购物场景）"""
        nutrition_info = extraction_results.get('nutrition_info', {})

        enhanced = {
            'basic_nutrition': nutrition_info,
            'health_scores': {},
            'dietary_flags': [],
            'recommendations': []
        }

        # 计算健康评分
        if nutrition_info:
            # 钠含量评分
            sodium = nutrition_info.get('sodium', 0)
            if sodium:
                if sodium < 140:
                    enhanced['health_scores']['sodium'] = {'score': 90, 'level': '低钠'}
                elif sodium < 400:
                    enhanced['health_scores']['sodium'] = {'score': 70, 'level': '中等'}
                else:
                    enhanced['health_scores']['sodium'] = {'score': 30, 'level': '高钠'}
                    enhanced['dietary_flags'].append('高钠食品')

            # 糖含量评分
            sugar = nutrition_info.get('sugar', 0)
            if sugar:
                if sugar < 5:
                    enhanced['health_scores']['sugar'] = {'score': 90, 'level': '低糖'}
                elif sugar < 15:
                    enhanced['health_scores']['sugar'] = {'score': 70, 'level': '中等'}
                else:
                    enhanced['health_scores']['sugar'] = {'score': 30, 'level': '高糖'}
                    enhanced['dietary_flags'].append('高糖食品')

            # 蛋白质评分
            protein = nutrition_info.get('protein', 0)
            if protein:
                if protein > 10:
                    enhanced['health_scores']['protein'] = {'score': 90, 'level': '高蛋白'}
                    enhanced['dietary_flags'].append('高蛋白食品')
                elif protein > 5:
                    enhanced['health_scores']['protein'] = {'score': 70, 'level': '中等蛋白'}
                else:
                    enhanced['health_scores']['protein'] = {'score': 50, 'level': '低蛋白'}

        return enhanced

    def _generate_personalized_shopping_advice(self, detection_results: Dict,
                                             extraction_results: Dict,
                                             enhanced_nutrition: Dict) -> Dict:
        """生成个性化购物建议"""
        advice = {
            'health_recommendations': [],
            'dietary_suggestions': [],
            'alternative_products': [],
            'usage_tips': []
        }

        # 基于营养分析的建议
        health_scores = enhanced_nutrition.get('health_scores', {})
        dietary_flags = enhanced_nutrition.get('dietary_flags', [])

        if '高钠食品' in dietary_flags:
            advice['health_recommendations'].append("此产品钠含量较高，建议适量食用")
            advice['dietary_suggestions'].append("可搭配新鲜蔬菜或水果平衡营养")

        if '高糖食品' in dietary_flags:
            advice['health_recommendations'].append("此产品糖含量较高，建议控制摄入量")
            advice['alternative_products'].append("考虑选择无糖或低糖替代品")

        if '高蛋白食品' in dietary_flags:
            advice['usage_tips'].append("适合运动后补充蛋白质")
            advice['dietary_suggestions'].append("可作为健身餐的一部分")



        return advice

    def _detailed_health_analysis(self, extraction_results: Dict, enhanced_nutrition: Dict) -> Dict:
        """详细健康分析"""
        analysis = {
            'overall_score': 0,
            'health_warnings': [],
            'health_benefits': [],
            'dietary_compatibility': {},
            'recommendations': []
        }

        # 计算总体健康评分
        scores = [score_info['score'] for score_info in enhanced_nutrition.get('health_scores', {}).values()]
        if scores:
            analysis['overall_score'] = sum(scores) / len(scores)

        # 饮食兼容性分析
        dietary_flags = enhanced_nutrition.get('dietary_flags', [])
        analysis['dietary_compatibility'] = {
            'low_sodium_diet': '高钠食品' not in dietary_flags,
            'low_sugar_diet': '高糖食品' not in dietary_flags,
            'high_protein_diet': '高蛋白食品' in dietary_flags,
            'general_healthy_diet': analysis['overall_score'] > 70
        }

        return analysis

    def _analyze_price_value(self, extraction_results: Dict) -> Dict:
        """价格和性价比分析"""
        analysis = {
            'barcode_found': False,
            'price_lookup_possible': False,
            'value_indicators': []
        }

        barcodes = extraction_results.get('barcodes', [])
        if barcodes:
            analysis['barcode_found'] = True
            analysis['price_lookup_possible'] = True
            analysis['value_indicators'].append("可通过条形码查询价格信息")

            for barcode in barcodes:
                analysis['value_indicators'].append(f"条形码: {barcode['data']}")

        return analysis

    def _generate_product_recommendations(self, extraction_results: Dict) -> List[str]:
        """生成产品推荐"""
        recommendations = []

        # 基于营养信息的推荐
        nutrition = extraction_results.get('nutrition_info', {})
        if nutrition.get('protein', 0) > 10:
            recommendations.append("适合健身人群")

        if nutrition.get('energy', 0) < 500:
            recommendations.append("低热量选择")



        return recommendations

    def _analyze_shelf_space(self, detection_results: Dict, image_shape: Tuple) -> Dict:
        """分析货架空间利用（货架审计场景）"""
        height, width = image_shape[:2]
        total_area = height * width

        analysis = {
            'total_shelf_area': total_area,
            'occupied_area': 0,
            'utilization_rate': 0,
            'product_density': 0,
            'empty_spaces': [],
            'hotspots': []
        }

        products = detection_results.get('products', [])
        if products:
            # 计算被占用的面积
            occupied_area = sum(self._calculate_area(p['bbox']) for p in products)
            analysis['occupied_area'] = occupied_area
            analysis['utilization_rate'] = (occupied_area / total_area) * 100
            analysis['product_density'] = len(products) / (total_area / 10000)  # 每万像素的产品数

            # 识别热点区域（产品密集区域）
            analysis['hotspots'] = self._identify_hotspots(products, width, height)

        return analysis

    def _identify_hotspots(self, products: List[Dict], width: int, height: int) -> List[Dict]:
        """识别产品密集的热点区域"""
        # 将图像分成网格
        grid_size = 4
        cell_width = width // grid_size
        cell_height = height // grid_size

        grid_counts = {}

        for product in products:
            bbox = product['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            grid_x = min(center_x // cell_width, grid_size - 1)
            grid_y = min(center_y // cell_height, grid_size - 1)

            grid_key = (grid_x, grid_y)
            grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1

        # 识别热点（产品数量超过平均值的网格）
        if grid_counts:
            avg_count = sum(grid_counts.values()) / len(grid_counts)
            hotspots = []

            for (grid_x, grid_y), count in grid_counts.items():
                if count > avg_count:
                    hotspots.append({
                        'grid_position': (grid_x, grid_y),
                        'product_count': count,
                        'area': {
                            'x1': grid_x * cell_width,
                            'y1': grid_y * cell_height,
                            'x2': (grid_x + 1) * cell_width,
                            'y2': (grid_y + 1) * cell_height
                        }
                    })

            return hotspots

        return []

    def _update_shelf_analytics(self, shelf_analytics: Dict, detection_results: Dict,
                               spatial_analysis: Dict, image_index: int):
        """更新货架分析数据"""
        if 'utilization_rates' not in shelf_analytics:
            shelf_analytics['utilization_rates'] = []
            shelf_analytics['product_densities'] = []
            shelf_analytics['total_products'] = 0
            shelf_analytics['hotspot_areas'] = []

        shelf_analytics['utilization_rates'].append(spatial_analysis['utilization_rate'])
        shelf_analytics['product_densities'].append(spatial_analysis['product_density'])
        shelf_analytics['total_products'] += len(detection_results.get('products', []))
        shelf_analytics['hotspot_areas'].extend(spatial_analysis.get('hotspots', []))

    def _generate_comprehensive_audit_report(self, all_results: List, brand_stats: Dict,
                                           product_locations: List, shelf_analytics: Dict) -> Dict:
        """生成综合审计报告"""
        report = self._generate_audit_report(all_results, brand_stats, product_locations)

        # 添加货架分析
        if shelf_analytics:
            utilization_rates = shelf_analytics.get('utilization_rates', [])
            if utilization_rates:
                report['shelf_utilization'] = {
                    'average_utilization': sum(utilization_rates) / len(utilization_rates),
                    'max_utilization': max(utilization_rates),
                    'min_utilization': min(utilization_rates)
                }

        return report

    def _analyze_brand_competition(self, brand_stats: Dict) -> Dict:
        """分析品牌竞争情况"""
        if not brand_stats:
            return {}

        total_instances = sum(stats['count'] for stats in brand_stats.values())

        competition_analysis = {
            'market_share': {},
            'dominant_brands': [],
            'competitive_landscape': 'balanced'
        }

        # 计算市场份额
        for brand, stats in brand_stats.items():
            share = (stats['count'] / total_instances) * 100
            competition_analysis['market_share'][brand] = round(share, 2)

            if share > 30:
                competition_analysis['dominant_brands'].append(brand)

        # 判断竞争格局
        max_share = max(competition_analysis['market_share'].values()) if competition_analysis['market_share'] else 0
        if max_share > 50:
            competition_analysis['competitive_landscape'] = 'monopolistic'
        elif max_share > 30:
            competition_analysis['competitive_landscape'] = 'dominant_player'

        return competition_analysis

    def _analyze_inventory_display(self, product_locations: List, shelf_analytics: Dict) -> Dict:
        """分析库存和陈列情况"""
        analysis = {
            'total_products_detected': len(product_locations),
            'display_efficiency': 'unknown',
            'recommendations': []
        }

        if shelf_analytics and shelf_analytics.get('utilization_rates'):
            avg_utilization = sum(shelf_analytics['utilization_rates']) / len(shelf_analytics['utilization_rates'])

            if avg_utilization > 80:
                analysis['display_efficiency'] = 'high'
                analysis['recommendations'].append("货架利用率高，陈列效果良好")
            elif avg_utilization > 60:
                analysis['display_efficiency'] = 'medium'
                analysis['recommendations'].append("货架利用率中等，可进一步优化陈列")
            else:
                analysis['display_efficiency'] = 'low'
                analysis['recommendations'].append("货架利用率较低，建议增加商品陈列")

        return analysis

    def _generate_business_insights(self, shelf_analytics: Dict) -> List[str]:
        """生成商业洞察"""
        insights = []

        if shelf_analytics and shelf_analytics.get('utilization_rates'):
            avg_utilization = sum(shelf_analytics['utilization_rates']) / len(shelf_analytics['utilization_rates'])
            if avg_utilization < 50:
                insights.append("货架空间利用不足，存在销售机会损失")
            elif avg_utilization > 90:
                insights.append("货架空间利用充分，但可能影响顾客选购体验")

        return insights

    def _analyze_shelf_space(self, detection_results: Dict, image_shape: Tuple) -> Dict:
        """分析货架空间利用（货架审计场景）"""
        height, width = image_shape[:2]
        total_area = height * width

        analysis = {
            'total_shelf_area': total_area,
            'occupied_area': 0,
            'utilization_rate': 0,
            'product_density': 0,
            'empty_spaces': [],
            'hotspots': []
        }

        products = detection_results.get('products', [])
        if products:
            # 计算被占用的面积
            occupied_area = sum(self._calculate_area(p['bbox']) for p in products)
            analysis['occupied_area'] = occupied_area
            analysis['utilization_rate'] = (occupied_area / total_area) * 100
            analysis['product_density'] = len(products) / (total_area / 10000)  # 每万像素的产品数

            # 识别热点区域（产品密集区域）
            analysis['hotspots'] = self._identify_hotspots(products, width, height)

        return analysis

    def _identify_hotspots(self, products: List[Dict], width: int, height: int) -> List[Dict]:
        """识别产品密集的热点区域"""
        # 将图像分成网格
        grid_size = 4
        cell_width = width // grid_size
        cell_height = height // grid_size

        grid_counts = {}

        for product in products:
            bbox = product['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            grid_x = min(center_x // cell_width, grid_size - 1)
            grid_y = min(center_y // cell_height, grid_size - 1)

            grid_key = (grid_x, grid_y)
            grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1

        # 识别热点（产品数量超过平均值的网格）
        if grid_counts:
            avg_count = sum(grid_counts.values()) / len(grid_counts)
            hotspots = []

            for (grid_x, grid_y), count in grid_counts.items():
                if count > avg_count:
                    hotspots.append({
                        'grid_position': (grid_x, grid_y),
                        'product_count': count,
                        'area': {
                            'x1': grid_x * cell_width,
                            'y1': grid_y * cell_height,
                            'x2': (grid_x + 1) * cell_width,
                            'y2': (grid_y + 1) * cell_height
                        }
                    })

            return hotspots

        return []

    def _update_shelf_analytics(self, shelf_analytics: Dict, detection_results: Dict,
                               spatial_analysis: Dict, image_index: int):
        """更新货架分析数据"""
        if 'utilization_rates' not in shelf_analytics:
            shelf_analytics['utilization_rates'] = []
            shelf_analytics['product_densities'] = []
            shelf_analytics['total_products'] = 0
            shelf_analytics['hotspot_areas'] = []

        shelf_analytics['utilization_rates'].append(spatial_analysis['utilization_rate'])
        shelf_analytics['product_densities'].append(spatial_analysis['product_density'])
        shelf_analytics['total_products'] += len(detection_results.get('products', []))
        shelf_analytics['hotspot_areas'].extend(spatial_analysis.get('hotspots', []))

    def _generate_comprehensive_audit_report(self, all_results: List,
                                           product_locations: List, shelf_analytics: Dict) -> Dict:
        """生成综合审计报告"""
        report = self._generate_audit_report(all_results, product_locations)

        # 添加货架分析
        if shelf_analytics:
            utilization_rates = shelf_analytics.get('utilization_rates', [])
            if utilization_rates:
                report['shelf_utilization'] = {
                    'average_utilization': sum(utilization_rates) / len(utilization_rates),
                    'max_utilization': max(utilization_rates),
                    'min_utilization': min(utilization_rates),
                    'utilization_variance': self._calculate_variance(utilization_rates)
                }

            product_densities = shelf_analytics.get('product_densities', [])
            if product_densities:
                report['product_density_analysis'] = {
                    'average_density': sum(product_densities) / len(product_densities),
                    'density_distribution': self._analyze_density_distribution(product_densities)
                }

        return report

    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _analyze_density_distribution(self, densities: List[float]) -> Dict:
        """分析密度分布"""
        if not densities:
            return {}

        sorted_densities = sorted(densities)
        n = len(sorted_densities)

        return {
            'low_density_areas': sum(1 for d in densities if d < sorted_densities[n//3]),
            'medium_density_areas': sum(1 for d in densities if sorted_densities[n//3] <= d < sorted_densities[2*n//3]),
            'high_density_areas': sum(1 for d in densities if d >= sorted_densities[2*n//3])
        }



    def _analyze_inventory_display(self, product_locations: List, shelf_analytics: Dict) -> Dict:
        """分析库存和陈列情况"""
        analysis = {
            'total_products_detected': len(product_locations),
            'average_products_per_shelf': 0,
            'display_efficiency': 'unknown',
            'recommendations': []
        }

        if shelf_analytics and shelf_analytics.get('utilization_rates'):
            avg_utilization = sum(shelf_analytics['utilization_rates']) / len(shelf_analytics['utilization_rates'])

            if avg_utilization > 80:
                analysis['display_efficiency'] = 'high'
                analysis['recommendations'].append("货架利用率高，陈列效果良好")
            elif avg_utilization > 60:
                analysis['display_efficiency'] = 'medium'
                analysis['recommendations'].append("货架利用率中等，可进一步优化陈列")
            else:
                analysis['display_efficiency'] = 'low'
                analysis['recommendations'].append("货架利用率较低，建议增加商品陈列或调整布局")

        return analysis

    def _generate_business_insights(self, brand_stats: Dict, shelf_analytics: Dict) -> List[str]:
        """生成商业洞察"""
        insights = []

        if brand_stats:
            brand_count = len(brand_stats)
            if brand_count > 10:
                insights.append("品牌种类丰富，消费者选择多样化")
            elif brand_count < 3:
                insights.append("品牌种类较少，可能存在供应链限制")

        if shelf_analytics and shelf_analytics.get('utilization_rates'):
            avg_utilization = sum(shelf_analytics['utilization_rates']) / len(shelf_analytics['utilization_rates'])
            if avg_utilization < 50:
                insights.append("货架空间利用不足，存在销售机会损失")
            elif avg_utilization > 90:
                insights.append("货架空间利用充分，但可能影响顾客选购体验")

        return insights
    
    def _create_personal_shopping_summary(self, detection_results: Dict,
                                        extraction_results: Dict,
                                        shopping_advice: Dict,
                                        health_analysis: Dict) -> Dict:
        """创建个人购物场景摘要"""
        return {
            'products_detected': len(detection_results.get('products', [])),
            'barcodes_found': len(extraction_results.get('barcodes', [])),
            'has_nutrition_info': bool(extraction_results.get('nutrition_info')),
            'health_score': health_analysis.get('health_score', 0),
            'advice_count': len(shopping_advice.get('general_advice', [])),
            'warnings_count': len(shopping_advice.get('nutrition_alerts', []))
        }
    
    def _create_shelf_audit_summary(self, all_results: List,
                                  audit_report: Dict) -> Dict:
        """创建货架审计场景摘要"""
        return {
            'total_images_processed': len(all_results),
            'total_products_detected': audit_report.get('total_products_detected', 0),
            'coverage_rate': audit_report.get('coverage_analysis', {}).get('coverage_rate', 0),
            'average_confidence': audit_report.get('quality_metrics', {}).get('average_detection_confidence', 0),
            'recommendations_count': len(audit_report.get('recommendations', []))
        }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品名称映射管理工具
Product Name Mapping Management Tool

用于管理商品名称映射配置的命令行工具
"""

import argparse
import json
import sys
from pathlib import Path
from utils.name_mapper import ProductNameMapper

def list_mappings(mapper: ProductNameMapper, model_name: str = None):
    """列出所有映射或指定模型的映射"""
    if model_name:
        mappings = mapper.get_model_mappings(model_name)
        if mappings:
            print(f"\n{model_name} 的名称映射:")
            print("-" * 50)
            for class_id, custom_name in mappings.items():
                print(f"  类别ID {class_id}: {custom_name}")
        else:
            print(f"\n{model_name} 没有配置名称映射")
    else:
        all_mappings = mapper.get_all_mappings()
        print("\n所有名称映射:")
        print("=" * 60)
        for model, mappings in all_mappings.items():
            print(f"\n{model}:")
            print("-" * 30)
            if mappings:
                for class_id, custom_name in mappings.items():
                    print(f"  类别ID {class_id}: {custom_name}")
            else:
                print("  (无映射)")

def add_mapping(mapper: ProductNameMapper, model_name: str, class_id: int, custom_name: str):
    """添加名称映射"""
    if mapper.add_mapping(model_name, class_id, custom_name):
        if mapper.save_mappings():
            print(f"✓ 成功添加映射: {model_name}[{class_id}] -> '{custom_name}'")
        else:
            print("✗ 添加映射成功但保存失败")
    else:
        print("✗ 添加映射失败")

def remove_mapping(mapper: ProductNameMapper, model_name: str, class_id: int):
    """删除名称映射"""
    if mapper.remove_mapping(model_name, class_id):
        if mapper.save_mappings():
            print(f"✓ 成功删除映射: {model_name}[{class_id}]")
        else:
            print("✗ 删除映射成功但保存失败")
    else:
        print(f"✗ 未找到映射: {model_name}[{class_id}]")

def import_mappings(mapper: ProductNameMapper, import_file: str):
    """从文件导入映射配置"""
    try:
        with open(import_file, 'r', encoding='utf-8') as f:
            new_mappings = json.load(f)
        
        # 合并映射配置
        current_mappings = mapper.get_all_mappings()
        for model_name, mappings in new_mappings.items():
            if model_name not in current_mappings:
                current_mappings[model_name] = {}
            current_mappings[model_name].update(mappings)
        
        mapper.mappings = current_mappings
        
        if mapper.save_mappings():
            print(f"✓ 成功从 {import_file} 导入映射配置")
        else:
            print("✗ 导入成功但保存失败")
            
    except Exception as e:
        print(f"✗ 导入映射配置失败: {e}")

def export_mappings(mapper: ProductNameMapper, export_file: str, model_name: str = None):
    """导出映射配置到文件"""
    try:
        if model_name:
            mappings = {model_name: mapper.get_model_mappings(model_name)}
        else:
            mappings = mapper.get_all_mappings()
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 成功导出映射配置到 {export_file}")
        
    except Exception as e:
        print(f"✗ 导出映射配置失败: {e}")

def batch_add_mappings(mapper: ProductNameMapper, model_name: str):
    """批量添加映射"""
    print(f"\n批量添加 {model_name} 的名称映射")
    print("输入格式: 类别ID,自定义名称")
    print("输入 'done' 完成，输入 'quit' 退出")
    print("-" * 40)
    
    added_count = 0
    while True:
        try:
            user_input = input("请输入映射 (类别ID,自定义名称): ").strip()
            
            if user_input.lower() == 'done':
                break
            elif user_input.lower() == 'quit':
                return
            
            if ',' not in user_input:
                print("✗ 格式错误，请使用: 类别ID,自定义名称")
                continue
            
            class_id_str, custom_name = user_input.split(',', 1)
            class_id = int(class_id_str.strip())
            custom_name = custom_name.strip()
            
            if mapper.add_mapping(model_name, class_id, custom_name):
                print(f"✓ 添加: {model_name}[{class_id}] -> '{custom_name}'")
                added_count += 1
            else:
                print(f"✗ 添加失败: {class_id}")
                
        except ValueError:
            print("✗ 类别ID必须是数字")
        except KeyboardInterrupt:
            print("\n操作被取消")
            return
        except Exception as e:
            print(f"✗ 错误: {e}")
    
    if added_count > 0:
        if mapper.save_mappings():
            print(f"\n✓ 成功添加 {added_count} 个映射并保存")
        else:
            print(f"\n✗ 添加了 {added_count} 个映射但保存失败")
    else:
        print("\n未添加任何映射")

def main():
    parser = argparse.ArgumentParser(
        description="商品名称映射管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python manage_product_names.py list                           # 列出所有映射
  python manage_product_names.py list -m product_detector       # 列出指定模型的映射
  python manage_product_names.py add product_detector 0 "可口可乐"  # 添加映射
  python manage_product_names.py remove product_detector 0      # 删除映射
  python manage_product_names.py batch product_detector         # 批量添加映射
  python manage_product_names.py import mappings.json           # 导入映射
  python manage_product_names.py export mappings.json           # 导出映射
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出名称映射')
    list_parser.add_argument('-m', '--model', help='指定模型名称')
    
    # add 命令
    add_parser = subparsers.add_parser('add', help='添加名称映射')
    add_parser.add_argument('model', help='模型名称')
    add_parser.add_argument('class_id', type=int, help='类别ID')
    add_parser.add_argument('custom_name', help='自定义名称')
    
    # remove 命令
    remove_parser = subparsers.add_parser('remove', help='删除名称映射')
    remove_parser.add_argument('model', help='模型名称')
    remove_parser.add_argument('class_id', type=int, help='类别ID')
    
    # batch 命令
    batch_parser = subparsers.add_parser('batch', help='批量添加名称映射')
    batch_parser.add_argument('model', help='模型名称')
    
    # import 命令
    import_parser = subparsers.add_parser('import', help='导入映射配置')
    import_parser.add_argument('file', help='导入文件路径')
    
    # export 命令
    export_parser = subparsers.add_parser('export', help='导出映射配置')
    export_parser.add_argument('file', help='导出文件路径')
    export_parser.add_argument('-m', '--model', help='指定模型名称')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 初始化名称映射器
    try:
        mapper = ProductNameMapper()
    except Exception as e:
        print(f"✗ 初始化名称映射器失败: {e}")
        return
    
    # 执行命令
    try:
        if args.command == 'list':
            list_mappings(mapper, args.model)
        
        elif args.command == 'add':
            add_mapping(mapper, args.model, args.class_id, args.custom_name)
        
        elif args.command == 'remove':
            remove_mapping(mapper, args.model, args.class_id)
        
        elif args.command == 'batch':
            batch_add_mappings(mapper, args.model)
        
        elif args.command == 'import':
            import_mappings(mapper, args.file)
        
        elif args.command == 'export':
            export_mappings(mapper, args.file, args.model)
        
    except KeyboardInterrupt:
        print("\n操作被取消")
    except Exception as e:
        print(f"✗ 执行命令时出错: {e}")

if __name__ == '__main__':
    main()
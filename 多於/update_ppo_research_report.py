#!/usr/bin/env python3
"""
Workflow script: Edit in Markdown, auto-convert to Word
Usage: python update_ppo_research_report.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def update_report():
    """Update Word document from Markdown source"""
    
    md_file = Path('flappy_bird_ppo_report.md')
    docx_file = Path('flappy_bird_ppo_report.docx')
    
    if not md_file.exists():
        print(f"❌ {md_file} not found")
        return False
    
    print(f"📝 更新報告...")
    print(f"   源文件：{md_file.name} ({md_file.stat().st_size / 1024:.1f} KB)")
    
    # Run conversion
    try:
        result = subprocess.run([sys.executable, 'convert_ppo_report_to_docx.py'],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(result.stdout)
            
            if docx_file.exists():
                size = docx_file.stat().st_size / 1024
                print(f"\n✅ 成功更新 Word 檔案")
                print(f"   時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   檔案：{docx_file.name} ({size:.1f} KB)")
                return True
        else:
            print(f"❌ 轉換失敗：{result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 錯誤：{e}")
        return False

def show_workflow():
    """Show recommended workflow"""
    print("\n" + "="*60)
    print("📋 推薦工作流程")
    print("="*60)
    print("""
1️⃣  在 VS Code 中編輯 Markdown：
   flappy_bird_ppo_report.md

2️⃣  保存後，在終端執行：
   python update_ppo_research_report.py

3️⃣  自動更新 Word 檔案：
   flappy_bird_ppo_report.docx

4️⃣  在 Word 中打開最新版本編輯或印刷

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ 優點：
• VS Code 原生支持 Markdown 編輯和預覽
• 版本控制友好（Markdown 是純文本）
• 快速更新 Word 檔案
• 保持最新的研究內容

""")

if __name__ == '__main__':
    show_workflow()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        update_report()
    else:
        print("\n💡 使用方法：")
        print("   python update_ppo_research_report.py update")
        print("\n   或直接執行：python convert_ppo_report_to_docx.py")

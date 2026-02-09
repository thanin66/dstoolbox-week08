from mcp.server.fastmcp import FastMCP
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model
import os

# สร้าง MCP Server ชื่อ "PyCaretFlow"
mcp = FastMCP("PyCaretFlow")

@mcp.tool()
def get_dataset_info(file_path: str) -> str:
    """
    อ่านไฟล์ CSV และสรุปข้อมูลเบื้องต้น (Columns, Missing values, Shape)
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        df = pd.read_csv(file_path)
        info = []
        info.append(f"Shape: {df.shape}")
        info.append(f"Columns: {', '.join(df.columns)}")
        info.append("Missing Values:\n" + df.isnull().sum().to_string())
        return "\n\n".join(info)
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def run_automl_classification(file_path: str, target_column: str, train_size: float = 0.7) -> str:
    """
    รัน PyCaret AutoML (Classification) 
    - file_path: ที่อยู่ไฟล์ csv
    - target_column: ชื่อคอลัมน์ที่ต้องการทำนาย
    Returns: ตารางเปรียบเทียบโมเดลในรูปแบบ Markdown
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    try:
        # 1. Load Data
        data = pd.read_csv(file_path)
        
        if target_column not in data.columns:
            return f"Error: Column '{target_column}' not found in dataset."

        # 2. Setup PyCaret (ปิด html เพื่อไม่ให้รก output)
        s = setup(data, target=target_column, train_size=train_size, session_id=123, verbose=False, html=False)
        
        # 3. Compare Models
        best_model = compare_models()
        
        # 4. Pull results (ดึงตารางคะแนน)
        results_df = pull()
        
        # 5. Save best model (บันทึกไว้เผื่อใช้)
        save_model(best_model, 'best_model_mcp')
        
        return f"AutoML Complete! Best Model Saved.\n\nLeaderboard:\n{results_df.to_markdown()}"
        
    except Exception as e:
        return f"PyCaret Error: {str(e)}"

if __name__ == "__main__":
    # สั่งรัน Server
    mcp.run()
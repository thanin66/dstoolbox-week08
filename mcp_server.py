# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp",
#     "pandas",
#     "pycaret[full]",
#     "matplotlib"
# ]
# ///

from mcp.server.fastmcp import FastMCP
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model, load_model, plot_model
import os
import warnings

# ปิด Warning รกๆ (เช่น cuml) เพื่อให้ Output สะอาด
warnings.filterwarnings('ignore')

# ตั้งค่า MCP Server
mcp = FastMCP("PyCaretFlow", dependencies=["pandas", "pycaret"])

# กำหนดโฟลเดอร์สำหรับเก็บกราฟและโมเดล
ARTIFACTS_DIR = os.path.abspath("artifacts")
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model_mcp")

@mcp.tool()
def get_dataset_info(file_path: str) -> str:
    """
    Step 1: อ่านไฟล์และสรุปโครงสร้างข้อมูล (Metadata Analysis)
    - แสดงจำนวนแถว/คอลัมน์
    - แสดงประเภทข้อมูล (Type) และค่าที่หายไป (Missing) ของแต่ละคอลัมน์
    """
    if not os.path.exists(file_path):
        return f"Error: หาไฟล์ไม่เจอที่ {file_path}"
    
    try:
        df = pd.read_csv(file_path)
        
        info = [f"### 📄 Dataset Analysis: {os.path.basename(file_path)}"]
        info.append(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # สร้างตารางรายละเอียดคอลัมน์
        info.append("\n**Column Details:**")
        info.append("| Column | Type | Missing | Unique | Sample Values |")
        info.append("|---|---|---|---|---|")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            n_unique = df[col].nunique()
            samples = str(df[col].dropna().head(3).tolist())
            info.append(f"| {col} | {dtype} | {missing} | {n_unique} | {samples} |")
            
        return "\n".join(info)
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def inspect_column(file_path: str, column_name: str) -> str:
    """
    Step 2: เจาะลึกข้อมูลในคอลัมน์ที่สนใจ (เช่น Target หรือ Categorical)
    - แสดงค่า Unique และจำนวนที่พบ (Frequency)
    """
    if not os.path.exists(file_path):
        return f"Error: File not found."
    
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Error: ไม่พบคอลัมน์ '{column_name}'"
        
        val_counts = df[column_name].value_counts()
        
        result = [f"### 🔍 Column Inspection: {column_name}"]
        result.append(f"**Type:** {df[column_name].dtype}")
        result.append("\n**Value Distribution:**")
        result.append(val_counts.to_markdown())
        
        return "\n".join(result)
    except Exception as e:
        return f"Error analyzing column: {str(e)}"

@mcp.tool()
def run_automl(file_path: str, target_column: str, train_size: float = 0.7, sort_metric: str = 'Accuracy') -> str:
    """
    Step 3: รัน PyCaret AutoML เพื่อหาโมเดลที่ดีที่สุด
    - file_path: ที่อยู่ไฟล์ csv
    - target_column: ชื่อคอลัมน์เป้าหมาย
    - sort_metric: เกณฑ์ตัดสิน (Accuracy, AUC, F1, Recall, Precision)
    """
    if not os.path.exists(file_path):
        return f"Error: File not found."

    try:
        # 1. Load Data
        data = pd.read_csv(file_path)
        if target_column not in data.columns:
            return f"Error: Target '{target_column}' not found."

        # 2. Setup (Silent Mode)
        s = setup(data, target=target_column, train_size=train_size, session_id=123, verbose=False, html=False)
        
        # 3. Compare Models
        best_model = compare_models(sort=sort_metric)
        
        # 4. Save Model & Results
        save_model(best_model, MODEL_PATH)
        results_df = pull()
        
        return (f"### 🚀 AutoML Complete\n"
                f"**Best Model:** {best_model}\n"
                f"**Saved to:** {MODEL_PATH}.pkl\n\n"
                f"### Leaderboard (Sorted by {sort_metric})\n"
                f"{results_df.to_markdown()}")
        
    except Exception as e:
        return f"PyCaret AutoML Failed: {str(e)}"

@mcp.tool()
def generate_plot(plot_type: str = 'confusion_matrix') -> str:
    """
    Step 4: สร้างกราฟจากโมเดลที่เทรนเสร็จแล้ว
    - plot_type options: 'auc', 'confusion_matrix', 'feature' (importance), 'class_report'
    Returns: ที่อยู่ไฟล์รูปภาพที่สร้างเสร็จ
    """
    try:
        # ตรวจสอบว่ามีโมเดลอยู่ใน Memory ไหม (ต้องรัน run_automl ก่อนใน session นี้)
        # หมายเหตุ: PyCaret Functional API เก็บ state ไว้ใน Global variable
        # หาก Server restart state อาจหายได้ แต่ถ้าคุยต่อเนื่องจะยังอยู่
        
        plot_filename = f"{plot_type}_{pd.Timestamp.now().strftime('%H%M%S')}"
        
        # สร้างกราฟและเซฟ (save=True จะคืนค่าเป็นชื่อไฟล์)
        # PyCaret จะพยายามใช้โมเดลล่าสุดใน Memory
        saved_file = plot_model(plot=plot_type, save=True, scale=1.0)
        
        if not saved_file:
            # กรณีไม่มีโมเดลใน memory ลองโหลด (แต่อาจขาด Test set context)
            return "Error: ไม่พบโมเดลใน Memory กรุณารัน 'run_automl' ก่อนสั่งพลอตกราฟ"

        # ย้ายไฟล์ไปที่โฟลเดอร์ artifacts เพื่อความเป็นระเบียบ
        import shutil
        original_path = f"{saved_file}" # ปกติเป็น .png
        new_path = os.path.join(ARTIFACTS_DIR, f"{plot_filename}.png")
        shutil.move(original_path, new_path)
        
        return f"📊 Plot Generated Successfully!\nPath: `{new_path}`"

    except Exception as e:
        return f"Error generating plot: {str(e)} \n(คำแนะนำ: ต้องรัน AutoML ก่อนถึงจะพลอตกราฟได้)"

if __name__ == "__main__":
    mcp.run()
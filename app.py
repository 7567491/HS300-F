import streamlit as st
import json
from func.data_fetcher import DataFetcher
from func.data_validator import DataValidator
from func.database import DatabaseManager
from func.ui_components import UIComponents

def load_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    st.set_page_config(page_title="沪深300指数预测数据获取工具", layout="wide")
    config = load_config()
    
    st.title("沪深300指数预测数据获取工具")
    
    # 初始化session_state
    if 'current_data_type' not in st.session_state:
        st.session_state.current_data_type = "沪深300指数"
    
    # 全局数据类型选择
    st.session_state.current_data_type = st.selectbox(
        "选择数据类型",
        [
            "沪深300指数",
            "市场总貌",
            "融资融券",
            "北向资金",
            "龙虎榜",
            "市场情绪",
            "市场结构",
            "宏观经济"
        ],
        key="global_data_type"
    )
    
    st.divider()  # 添加分隔线
    
    ui = UIComponents()
    tabs = st.tabs(["数据源配置", "数据获取", "数据校验", "数据概览", "指数预测"])
    
    with tabs[0]:
        ui.render_data_source_config()
        
    with tabs[1]:
        ui.render_data_fetching()
        
    with tabs[2]:
        ui.render_data_validation()
        
    with tabs[3]:
        ui.render_data_overview()
        
    with tabs[4]:
        ui.render_prediction()

if __name__ == "__main__":
    main() 
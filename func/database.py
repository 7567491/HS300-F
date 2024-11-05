import sqlite3
import pandas as pd
import logging
from pathlib import Path

class DatabaseManager:
    def __init__(self):
        self.db_path = Path('db/market_data.db')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = Path('data/cache')
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        self.init_database()
    
    def setup_logging(self):
        log_file = self.log_path / 'database.log'
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            # 沪深300日线数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS hs300_daily (
                    date TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    CHECK (date LIKE '____-__-__')
                )
            ''')
            
            # 宏观经济数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS macro_gdp (
                    date TEXT PRIMARY KEY,
                    gdp REAL,
                    CHECK (date LIKE '____-__-__')
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS macro_cpi (
                    date TEXT PRIMARY KEY,
                    cpi REAL,
                    CHECK (date LIKE '____-__-__')
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS macro_money (
                    date TEXT PRIMARY KEY,
                    m0 REAL,
                    m1 REAL,
                    m2 REAL,
                    CHECK (date LIKE '____-__-__')
                )
            ''')
            
            # 市场总貌数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_overview (
                    date TEXT PRIMARY KEY,
                    total_market_value REAL,
                    trading_volume REAL,
                    trading_amount REAL,
                    up_count INTEGER,
                    down_count INTEGER,
                    CHECK (date LIKE '____-__-__')
                )
            ''')
            
            # 融资融券数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS margin_trading (
                    date TEXT PRIMARY KEY,
                    margin_balance REAL,
                    margin_buy REAL,
                    margin_sell REAL,
                    margin_repay REAL,
                    CHECK (date LIKE '____-__-__')
                )
            ''')
            
            # 北向资金数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS north_money (
                    date TEXT PRIMARY KEY,
                    sh_net_flow REAL,
                    sz_net_flow REAL,
                    total_net_flow REAL,
                    CHECK (date LIKE '____-__-__')
                )
            ''')
            
            # 龙虎榜数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS top_list (
                    date TEXT,
                    stock_code TEXT,
                    stock_name TEXT,
                    reason TEXT,
                    buy_amount REAL,
                    sell_amount REAL,
                    net_amount REAL,
                    PRIMARY KEY (date, stock_code),
                    CHECK (date LIKE '____-__-__')
                )
            ''')
    
    def save_data(self, table_name, df):
        """保存数据到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 确保日期格式正确
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                # 保存数据
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logging.info(f"Successfully saved {len(df)} rows to {table_name}")
        except Exception as e:
            logging.error(f"Error saving data to {table_name}: {str(e)}")
            raise
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

class DataValidator:
    def __init__(self):
        self.log_path = Path('data/cache')
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # 定义不同数据类型的字段映射
        self.field_mappings = {
            'hs300_daily': {
                'date_field': 'date',
                'numeric_fields': ['open', 'high', 'low', 'close', 'volume']
            },
            'market_overview': {
                'date_field': 'date',
                'numeric_fields': ['total_market_value', 'trading_volume', 'trading_amount', 'up_count', 'down_count']
            },
            'margin_trading': {
                'date_field': 'date',
                'numeric_fields': ['margin_balance', 'margin_buy', 'margin_sell', 'margin_repay']
            },
            'north_money': {
                'date_field': 'date',
                'numeric_fields': ['sh_net_flow', 'sz_net_flow', 'total_net_flow']
            },
            'top_list': {
                'date_field': 'date',
                'numeric_fields': ['buy_amount', 'sell_amount', 'net_amount']
            },
            'market_sentiment': {
                'date_field': 'date',
                'numeric_fields': ['margin_balance', 'short_balance', 'north_money_net', 'turnover_rate']
            },
            'market_structure': {
                'date_field': 'date',
                'numeric_fields': ['total_mv', 'pe_ratio', 'new_accounts', 'total_accounts']
            },
            'macro_indicators': {
                'date_field': 'date',
                'numeric_fields': ['cpi_yoy', 'ppi_yoy', 'm2_amount', 'm2_yoy']
            }
        }
    
    def setup_logging(self):
        log_file = self.log_path / 'validation.log'
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def validate_market_data(self, df, data_type):
        """验证市场数据的完整性和质量"""
        # 获取当前数据类型的字段映射
        field_map = self.field_mappings.get(data_type, {
            'date_field': 'date',
            'numeric_fields': []
        })
        
        date_field = field_map['date_field']
        numeric_fields = field_map['numeric_fields']
        
        # 检查日期字段是否存在
        if date_field not in df.columns:
            return {
                'total_rows': len(df),
                'error': f"找不到日期字段 '{date_field}'",
                'is_valid': False
            }
        
        validation_results = {
            'total_rows': len(df),
            'date_range': f"{df[date_field].min()} to {df[date_field].max()}",
            'missing_dates': [],
            'abnormal_values': [],
            'data_quality': {},
            'logic_check': [],
            'is_valid': True
        }
        
        # 确保日期格式正确
        df[date_field] = pd.to_datetime(df[date_field])
        
        # 检查日期连续性
        if data_type not in ['top_list']:  # 龙虎榜数据不需要检查连续性
            date_range = pd.date_range(start=df[date_field].min(), end=df[date_field].max(), freq='B')
            missing_dates = set(date_range) - set(df[date_field])
            validation_results['missing_dates'] = list(missing_dates)
        
        # 检查异常值
        for col in numeric_fields:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mean = df[col].mean()
                std = df[col].std()
                abnormal = df[df[col] > mean + 3*std].index.tolist()
                if abnormal:
                    validation_results['abnormal_values'].append({
                        'column': col,
                        'dates': abnormal
                    })
        
        # 数据质量检查
        validation_results['data_quality'] = self._check_data_quality(df, numeric_fields)
        
        # 逻辑性检查
        if data_type == 'hs300_daily':
            validation_results['logic_check'] = self._check_data_logic(df)
        
        return validation_results
    
    def _check_data_quality(self, df, numeric_fields):
        """检查数据质量"""
        quality_results = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': len(df[df.duplicated()]),
            'zero_values': {
                col: len(df[df[col] == 0]) for col in numeric_fields if col in df.columns
            }
        }
        return quality_results
    
    def _check_data_logic(self, df):
        """检查数据逻辑性（仅适用于OHLC数据）"""
        logic_errors = []
        
        if all(field in df.columns for field in ['high', 'low', 'open', 'close']):
            # 检查OHLC逻辑
            high_low = df[df['high'] < df['low']]
            if not high_low.empty:
                logic_errors.append({
                    'type': 'high_lower_than_low',
                    'dates': high_low.index.tolist()
                })
            
            open_outside = df[(df['open'] > df['high']) | (df['open'] < df['low'])]
            if not open_outside.empty:
                logic_errors.append({
                    'type': 'open_outside_range',
                    'dates': open_outside.index.tolist()
                })
            
            close_outside = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
            if not close_outside.empty:
                logic_errors.append({
                    'type': 'close_outside_range',
                    'dates': close_outside.index.tolist()
                })
        
        return logic_errors
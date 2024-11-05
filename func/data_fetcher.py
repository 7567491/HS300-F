import akshare as ak
import baostock as bs
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from .database import DatabaseManager

class DataFetcher:
    def __init__(self):
        self.db = DatabaseManager()
        
        # 创建日志目录
        self.log_path = Path('data/cache')
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
    def setup_logging(self):
        log_file = self.log_path / 'fetch.log'
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def fetch_hs300_data(self, start_date, end_date):
        """获取沪深300指数数据"""
        try:
            # 获取数据
            df = ak.stock_zh_index_daily_em(symbol="sh000300")
            
            # 确保日期列格式正确
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 确保数值列的类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 按日期排序
            df = df.sort_values('date')
            
            # 保存到数据库
            self.db.save_data('hs300_daily', df)
            return True, df
            
        except Exception as e:
            logging.error(f"获取沪深300数据失败: {str(e)}")
            return False, str(e)
            
    def fetch_macro_data(self, start_date, end_date):
        """获取宏观经济指标"""
        try:
            # 获取CPI数据
            cpi_df = ak.macro_china_cpi_monthly()
            cpi_df['date'] = pd.to_datetime(cpi_df['月份']).dt.strftime('%Y-%m-%d')
            cpi_df = cpi_df.rename(columns={
                '当月同比': 'cpi_yoy'
            })
            
            # 获取PPI数据
            ppi_df = ak.macro_china_ppi_monthly()
            ppi_df['date'] = pd.to_datetime(ppi_df['月份']).dt.strftime('%Y-%m-%d')
            ppi_df = ppi_df.rename(columns={
                '当月同比': 'ppi_yoy'
            })
            
            # 获取M2数据
            m2_df = ak.macro_china_money_supply()
            m2_df['date'] = pd.to_datetime(m2_df['月份']).dt.strftime('%Y-%m-%d')
            m2_df = m2_df.rename(columns={
                'M2-数量': 'm2_amount',
                'M2-同比': 'm2_yoy'
            })
            
            # 合并数据
            dfs = [cpi_df, ppi_df, m2_df]
            df = dfs[0][['date']]
            for other_df in dfs[1:]:
                df = pd.merge(df, other_df[['date', other_df.columns[-1]]], 
                            on='date', how='outer')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 保存到数据库
            self.db.save_data('macro_indicators', df)
            return True, df
            
        except Exception as e:
            logging.error(f"获取宏观经济数据失败: {str(e)}")
            return False, str(e)

    def fetch_market_overview(self, start_date, end_date):
        """获取市场总貌数据"""
        try:
            # 使用akshare获取市场总貌数据
            df = ak.stock_zh_a_spot_em()  # 当日市场总貌
            
            # 打印原始列名，以便调试
            logging.info(f"Original columns: {df.columns.tolist()}")
            
            # 重命名字段，统一字段名称
            column_mapping = {
                '日期': 'date',  # 修改为实际的日期字段名
                '总市值': 'total_market_value',
                '成交量': 'trading_volume',
                '成交额': 'trading_amount',
                '上涨家数': 'up_count',
                '下跌家数': 'down_count'
            }
            
            # 检查列名是否存在
            existing_columns = set(df.columns) & set(column_mapping.keys())
            if not existing_columns:
                logging.error(f"No matching columns found. Available columns: {df.columns.tolist()}")
                return False, "数据字段不匹配"
            
            # 只重命名存在的列
            rename_dict = {k: column_mapping[k] for k in existing_columns}
            df = df.rename(columns=rename_dict)
            
            # 如果没有日期列，添加当前日期
            if 'date' not in df.columns:
                df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # 确保日期列格式正确
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 只保留需要的列
            needed_columns = ['date', 'total_market_value', 'trading_volume', 
                            'trading_amount', 'up_count', 'down_count']
            existing_needed_columns = [col for col in needed_columns if col in df.columns]
            df = df[existing_needed_columns]
            
            # 确保数值列的类型
            numeric_columns = [col for col in ['total_market_value', 'trading_volume', 
                                             'trading_amount', 'up_count', 'down_count'] 
                             if col in df.columns]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 按日期排序
            df = df.sort_values('date')
            
            # 保存到数据库
            self.db.save_data('market_overview', df)
            return True, df
            
        except Exception as e:
            logging.error(f"获取市场总貌数据失败: {str(e)}")
            return False, str(e)

    def fetch_margin_trading(self, start_date, end_date):
        """获取融资融券数据"""
        try:
            # 使用新的akshare接口：stock_margin_sz_summary_em 和 stock_margin_sh_summary_em
            df_sz = ak.stock_margin_sz_summary_em()  # 深圳融资融券
            df_sh = ak.stock_margin_sh_summary_em()  # 上海融资融券
            
            # 打印列名以便调试
            logging.info(f"深圳融资融券数据列名: {df_sz.columns.tolist()}")
            logging.info(f"上海融资融券数据列名: {df_sh.columns.tolist()}")
            
            # 统一日期列名
            if '日期' in df_sz.columns:
                df_sz = df_sz.rename(columns={'日期': 'date'})
            if '日期' in df_sh.columns:
                df_sh = df_sh.rename(columns={'日期': 'date'})
            
            # 合并数据
            df = pd.concat([df_sz, df_sh], ignore_index=True)
            df = df.groupby('date').sum().reset_index()
            
            # 重命名其他列
            column_mapping = {
                '融资余额': 'margin_balance',
                '融资买入额': 'margin_buy',
                '融券余量': 'margin_sell',
                '融资偿还额': 'margin_repay'
            }
            
            # 重命名存在的列
            existing_columns = set(df.columns) & set(column_mapping.keys())
            rename_dict = {k: column_mapping[k] for k in existing_columns}
            df = df.rename(columns=rename_dict)
            
            # 确保日期列格式正确
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 保存到数据库
            self.db.save_data('margin_trading', df)
            return True, df
            
        except Exception as e:
            logging.error(f"获取融资融券数据失败: {str(e)}")
            return False, str(e)

    def fetch_north_money(self, start_date, end_date):
        """获取北向资金数据"""
        try:
            # 使用新的akshare接口：stock_em_hsgt_north_net_flow_in
            df = ak.stock_em_hsgt_north_net_flow_in()
            
            # 打印列名以便调试
            logging.info(f"北向资金数据列名: {df.columns.tolist()}")
            
            # 重命名字段
            column_mapping = {
                '日期': 'date',
                '当日净流入': 'total_net_flow',
                '北上净流入': 'total_net_flow',
                '沪股通净流入': 'sh_net_flow',
                '深股通净流入': 'sz_net_flow'
            }
            
            # 检查列名是否存在并重命名
            existing_columns = set(df.columns) & set(column_mapping.keys())
            rename_dict = {k: column_mapping[k] for k in existing_columns}
            df = df.rename(columns=rename_dict)
            
            # 确保日期列格式正确
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 保存到数据库
            self.db.save_data('north_money', df)
            return True, df
            
        except Exception as e:
            logging.error(f"获取北向资金数据失败: {str(e)}")
            return False, str(e)

    def fetch_top_list(self, start_date, end_date):
        """获取龙虎榜数据"""
        try:
            # 使用新的akshare接口：stock_em_tfp_detail
            df = ak.stock_em_tfp_detail()  # 获取最新交易日龙虎榜数据
            
            # 打印列名以便调试
            logging.info(f"龙虎榜数据列名: {df.columns.tolist()}")
            
            # 重命名字段
            column_mapping = {
                '交易日期': 'date',
                '股票代码': 'stock_code',
                '股票名称': 'stock_name',
                '上榜原因': 'reason',
                '买入额': 'buy_amount',
                '卖出额': 'sell_amount',
                '净买额': 'net_amount'
            }
            
            # 检查列名是否存在并重命名
            existing_columns = set(df.columns) & set(column_mapping.keys())
            rename_dict = {k: column_mapping[k] for k in existing_columns}
            df = df.rename(columns=rename_dict)
            
            # 确保日期列格式正确
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            else:
                df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 保存到数据库
            self.db.save_data('top_list', df)
            return True, df
            
        except Exception as e:
            logging.error(f"获取龙虎榜数据���败: {str(e)}")
            return False, str(e)

    def fetch_market_sentiment(self, start_date, end_date):
        """获取市场情绪数据"""
        try:
            # 获取融资融券数据
            margin_df = ak.stock_margin_sz_summary_em()
            margin_df['date'] = pd.to_datetime(margin_df['日期']).dt.strftime('%Y-%m-%d')
            margin_df = margin_df.rename(columns={
                '融资余额': 'margin_balance',
                '融券余额': 'short_balance'
            })
            
            # 获取北向资金数据
            north_df = ak.stock_em_hsgt_north_net_flow_in()
            north_df['date'] = pd.to_datetime(north_df['日期']).dt.strftime('%Y-%m-%d')
            north_df = north_df.rename(columns={
                '当日净流入': 'north_money_net'
            })
            
            # 获取市场换手率
            turnover_df = ak.stock_zh_a_hist()
            turnover_df['date'] = pd.to_datetime(turnover_df['日期']).dt.strftime('%Y-%m-%d')
            turnover_df = turnover_df.rename(columns={
                '换手率': 'turnover_rate'
            })
            
            # 合并数据
            dfs = [margin_df, north_df, turnover_df]
            df = dfs[0][['date']]
            for other_df in dfs[1:]:
                df = pd.merge(df, other_df[['date', other_df.columns[-1]]], 
                            on='date', how='outer')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 保存到数据库
            self.db.save_data('market_sentiment', df)
            return True, df
            
        except Exception as e:
            logging.error(f"获取市场情绪数据失败: {str(e)}")
            return False, str(e)

    def fetch_market_structure(self, start_date, end_date):
        """获取市场结构指标"""
        try:
            # 获取市场总市值
            mv_df = ak.stock_zh_a_spot_em()
            mv_df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            mv_df = mv_df.rename(columns={
                '总市值': 'total_mv'
            })
            
            # 获取市盈率数据
            pe_df = ak.stock_a_pe_and_pb_em()
            pe_df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            pe_df = pe_df.rename(columns={
                '平均市盈率': 'pe_ratio'
            })
            
            # 获取股票账户数据
            account_df = ak.stock_account_statistics_em()
            account_df['date'] = pd.to_datetime(account_df['日期']).dt.strftime('%Y-%m-%d')
            account_df = account_df.rename(columns={
                '新增投资者-数量': 'new_accounts',
                '期末投资者-数量': 'total_accounts'
            })
            
            # 合并数据
            dfs = [mv_df, pe_df, account_df]
            df = dfs[0][['date']]
            for other_df in dfs[1:]:
                df = pd.merge(df, other_df[['date', other_df.columns[-1]]], 
                            on='date', how='outer')
            
            # 过滤日期范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # 保存到数据库
            self.db.save_data('market_structure', df)
            return True, df
            
        except Exception as e:
            logging.error(f"获取市场结构数据失败: {str(e)}")
            return False, str(e)
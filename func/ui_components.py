import streamlit as st
from .data_fetcher import DataFetcher
from .data_validator import DataValidator
from .database import DatabaseManager
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import numpy as np

class UIComponents:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.validator = DataValidator()
        self.db = DatabaseManager()
    
    def render_data_source_config(self):
        st.header("数据源配置")
        st.write("当前支持的数据源：")
        st.write("- AKShare (主要数据源)")
        st.write("- Baostock (备用数据源)")
        
        with st.expander("数据源参数设置"):
            start_date = st.date_input(
                "开始日期",
                value=pd.to_datetime("2009-11-01"),
                key="start_date"
            )
            end_date = st.date_input(
                "结束日期",
                value=pd.to_datetime("2024-11-01"),
                key="end_date"
            )
            return start_date, end_date
    
    def render_data_fetching(self):
        st.header("数据获取")
        
        if st.button(f"获取{st.session_state.current_data_type}数据"):
            with st.spinner(f"正在获取{st.session_state.current_data_type}数据..."):
                if st.session_state.current_data_type == "沪深300指数":
                    success, data = self.fetcher.fetch_hs300_data(
                        start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                        end_date=st.session_state.end_date.strftime("%Y-%m-%d")
                    )
                elif st.session_state.current_data_type == "市场总貌":
                    success, data = self.fetcher.fetch_market_overview(
                        start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                        end_date=st.session_state.end_date.strftime("%Y-%m-%d")
                    )
                elif st.session_state.current_data_type == "融资融券":
                    success, data = self.fetcher.fetch_margin_trading(
                        start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                        end_date=st.session_state.end_date.strftime("%Y-%m-%d")
                    )
                elif st.session_state.current_data_type == "北向资金":
                    success, data = self.fetcher.fetch_north_money(
                        start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                        end_date=st.session_state.end_date.strftime("%Y-%m-%d")
                    )
                elif st.session_state.current_data_type == "龙虎榜":
                    success, data = self.fetcher.fetch_top_list(
                        start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                        end_date=st.session_state.end_date.strftime("%Y-%m-%d")
                    )
                elif st.session_state.current_data_type == "宏观经济":
                    success, data = self.fetcher.fetch_macro_data(
                        start_date=st.session_state.start_date.strftime("%Y-%m-%d"),
                        end_date=st.session_state.end_date.strftime("%Y-%m-%d")
                    )
                
                if success:
                    st.success(f"{st.session_state.current_data_type}数据获取成功！")
                    st.dataframe(data.head() if isinstance(data, pd.DataFrame) else 
                               next(iter(data.values())).head())
                else:
                    st.error(f"数据获取失败: {data}")
    
    def render_data_validation(self):
        st.header("数据校验")
        
        # 数据类型映射
        type_map = {
            "沪深300指数": "hs300_daily",
            "市场总貌": "market_overview",
            "融资融券": "margin_trading",
            "北向资金": "north_money",
            "龙虎榜": "top_list",
            "市场情绪": "market_sentiment",
            "市场结构": "market_structure",
            "宏观经济": "macro_indicators"
        }
        
        if st.session_state.current_data_type == "宏观经济":
            st.info("宏观经济数据使用单独的校验规则")
            return
        
        if st.button("开始校验"):
            with st.spinner("正在校验数据..."):
                table_name = type_map[st.session_state.current_data_type]
                
                # 获取数据
                with sqlite3.connect(self.db.db_path) as conn:
                    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                
                # 执行校验，传入数据类型
                results = self.validator.validate_market_data(df, table_name)
                
                if results.get('error'):
                    st.error(results['error'])
                    return
                
                # 显示校验结果
                st.write("### 校验结果")
                
                # 基本信息
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("总数据条数", results['total_rows'])
                with col2:
                    st.metric("数据时间范围", results['date_range'])
                
                # 缺失日期检查
                st.subheader("数据连续性检查")
                if results['missing_dates']:
                    st.warning(f"发现 {len(results['missing_dates'])} 个交易日缺失数据")
                    missing_dates_df = pd.DataFrame(
                        sorted(results['missing_dates']), 
                        columns=['缺失日期']
                    )
                    st.dataframe(
                        missing_dates_df,
                        column_config={
                            "缺失日期": st.column_config.DateColumn(
                                "缺失日期",
                                format="YYYY-MM-DD"
                            )
                        }
                    )
                else:
                    st.success("数据连续性检查通过")
                
                # 异常值检查
                st.subheader("异常值检查")
                if results['abnormal_values']:
                    st.warning("发现异常值")
                    for abnormal in results['abnormal_values']:
                        st.write(f"字段 `{abnormal['column']}` 在以下日期出现异常值：")
                        abnormal_dates_df = pd.DataFrame(
                            sorted(abnormal['dates']), 
                            columns=['异常日期']
                        )
                        st.dataframe(
                            abnormal_dates_df,
                            column_config={
                                "异常日期": st.column_config.DateColumn(
                                    "异常日期",
                                    format="YYYY-MM-DD"
                                )
                            }
                        )
                else:
                    st.success("异常值检查通过")
                
                # 数据质量检查
                st.subheader("数据质量检查")
                quality = results['data_quality']
                
                # 缺失值
                if any(quality['missing_values'].values()):
                    st.warning("发现缺失值：")
                    missing_df = pd.DataFrame.from_dict(
                        quality['missing_values'], 
                        orient='index',
                        columns=['缺失数量']
                    )
                    st.dataframe(missing_df[missing_df['缺失数量'] > 0])
                else:
                    st.success("无缺失值")
                
                # 重复值
                if quality['duplicates'] > 0:
                    st.warning(f"发现 {quality['duplicates']} 条重复记录")
                else:
                    st.success("无重复记录")
                
                # 零值检查
                zero_values = quality['zero_values']
                if any(zero_values.values()):
                    st.warning("发现零值：")
                    zero_df = pd.DataFrame.from_dict(
                        zero_values, 
                        orient='index',
                        columns=['零值数量']
                    )
                    st.dataframe(zero_df[zero_df['零值数量'] > 0])
                else:
                    st.success("无零值")
                
                # 逻辑检查
                st.subheader("数据逻辑检查")
                if results['logic_check']:
                    st.warning("发现逻辑误：")
                    for error in results['logic_check']:
                        st.write(f"类型：{error['type']}")
                        error_dates_df = pd.DataFrame(
                            sorted(error['dates']), 
                            columns=['错误日期']
                        )
                        st.dataframe(
                            error_dates_df,
                            column_config={
                                "错误日期": st.column_config.DateColumn(
                                    "错误日期",
                                    format="YYYY-MM-DD"
                                )
                            }
                        )
                else:
                    st.success("数据逻辑检查通过")
    
    def render_data_overview(self):
        st.header("数据概览")
        
        # 数据类型映射
        type_map = {
            "沪深300指数": "hs300_daily",
            "市场总貌": "market_overview",
            "融资融券": "margin_trading",
            "北向资金": "north_money",
            "龙虎榜": "top_list"
        }
        
        try:
            if st.session_state.current_data_type == "宏观经济":
                self.render_macro_overview()
                return
                
            table_name = type_map[st.session_state.current_data_type]
            
            # 从数据库读取数据
            with sqlite3.connect(self.db.db_path) as conn:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
            
            # 基本统计信息
            with st.expander("基本统计信息"):
                st.write("### 沪深300指数统计信息")
                st.dataframe(df.describe())
            
            # 数据可视化
            with st.expander("数据可视化"):
                # K线图
                fig = go.Figure(data=[go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )])
                
                fig.update_layout(
                    title='沪深300指数K线图',
                    yaxis_title='价格',
                    xaxis_title='日期'
                )
                
                st.plotly_chart(fig)
                
                # 成交量图
                volume_fig = go.Figure(data=[go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name='成交量'
                )])
                
                volume_fig.update_layout(
                    title='沪深300成交量',
                    yaxis_title='成交量',
                    xaxis_title='日期'
                )
                
                st.plotly_chart(volume_fig)
            
            # 数据质量报告
            with st.expander("数据质量报告"):
                st.write("### 数据质量统计")
                total_days = (df['date'].max() - df['date'].min()).days
                trading_days = len(df)
                expected_trading_days = len(pd.bdate_range(df['date'].min(), df['date'].max()))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("时间范围统计：")
                    st.write(f"- 数据起始日期：{df['date'].min().strftime('%Y-%m-%d')}")
                    st.write(f"- 数据结束日期：{df['date'].max().strftime('%Y-%m-%d')}")
                    st.write(f"- 总天数（含周末）：{total_days}天")
                
                with col2:
                    st.write("交易日统计：")
                    st.write(f"- 预期交易天数：{expected_trading_days}天")
                    st.write(f"- 实际交易天数：{trading_days}天")
                    st.write(f"- 缺失交易日数：{expected_trading_days - trading_days}天")
                
                st.write("完整性统计：")
                st.write(f"- 日历完整率：{(trading_days/total_days*100):.2f}%（相对于总天数）")
                st.write(f"- 交易日完整率：{(trading_days/expected_trading_days*100):.2f}%（对于作日）")
                
                # 数据值完整性检查
                missing_values = df.isnull().sum()
                if missing_values.sum() > 0:
                    st.warning("字段缺失值统计：")
                    st.write(missing_values[missing_values > 0])
                else:
                    st.success("所有字段数据完整，无缺失值")
                
        except Exception as e:
            st.error(f"数据加载失败: {str(e)}")
    
    def render_prediction(self):
        st.header("指数预测")
        
        try:
            # 从数据库加载历史数据
            with sqlite3.connect(self.db.db_path) as conn:
                df = pd.read_sql("SELECT * FROM hs300_daily", conn)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
            
            if df.empty:
                st.warning("请先获取历史数据")
                return
            
            # 创建预测器实例
            from .predictor import MarketPredictor
            predictor = MarketPredictor()
            
            # 添加预测方式选择
            prediction_method = st.radio(
                "选择预测方式",
                ["机器学习预测", "拍脑袋预测"],
                horizontal=True
            )
            
            if prediction_method == "机器学习预测":
                # 原有的机器学习预测逻辑
                X, y, features = predictor.prepare_data(df)
                metrics, test_pred, y_test = predictor.train_model(X, y)
                
                # 显示模型评估指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("训练集RMSE", f"{metrics['train_rmse']:.2f}")
                with col2:
                    st.metric("测试集RMSE", f"{metrics['test_rmse']:.2f}")
                with col3:
                    st.metric("训练集R²", f"{metrics['train_r2']:.2f}")
                with col4:
                    st.metric("测试集R²", f"{metrics['test_r2']:.2f}")
                
                # 预测未来走势
                with st.spinner("正在预测未来走势..."):
                    future_df = predictor.predict_future(df)
                    fig = predictor.create_prediction_plot(df, future_df)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 添加预测区间信息
                st.divider()
                st.subheader("预测区间分析")
                
                # 计算预测的最大区间
                max_price = future_df['predicted_close'].max()
                min_price = future_df['predicted_close'].min()
                max_date = future_df['predicted_close'].idxmax()
                min_date = future_df['predicted_close'].idxmin()
                
                # 计算可信区间
                std_dev = df['close'].pct_change().std() * np.sqrt(252)  # 年化波动率
                current_price = df['close'].iloc[-1]
                confidence_upper = current_price * (1 + std_dev)
                confidence_lower = current_price * (1 - std_dev)
                
                # 显示区间信息
                col1, col2 = st.columns(2)
                with col1:
                    st.write("##### 预测最大区间")
                    st.write(f"最高点：{max_price:.2f} ({max_date.strftime('%Y年%m月%d日')})")
                    st.write(f"最低点：{min_price:.2f} ({min_date.strftime('%Y年%m月%d日')})")
                    st.write(f"区间范围：{(max_price - min_price):.2f}")
                    st.write(f"相对当前价格波动：{((max_price/current_price - 1) * 100):.2f}% / {((min_price/current_price - 1) * 100):.2f}%")
                
                with col2:
                    st.write("##### 一年期可信区间 (68% 置信度)")
                    st.write(f"上限：{confidence_upper:.2f}")
                    st.write(f"下限：{confidence_lower:.2f}")
                    st.write(f"区间范围：{(confidence_upper - confidence_lower):.2f}")
                    st.write(f"相对当前价格波动：±{(std_dev * 100):.2f}%")
                
                # 添加风险提示
                st.info("""
                📊 预测说明：
                - 最大区间表示模型预测的未来三年可能出现的最高和最低点
                - 可信区间基于历史波动率计算，表示在正态分布假设下，未来一年价格有68%的概率落在此区间内
                - 实际市场走势受多种因素影响，此预测仅供参考
                """)
            
            else:  # 拍脑袋预测
                with st.spinner("正在拍脑袋预测..."):
                    results = predictor.render_extreme_points_prediction(df)
                    
                    # 显示预测的极值点信息
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "预测最高点", 
                            f"{results['extreme_points']['high_price']:.2f}",
                            f"预计出现在 {results['extreme_points']['high_date'].strftime('%Y年%m月%d日')}"
                        )
                    with col2:
                        st.metric(
                            "预测最低点",
                            f"{results['extreme_points']['low_price']:.2f}",
                            f"预计出现在 {results['extreme_points']['low_date'].strftime('%Y年%m月%d日')}"
                        )
                    
                    # 显示预测图表
                    st.plotly_chart(results['figure'], use_container_width=True)
                
                # 添加区间分析
                st.divider()
                st.subheader("预测区间分析")
                
                # 计算当前价格
                current_price = df['close'].iloc[-1]
                
                # 显示区间信息
                col1, col2 = st.columns(2)
                with col1:
                    st.write("##### 预测最大区间")
                    high_price = results['extreme_points']['high_price']
                    low_price = results['extreme_points']['low_price']
                    st.write(f"最高点：{high_price:.2f} ({results['extreme_points']['high_date'].strftime('%Y年%m月%d日')})")
                    st.write(f"最低点：{low_price:.2f} ({results['extreme_points']['low_date'].strftime('%Y年%m月%d日')})")
                    st.write(f"区间范围：{(high_price - low_price):.2f}")
                    st.write(f"相对当前价格波动：{((high_price/current_price - 1) * 100):.2f}% / {((low_price/current_price - 1) * 100):.2f}%")
                
                with col2:
                    st.write("##### 波动特征")
                    historical_volatility = df['close'].pct_change().std() * np.sqrt(252)
                    st.write(f"历史年化波动率：{(historical_volatility * 100):.2f}%")
                    st.write(f"预期年化收益率：{(df['close'].pct_change().mean() * 252 * 100):.2f}%")
                    st.write(f"夏普比率：{(df['close'].pct_change().mean() * 252 - 0.03) / (historical_volatility):.2f}")
                
                # 添加市场周期分析
                st.divider()
                st.subheader("市场周期分析结论")
                
                # 计算当前趋势
                current_price = df['close'].iloc[-1]
                last_year_price = df['close'].iloc[-252]  # 一年前的价格
                price_change = (current_price / last_year_price - 1) * 100
                
                # 计算当前位置相对于预测高点的距离
                high_price = results['extreme_points']['high_price']
                high_date = results['extreme_points']['high_date']
                distance_to_high = (high_price / current_price - 1) * 100
                
                # 分析市场周期
                ma50 = df['close'].rolling(window=50).mean().iloc[-1]
                ma200 = df['close'].rolling(window=200).mean().iloc[-1]
                
                # 生成结论
                st.write("##### 🎯 市场周期研判")
                
                # 1. 牛市是否已经来临
                st.write("1️⃣ 牛市是否已经来临？")
                if current_price > ma200 and ma50 > ma200 and price_change > 20:
                    st.success("✅ 技术指标显示牛市已经启动")
                    st.write(f"- 年度涨幅达到 {price_change:.1f}%")
                    st.write("- 当前价格位于长期均线上方")
                    st.write("- 短期均线突破长期均线")
                elif current_price > ma200 and price_change > 0:
                    st.info("⚠️ 市场处于震荡上行阶段，可能是牛市初期")
                    st.write(f"- 年度涨幅为 {price_change:.1f}%")
                    st.write("- 需要进一步确认突破")
                else:
                    st.error("❌ 尚未出现明确的牛市信号")
                    st.write(f"- 年度涨幅为 {price_change:.1f}%")
                    st.write("- 需要等待市场转势")
                
                # 2. 牛市顶点时间预测
                st.write("\n2️⃣ 牛市顶点预测")
                days_to_high = (high_date - pd.Timestamp.now()).days
                if days_to_high > 0:
                    st.info(f"🎯 预计牛市顶点将出现在: {high_date.strftime('%Y年%m月%d日')}")
                    st.write(f"- 距离现在还有约 {days_to_high} 天")
                    st.write(f"- 对应时间点位于第 {(days_to_high//365)+1} 年")
                else:
                    st.warning("需要重新计算顶点时间")
                
                # 3. 牛市最高点预测
                st.write("\n3️⃣ 牛市最高点预测")
                st.info(f"🎯 预计最高点位: {high_price:.2f}")
                st.write(f"- 相对当前价格上涨空间: {distance_to_high:.1f}%")
                
                # 添加可信度评估
                confidence_score = min(100, max(0, 60 + 
                    (20 if price_change > 0 else 0) +
                    (10 if current_price > ma200 else 0) +
                    (10 if ma50 > ma200 else 0)))
                
                st.write("\n##### 🎲 预测可信度评估")
                st.progress(confidence_score/100)
                st.write(f"预测可信度: {confidence_score}%")
                
                # 风险提示
                st.warning("""
                ⚠️ 风险提示：
                1. 此预测基于历史数据模式和技术分析，不构成投资建议
                2. 市场受多种因素影响，实际走势可能与预测有重大差异
                3. 投资决策需考虑个人风险承受能力和市场环境
                """)
            
            # 显示预测数据表格
            with st.expander("查看预测数据"):
                display_df = (future_df if prediction_method == "机器学习预测" 
                             else results['predictions'])
                st.dataframe(
                    display_df.reset_index().rename(columns={
                        'index': '日期',
                        'predicted_close': '预测收盘价'
                    })
                )
            
            # 下载预测数据
            csv = display_df.to_csv()
            st.download_button(
                label="下载预测数据",
                data=csv,
                file_name="hs300_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"预测过程出错: {str(e)}")
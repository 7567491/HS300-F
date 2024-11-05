import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

class MarketPredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            random_state=42
        )
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df, target_col='close', window_size=30):
        """准备训练数据，创建技术指标和特征"""
        df = df.copy()
        
        # 计算技术指标
        # 移动平均线
        df['MA5'] = df[target_col].rolling(window=5).mean()
        df['MA10'] = df[target_col].rolling(window=10).mean()
        df['MA20'] = df[target_col].rolling(window=20).mean()
        df['MA30'] = df[target_col].rolling(window=30).mean()
        
        # RSI
        delta = df[target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df[target_col].ewm(span=12, adjust=False).mean()
        exp2 = df[target_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 波动率
        df['Volatility'] = df[target_col].rolling(window=20).std()
        
        # 价格动量
        df['Momentum'] = df[target_col].diff(periods=10)
        
        # 成交量指标
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
        
        # 删除NaN值
        df = df.dropna()
        
        # 创建特征矩阵和目标变量
        features = ['MA5', 'MA10', 'MA20', 'MA30', 'RSI', 'MACD', 'Signal_Line',
                   'Volatility', 'Momentum', 'Volume_MA5', 'Volume_MA20', 'volume']
        
        # 归一化特征
        X = self.scaler.fit_transform(df[features])
        y = df[target_col].values
        
        return X, y, features
    
    def train_model(self, X, y):
        """训练XGBoost模型"""
        # 使用80%的数据作为训练集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        return metrics, test_pred, y_test
    
    def predict_future(self, df, days=756):  # 756天约等于3年的交易日
        """预测未来走势"""
        X, y, features = self.prepare_data(df)
        
        # 训练模型
        self.train_model(X, y)
        
        # 准备预测的起始数据
        last_data = df.iloc[-30:].copy()  # 获取最后30天的数据
        
        # 存储预测结果
        future_predictions = []
        
        # 计算历史波动率
        historical_volatility = df['close'].pct_change().std()
        
        # 创建特征缩放器的副本
        feature_scaler = MinMaxScaler().fit(df[['close', 'volume', 'open', 'high', 'low']])
        
        # 逐日预测
        current_date = last_data.index[-1]
        
        for _ in range(days):
            try:
                # 计算技术指标
                current_data = last_data.copy()
                current_data['MA5'] = current_data['close'].rolling(window=5).mean()
                current_data['MA10'] = current_data['close'].rolling(window=10).mean()
                current_data['MA20'] = current_data['close'].rolling(window=20).mean()
                current_data['MA30'] = current_data['close'].rolling(window=30).mean()
                
                # RSI
                delta = current_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                current_data['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                exp1 = current_data['close'].ewm(span=12, adjust=False).mean()
                exp2 = current_data['close'].ewm(span=26, adjust=False).mean()
                current_data['MACD'] = exp1 - exp2
                current_data['Signal_Line'] = current_data['MACD'].ewm(span=9, adjust=False).mean()
                
                # 波动率
                current_data['Volatility'] = current_data['close'].rolling(window=20).std()
                
                # 价格动量
                current_data['Momentum'] = current_data['close'].diff(periods=10)
                
                # 成交量指标
                current_data['Volume_MA5'] = current_data['volume'].rolling(window=5).mean()
                current_data['Volume_MA20'] = current_data['volume'].rolling(window=20).mean()
                
                # 获取最后一行的特征
                features = ['MA5', 'MA10', 'MA20', 'MA30', 'RSI', 'MACD', 'Signal_Line',
                           'Volatility', 'Momentum', 'Volume_MA5', 'Volume_MA20', 'volume']
                current_features = current_data[features].iloc[-1:].values
                
                # 预测基准值
                if not np.isnan(current_features).any():  # 确保没有NaN值
                    base_pred = self.model.predict(current_features)[0]
                    
                    # 添加随机波动
                    random_volatility = np.random.normal(0, historical_volatility)
                    next_pred = base_pred * (1 + random_volatility)
                    
                    future_predictions.append(next_pred)
                    
                    # 更新current_date
                    current_date = current_date + pd.Timedelta(days=1)
                    while current_date.weekday() > 4:  # 跳过周末
                        current_date = current_date + pd.Timedelta(days=1)
                    
                    # 创建新的数据行
                    new_row = pd.DataFrame({
                        'close': [next_pred],
                        'volume': [last_data['volume'].mean()],
                        'open': [next_pred * (1 + np.random.normal(0, historical_volatility/2))],
                        'high': [next_pred * (1 + abs(np.random.normal(0, historical_volatility)))],
                        'low': [next_pred * (1 - abs(np.random.normal(0, historical_volatility)))]
                    }, index=[current_date])
                    
                    # 更新last_data
                    last_data = pd.concat([last_data[1:], new_row])
                
            except Exception as e:
                print(f"预测第{len(future_predictions)}天时出错: {str(e)}")
                continue
        
        # 创建未来日期序列
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=len(future_predictions), freq='B')
        
        # 创建预测数据框
        future_df = pd.DataFrame({
            'predicted_close': future_predictions
        }, index=future_dates)
        
        return future_df
    
    def create_prediction_plot(self, historical_df, future_df):
        """创建预测可视化图表"""
        fig = go.Figure()
        
        # 添加历史数据
        fig.add_trace(go.Scatter(
            x=historical_df.index,
            y=historical_df['close'],
            name='历史数据',
            line=dict(color='blue')
        ))
        
        # 计算预测的置信区间
        std_dev = historical_df['close'].pct_change().std() * np.sqrt(252)  # 年化波动率
        upper_bound = future_df['predicted_close'] * (1 + std_dev)
        lower_bound = future_df['predicted_close'] * (1 - std_dev)
        
        # 添加置信区间
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=upper_bound,
            name='上限',
            line=dict(color='rgba(255,0,0,0.2)', dash='dash'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=lower_bound,
            name='下限',
            line=dict(color='rgba(255,0,0,0.2)', dash='dash'),
            fill='tonexty',
            showlegend=False
        ))
        
        # 添加预测数据
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df['predicted_close'],
            name='预测数据',
            line=dict(color='red')
        ))
        
        # 更新布局
        fig.update_layout(
            title='沪深300指数预测（含置信区间）',
            xaxis_title='日期',
            yaxis_title='指数值',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig 

    def predict_extreme_points(self, df, days=756):
        """预测未来3年可能的最高点和最低点"""
        # 分析历史数据的季节性和周期性
        df = df.copy()
        
        # 计算历史上的周期性特征
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        df['year_week'] = df.index.isocalendar().week
        
        # 分析月度表现
        monthly_returns = df.groupby('month')['close'].agg(['mean', 'std'])
        best_month = monthly_returns['mean'].idxmax()
        worst_month = monthly_returns['mean'].idxmin()
        
        # 分析周度表现
        weekly_returns = df.groupby('year_week')['close'].agg(['mean', 'std'])
        best_week = weekly_returns['mean'].idxmax()
        worst_week = weekly_returns['mean'].idxmin()
        
        # 计算历史波动特征
        historical_volatility = df['close'].pct_change().std() * np.sqrt(252)
        current_price = df['close'].iloc[-1]
        
        # 生成未来3年的日期序列
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=days, freq='B')
        
        # 预测最高点和最低点的时间
        # 使用历史数据的模式来预测
        high_year = np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25])  # 偏向于较近的时间
        low_year = np.random.choice([1, 2, 3], p=[0.25, 0.35, 0.4])   # 偏向于较远的时间
        
        # 确保高点和低点不在同一年
        if high_year == low_year:
            if high_year < 3:
                low_year = high_year + 1
            else:
                high_year = low_year - 1
        
        # 生成具体日期
        high_date = last_date + pd.DateOffset(years=high_year, months=best_month-1, weeks=1)
        low_date = last_date + pd.DateOffset(years=low_year, months=worst_month-1, weeks=1)
        
        # 预测极值
        expected_return = df['close'].pct_change().mean() * 252  # 年化收益率
        high_price = current_price * (1 + expected_return * high_year + historical_volatility)
        low_price = current_price * (1 - historical_volatility + expected_return * (low_year - 1))
        
        # 添加随机性
        high_price *= (1 + np.random.normal(0, 0.1))  # 添加10%的随机波动
        low_price *= (1 + np.random.normal(0, 0.1))   # 添加10%的随机波动
        
        return {
            'high_date': high_date,
            'low_date': low_date,
            'high_price': high_price,
            'low_price': low_price
        }

    def predict_with_extreme_points(self, df, extreme_points, days=756):
        """基于预测的极值点生成完整的预测序列"""
        # 获取当前价格和日期
        current_price = df['close'].iloc[-1]
        last_date = df.index[-1]
        
        # 创建时间序列
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=days, freq='B')
        
        # 将极值点转换为相对于起始日期的天数
        high_days = (extreme_points['high_date'] - last_date).days
        low_days = (extreme_points['low_date'] - last_date).days
        
        # 创建价格序列
        prices = []
        historical_volatility = df['close'].pct_change().std()
        
        for i in range(days):
            # 计算当前日期到高点和低点的距离
            current_date = last_date + timedelta(days=i)
            dist_to_high = abs((current_date - extreme_points['high_date']).days)
            dist_to_low = abs((current_date - extreme_points['low_date']).days)
            
            # 使用距离加权计算目标价格
            weight_high = 1 / (1 + dist_to_high/100)
            weight_low = 1 / (1 + dist_to_low/100)
            
            # 计算基准价格
            base_price = (extreme_points['high_price'] * weight_high + 
                         extreme_points['low_price'] * weight_low) / (weight_high + weight_low)
            
            # 添加随机波动
            random_volatility = np.random.normal(0, historical_volatility)
            price = base_price * (1 + random_volatility)
            prices.append(price)
        
        # 创建预测数据框
        future_df = pd.DataFrame({
            'predicted_close': prices
        }, index=future_dates)
        
        return future_df

    def render_extreme_points_prediction(self, df):
        """生成极值点预测的可视化和描述"""
        # 预测极值点
        extreme_points = self.predict_extreme_points(df)
        
        # 基于极值点生成预测序列
        future_df = self.predict_with_extreme_points(df, extreme_points)
        
        # 创建可视化
        fig = go.Figure()
        
        # 添加历史数据
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            name='历史数据',
            line=dict(color='blue')
        ))
        
        # 添加预测数据
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df['predicted_close'],
            name='预测数据',
            line=dict(color='red', dash='dash')
        ))
        
        # 标记极值点
        fig.add_trace(go.Scatter(
            x=[extreme_points['high_date']],
            y=[extreme_points['high_price']],
            mode='markers+text',
            name='预测最高点',
            marker=dict(color='red', size=12, symbol='triangle-up'),
            text=['预测最高点'],
            textposition='top center'
        ))
        
        fig.add_trace(go.Scatter(
            x=[extreme_points['low_date']],
            y=[extreme_points['low_price']],
            mode='markers+text',
            name='预测最低点',
            marker=dict(color='green', size=12, symbol='triangle-down'),
            text=['预测最低点'],
            textposition='bottom center'
        ))
        
        # 更新布局
        fig.update_layout(
            title='沪深300指数"拍脑袋"预测（含极值点）',
            xaxis_title='日期',
            yaxis_title='指数值',
            hovermode='x unified',
            showlegend=True
        )
        
        return {
            'figure': fig,
            'extreme_points': extreme_points,
            'predictions': future_df
        }
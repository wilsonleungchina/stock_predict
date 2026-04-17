# -*- coding: utf-8 -*-
"""
股票涨跌预测器
目标：基于历史K线特征，预测 T+10 日涨跌概率（≥2% 判定为涨）
数据源：腾讯财经日K（主）→ 新浪日K（备用）
模型：XGBoost + 随机森林 + 逻辑回归，对比选优
规则：禁止硬编码数据，取数失败明确报错
"""

import sys
import io
import json
import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# ══════════════════════════════════════════════════════════════════
#  外部依赖检查
# ══════════════════════════════════════════════════════════════════

MISSING_DEPS = []
try:
    import xgboost as xgb
except ImportError:
    MISSING_DEPS.append('xgboost')
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
except ImportError:
    MISSING_DEPS.append('scikit-learn')

if MISSING_DEPS:
    print(f"[警告] 缺少依赖库: {', '.join(MISSING_DEPS)}")
    print(f"安装命令: pip install {' '.join(MISSING_DEPS)}")


# ══════════════════════════════════════════════════════════════════
#  请求头（复用了 real_stock_fetcher.py 的配置）
# ══════════════════════════════════════════════════════════════════

_TENCENT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/124.0.0.0 Safari/537.36',
    'Referer': 'https://gu.qq.com/',
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

_SINA_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/124.0.0.0 Safari/537.36',
    'Referer': 'https://finance.sina.com.cn/',
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}


# ══════════════════════════════════════════════════════════════════
#  数据获取层
# ══════════════════════════════════════════════════════════════════

class KLineFetcher:
    """
    历史K线数据获取器
    主接口：腾讯财经日K（可一次拉满500条，覆盖2年）
    备用接口：新浪日K（每次最多100条，用于补充）
    """

    def __init__(self):
        self._tencent_ok = True
        self._sina_ok = True

    # ── 腾讯财经日K（主）─────────────────────────────────────────

    def get_daily_tencent(self, stock_code: str, count: int = 500) -> pd.DataFrame:
        """
        获取日K线（前复权），使用腾讯财经接口。
        stock_code: 完整代码如 'sz002837' / 'sh600519'
        count: 最多500条（腾讯接口上限）
        返回: DataFrame，含 [date, open, high, low, close, volume]
        """
        today = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=count * 2)).strftime('%Y-%m-%d')
        url = (f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
               f'?_var=kline_dayhfq&param={stock_code},day,{start},{today},{count},qfq')

        try:
            resp = requests.get(url, headers=_TENCENT_HEADERS, timeout=15)
            if resp.status_code != 200:
                self._tencent_ok = False
                raise RuntimeError(f'腾讯日K HTTP {resp.status_code}')

            text = resp.text
            if text.startswith('kline_dayhfq='):
                text = text[len('kline_dayhfq='):]
            data = json.loads(text)

            stock_data = data.get('data', {}).get(stock_code, {})
            # 优先取前复权数据 qfqday，其次取普通 day
            bars = stock_data.get('qfqday') or stock_data.get('day')
            if not bars:
                self._tencent_ok = False
                raise RuntimeError(f'腾讯日K返回无数据字段: {stock_code}')

            # 解析：['date', 'open', 'close', 'high', 'low', 'volume']
            records = []
            for bar in bars:
                try:
                    records.append({
                        'date':   bar[0],
                        'open':   float(bar[1]),
                        'close':  float(bar[2]),
                        'high':   float(bar[3]),
                        'low':    float(bar[4]),
                        'volume': float(bar[5]),
                    })
                except (ValueError, IndexError):
                    continue

            df = pd.DataFrame(records)
            if df.empty:
                raise RuntimeError('腾讯日K解析后数据为空')
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df

        except Exception as e:
            self._tencent_ok = False
            raise RuntimeError(f'腾讯日K接口异常 [{stock_code}]: {e}')

    # ── 新浪财经日K（备用）───────────────────────────────────────

    def get_daily_sina(self, stock_code: str, count: int = 500) -> pd.DataFrame:
        """
        获取日K线，使用新浪财经接口（备用，每次最多100条，需分批）。
        stock_code: 完整代码如 'sz002837' / 'sh600519'
        返回: DataFrame
        """
        all_records = []
        batch_size = 100
        batches = math.ceil(count / batch_size)

        for i in range(batches):
            url = (f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php'
                   f'/CN_MarketData.getKLineData?symbol={stock_code}'
                   f'&scale=240&ma=5&datalen={batch_size}')
            try:
                resp = requests.get(url, headers=_SINA_HEADERS, timeout=12)
                if resp.status_code != 200:
                    self._sina_ok = False
                    raise RuntimeError(f'新浪日K HTTP {resp.status_code}')

                bars = json.loads(resp.text)
                if not bars:
                    break

                for bar in bars:
                    try:
                        all_records.append({
                            'date':   bar['day'],
                            'open':   float(bar['open']),
                            'close':  float(bar['close']),
                            'high':   float(bar['high']),
                            'low':    float(bar['low']),
                            'volume': float(bar['volume']),
                        })
                    except (ValueError, KeyError):
                        continue

                time.sleep(0.2)
            except Exception as e:
                self._sina_ok = False
                raise RuntimeError(f'新浪日K接口异常 [{stock_code}]: {e}')

        if not all_records:
            raise RuntimeError(f'新浪日K无数据: {stock_code}')

        df = pd.DataFrame(all_records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # 取最新的 count 条
        return df.tail(count).reset_index(drop=True)

    # ── 统一入口 ────────────────────────────────────────────────

    def get_daily(self, stock_code: str, count: int = 500) -> pd.DataFrame:
        """
        获取日K线，自动按优先级尝试。
        先腾讯（主），失败则新浪（备用）。
        """
        errors = []

        if self._tencent_ok:
            try:
                df = self.get_daily_tencent(stock_code, count)
                print(f"  [数据] 腾讯日K → {len(df)} 条 ({df['date'].min().date()} ~ {df['date'].max().date()})")
                return df
            except Exception as e:
                errors.append(str(e))
                print(f"  [警告] 腾讯日K失败，尝试新浪: {e}")

        # 降级新浪
        self._sina_ok = True
        try:
            df = self.get_daily_sina(stock_code, count)
            print(f"  [数据] 新浪日K → {len(df)} 条 ({df['date'].min().date()} ~ {df['date'].max().date()})")
            return df
        except Exception as e:
            errors.append(str(e))
            raise RuntimeError(f'所有日K接口均失败: {"; ".join(errors)}')


# ══════════════════════════════════════════════════════════════════
#  特征工程层
# ══════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    基于K线数据构造机器学习特征。
    所有特征均从真实K线数据计算得出，无硬编码。
    """

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        输入日K DataFrame，输出特征 DataFrame。
        计算以下技术指标作为特征：
          - 价格类：收益率、N日收益率、均线乖离率
          - 动量类：RSI、MACD、KDJ
          - 波动类：布林带位置ATR
          - 成交量类：量比、量能变化率
        """
        df = df.copy()

        # ── 基础价格特征 ────────────────────────────────────────
        df['return_1d']  = df['close'].pct_change(1)
        df['return_5d']  = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        # ── 均线乖离率 ──────────────────────────────────────────
        for window in [5, 10, 20]:
            ma = df['close'].rolling(window).mean()
            df[f'bias_{window}'] = (df['close'] - ma) / ma * 100

        # ── RSI（相对强弱指标）─────────────────────────────────
        for window in [6, 12, 24]:
            delta = df['close'].diff()
            gain  = delta.where(delta > 0, 0).rolling(window).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs    = gain / loss.replace(0, np.nan)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))

        # ── MACD ───────────────────────────────────────────────
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd_dif']  = ema12 - ema26
        df['macd_dea']  = df['macd_dif'].ewm(span=9).mean()
        df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2

        # ── KDJ ─────────────────────────────────────────────────
        for n in [9, 14]:
            low_n  = df['low'].rolling(n).min()
            high_n = df['high'].rolling(n).max()
            rsv = (df['close'] - low_n) / (high_n - low_n + 1e-9) * 100
            df[f'k_{n}'] = rsv.ewm(alpha=1/3).mean()
            df[f'd_{n}'] = df[f'k_{n}'].ewm(alpha=1/3).mean()
            df[f'j_{n}'] = 3 * df[f'k_{n}'] - 2 * df[f'd_{n}']

        # ── 布林带 ─────────────────────────────────────────────
        for window in [10, 20]:
            mid = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'bb_upper_{window}'] = mid + 2 * std
            df[f'bb_lower_{window}'] = mid - 2 * std
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (
                df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'] + 1e-9)

        # ── ATR（真实波幅）─────────────────────────────────────
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low']  - df['close'].shift(1))
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()

        # ── 成交量特征 ──────────────────────────────────────────
        df['vol_ma5']  = df['volume'].rolling(5).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)
        df['vol_change'] = df['volume'].pct_change(5)

        # ── 波动率 ──────────────────────────────────────────────
        df['volatility_5']  = df['return_1d'].rolling(5).std()
        df['volatility_20'] = df['return_1d'].rolling(20).std()

        # ── 价格位置 ────────────────────────────────────────────
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20']  = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (
            df['high_20'] - df['low_20'] + 1e-9)

        # ── 目标变量（T+10 涨跌）────────────────────────────────
        #  T+10 日收盘价相对今日收盘价涨幅 ≥2% → 1（涨），否则 0（不涨）
        df['future_close'] = df['close'].shift(-10)
        df['future_return'] = (df['future_close'] - df['close']) / df['close'] * 100
        df['target'] = (df['future_return'] >= 2.0).astype(int)

        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """
        返回所有用于模型训练的特征列名（不含目标变量和未来数据）。
        """
        return [
            # 价格/动量
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'bias_5', 'bias_10', 'bias_20',
            'rsi_6', 'rsi_12', 'rsi_24',
            'macd_dif', 'macd_dea', 'macd_hist',
            'k_9', 'd_9', 'j_9',
            'k_14', 'd_14', 'j_14',
            # 布林带/ATR
            'bb_position_10', 'bb_position_20',
            'atr_14',
            # 成交量
            'vol_ratio', 'vol_change',
            # 波动率/位置
            'volatility_5', 'volatility_20',
            'price_position',
        ]


# ══════════════════════════════════════════════════════════════════
#  模型训练层
# ══════════════════════════════════════════════════════════════════

class StockPredictor:
    """
    股票涨跌预测器。
    支持 XGBoost、随机森林、逻辑回归 三模型对比输出 T+10 涨跌概率。
    """

    def __init__(self):
        self._scaler = StandardScaler()
        self._feature_cols = FeatureEngineer.get_feature_columns()
        self._models = {}
        self._trained = False

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征清洗：
          1. 移除含 NaN 的行（前20行因滚动窗口无法计算）
          2. 移除目标变量为 NaN 的行（最后10行无未来数据）
          3. 只保留有特征列的行
        """
        feature_df = df[self._feature_cols + ['target']].copy()
        feature_df = feature_df.dropna(subset=self._feature_cols)
        # 最后10行 target 可能为 NaN（无未来数据），保留用于预测
        X = feature_df[self._feature_cols]
        y = feature_df['target']
        # y 中 NaN 行是最后10行，这里删除
        valid = y.notna()
        return X[valid], y[valid]

    def train(self, df: pd.DataFrame) -> Dict:
        """
        训练三模型并输出评估指标。
        返回：各模型的评估结果字典
        """
        if len(df) < 100:
            raise ValueError(f'数据不足（{len(df)}条），至少需要100条K线')

        X, y = self._prepare_data(df)

        # 剔除目标变量全为0或全为1的异常情况
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            raise ValueError('正负样本数过少（<5），模型训练无意义')

        # 时序分割：保留最后60天作测试集（模拟真实预测场景）
        split_idx = len(X) - 60
        if split_idx < 100:
            split_idx = int(len(X) * 0.8)

        X_train = X.iloc[:split_idx]
        X_test  = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test  = y.iloc[split_idx:]

        print(f"  [训练] 训练集: {len(X_train)} 样本 | 测试集: {len(X_test)} 样本")
        print(f"  [样本] 训练集正例(y=1)占比: {y_train.mean()*100:.1f}%")

        # 标准化
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled  = self._scaler.transform(X_test)

        results = {}

        # ── XGBoost ─────────────────────────────────────────────
        try:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
                eval_metric='auc',
                random_state=42,
            )
            xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
            xgb_pred  = (xgb_proba >= 0.5).astype(int)
            results['XGBoost'] = {
                'model':      xgb_clf,
                'accuracy':   accuracy_score(y_test, xgb_pred),
                'auc':        roc_auc_score(y_test, xgb_proba),
                'proba':      xgb_proba,
                'feature_importance': dict(zip(self._feature_cols, xgb_clf.feature_importances_)),
            }
            print(f"  [XGBoost]  准确率: {results['XGBoost']['accuracy']:.3f}  AUC: {results['XGBoost']['auc']:.3f}")
        except Exception as e:
            print(f"  [XGBoost] 训练失败: {e}")

        # ── 随机森林 ────────────────────────────────────────────
        try:
            rf_clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )
            rf_clf.fit(X_train, y_train)
            rf_proba = rf_clf.predict_proba(X_test)[:, 1]
            rf_pred  = (rf_proba >= 0.5).astype(int)
            results['RandomForest'] = {
                'model':      rf_clf,
                'accuracy':   accuracy_score(y_test, rf_pred),
                'auc':        roc_auc_score(y_test, rf_proba),
                'proba':      rf_proba,
                'feature_importance': dict(zip(self._feature_cols, rf_clf.feature_importances_)),
            }
            print(f"  [RandomForest] 准确率: {results['RandomForest']['accuracy']:.3f}  AUC: {results['RandomForest']['auc']:.3f}")
        except Exception as e:
            print(f"  [RandomForest] 训练失败: {e}")

        # ── 逻辑回归 ────────────────────────────────────────────
        try:
            lr_clf = LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
            )
            lr_clf.fit(X_train_scaled, y_train)
            lr_proba = lr_clf.predict_proba(X_test_scaled)[:, 1]
            lr_pred  = (lr_proba >= 0.5).astype(int)
            results['LogisticRegression'] = {
                'model':      lr_clf,
                'accuracy':   accuracy_score(y_test, lr_pred),
                'auc':        roc_auc_score(y_test, lr_proba),
                'proba':      lr_proba,
                'feature_importance': None,
            }
            print(f"  [LogisticRegression] 准确率: {results['LogisticRegression']['accuracy']:.3f}  AUC: {results['LogisticRegression']['auc']:.3f}")
        except Exception as e:
            print(f"  [LogisticRegression] 训练失败: {e}")

        self._models = results
        self._trained = True
        return results

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        使用训练好的模型，对最近交易日做预测。
        返回：各模型 T+10 涨跌概率，以及综合推荐。
        """
        if not self._trained:
            raise RuntimeError('请先调用 train() 训练模型')

        # 取最后一条有完整特征的数据作为预测样本
        feature_df = df[self._feature_cols + ['target']].copy()
        feature_df = feature_df.dropna(subset=self._feature_cols)
        last_row = feature_df[self._feature_cols].iloc[-1:].copy()

        predictions = {}
        for name, result in self._models.items():
            model = result['model']
            if name == 'LogisticRegression':
                X_scaled = self._scaler.transform(last_row)
                proba = model.predict_proba(X_scaled)[0, 1]
            else:
                proba = model.predict_proba(last_row)[0, 1]

            predictions[name] = {
                'prob_up':   round(proba * 100, 1),   # 涨(≥2%)概率%
                'prob_down': round((1 - proba) * 100, 1),  # 不涨(跌)概率%
            }
            print(f"  [{name}] T+10上涨(≥2%)概率: {predictions[name]['prob_up']}%")

        # 综合推荐：取三模型概率均值，附各模型结果
        avg_prob = sum(p['prob_up'] for p in predictions.values()) / len(predictions)

        # 最优模型（基于测试集AUC）
        best_model = max(
            self._models.items(),
            key=lambda x: x[1]['auc']
        )[0]

        return {
            'predictions':      predictions,
            'avg_prob_up':      round(avg_prob, 1),
            'best_model':       best_model,
            'best_model_auc':   round(self._models[best_model]['auc'], 3),
            'last_close':       df['close'].iloc[-1],
            'last_date':        df['date'].iloc[-1].strftime('%Y-%m-%d'),
        }


# ══════════════════════════════════════════════════════════════════
#  主程序入口
# ══════════════════════════════════════════════════════════════════

def analyze_stock(stock_code: str, stock_name: str = None,
                  lookback_days: int = 500, predict_days: int = 10,
                  up_threshold: float = 2.0):
    """
    股票走势预测主函数。
    参数:
      stock_code: 股票代码含前缀，如 'sz002837'（英维克）
      stock_name: 显示名称，如 '英维克'
      lookback_days: 历史K线天数（默认500，约2年）
      predict_days: 预测周期（默认T+10）
      up_threshold: 上涨判定阈值%（默认≥2%）
    返回: 预测结果字典
    """
    name = stock_name or stock_code
    print("=" * 60)
    print(f"股票走势预测: {name} ({stock_code})")
    print(f"预测参数: T+{predict_days} 上涨阈值≥{up_threshold}%")
    print("=" * 60)

    # 1. 获取数据
    fetcher = KLineFetcher()
    df = fetcher.get_daily(stock_code, count=lookback_days)
    print(f"  [数据] 共获取 {len(df)} 条日K线")
    print(f"  [数据] 时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

    if len(df) < lookback_days * 0.7:
        print(f"  [警告] 数据量偏少（{len(df)} 条），模型效果可能受影响")

    # 2. 特征工程
    print("\n[特征工程] 计算技术指标...")
    df_features = FeatureEngineer.build_features(df)
    print(f"  [特征] 共生成 {len(FeatureEngineer.get_feature_columns())} 个特征")

    # 3. 训练模型
    print("\n[模型训练] 训练三模型对比...")
    predictor = StockPredictor()
    try:
        results = predictor.train(df_features)
    except ValueError as e:
        print(f"  [错误] 训练终止: {e}")
        return {'error': str(e)}

    # 4. 预测
    print("\n[T+10 预测] 最新交易日:", end=' ')
    pred_result = predictor.predict(df_features)
    print()

    # 5. 汇总输出
    print("\n" + "=" * 60)
    print(f"  股票: {name} ({stock_code})")
    print(f"  最新收盘价: {pred_result['last_close']:.2f}")
    print(f"  数据截止: {pred_result['last_date']}")
    print(f"  最优模型: {pred_result['best_model']} (AUC={pred_result['best_model_auc']})")
    print(f"  ── T+{predict_days} 涨跌概率 ──")
    for model_name, pred in pred_result['predictions'].items():
        marker = ' ★' if model_name == pred_result['best_model'] else ''
        print(f"    {model_name:<20s} 上涨概率: {pred['prob_up']}%  |  不涨概率: {pred['prob_down']}%{marker}")
    print(f"  ──────────────────────────────")
    avg = pred_result['avg_prob_up']
    color = '🔴' if avg >= 60 else ('🟡' if avg >= 40 else '🟢')
    trend = '偏涨' if avg >= 55 else ('偏跌' if avg <= 45 else '中性')
    print(f"  综合概率(均值): {avg}%  [{trend}]  {color}")
    print("=" * 60)

    return pred_result


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 英维克 sz002837
    result = analyze_stock('sz002837', '英维克')

    if 'error' not in result:
        print("\n模型对比摘要:")
        print(f"  综合 T+10 上涨概率: {result['avg_prob_up']}%")
        print(f"  推荐参考模型: {result['best_model']} (AUC={result['best_model_auc']})")
        for m, p in result['predictions'].items():
            print(f"    {m}: {p['prob_up']}%")

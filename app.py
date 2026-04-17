# -*- coding: utf-8 -*-
"""
股票趋势预测 Web 应用
基于 Streamlit 框架，支持在线查询任意A股走势概率
数据源：腾讯财经日K（前复权）→ 新浪日K（备用）
模型：XGBoost + 随机森林 + 逻辑回归，对比选优

部署：Streamlit Cloud（GitHub 连接后自动部署）
依赖：pip install -r requirements.txt
"""

import sys
import io
import json
import time
import math
import warnings

import requests
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  Streamlit 配置（必须放在最前面）
# ══════════════════════════════════════════════════════════════════

import streamlit as st

st.set_page_config(
    page_title="股票趋势预测器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  ML 模型 & 依赖（延迟导入，避免部署环境缺失时页面崩溃）
# ══════════════════════════════════════════════════════════════════

_deps_ok = True
_model_import_error = None
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError as e:
    _deps_ok = False
    _model_import_error = str(e)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 后端
    PLT_OK = True
except Exception:
    PLT_OK = False

# ══════════════════════════════════════════════════════════════════
#  数据获取层
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


@st.cache_data(ttl=3600)
def get_daily_kline(stock_code: str, count: int = 500) -> pd.DataFrame:
    """
    获取日K线（前复权），自动按优先级尝试。
    先腾讯（主），失败则新浪（备用）。
    """
    errors = []
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    start = (pd.Timestamp.now() - pd.Timedelta(days=count * 2)).strftime('%Y-%m-%d')

    # ── 腾讯财经日K（主）─────────────────────────────────────
    url = (f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
           f'?_var=kline_dayhfq&param={stock_code},day,{start},{today},{count},qfq')
    try:
        resp = requests.get(url, headers=_TENCENT_HEADERS, timeout=15)
        if resp.status_code == 200:
            text = resp.text
            if text.startswith('kline_dayhfq='):
                text = text[len('kline_dayhfq='):]
            data = json.loads(text)
            stock_data = data.get('data', {}).get(stock_code, {})
            bars = stock_data.get('qfqday') or stock_data.get('day')
            if bars:
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
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                if not df.empty:
                    return df
    except Exception as e:
        errors.append(f'腾讯: {e}')

    # ── 新浪财经日K（备用）───────────────────────────────────
    all_records = []
    batch_size = 100
    batches = math.ceil(count / batch_size)
    for i in range(batches):
        url = (f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php'
               f'/CN_MarketData.getKLineData?symbol={stock_code}'
               f'&scale=240&ma=5&datalen={batch_size}')
        try:
            resp = requests.get(url, headers=_SINA_HEADERS, timeout=12)
            if resp.status_code == 200:
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
            errors.append(f'新浪: {e}')
            break

    if not all_records:
        raise RuntimeError(f'所有日K接口均失败: {"; ".join(errors)}')

    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df.tail(count).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  特征工程
# ══════════════════════════════════════════════════════════════════

_FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'bias_5', 'bias_10', 'bias_20',
    'rsi_6', 'rsi_12', 'rsi_24',
    'macd_dif', 'macd_dea', 'macd_hist',
    'k_9', 'd_9', 'j_9',
    'k_14', 'd_14', 'j_14',
    'bb_position_10', 'bb_position_20',
    'atr_14',
    'vol_ratio', 'vol_change',
    'volatility_5', 'volatility_20',
    'price_position',
]


def build_features(df: pd.DataFrame, up_threshold: float = 2.0,
                   predict_days: int = 10) -> pd.DataFrame:
    """计算技术指标特征"""
    df = df.copy()

    # 价格特征
    df['return_1d']  = df['close'].pct_change(1)
    df['return_5d']  = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['return_20d'] = df['close'].pct_change(20)

    # 均线乖离率
    for w in [5, 10, 20]:
        ma = df['close'].rolling(w).mean()
        df[f'bias_{w}'] = (df['close'] - ma) / ma * 100

    # RSI
    for w in [6, 12, 24]:
        delta = df['close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(w).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f'rsi_{w}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd_dif']  = ema12 - ema26
    df['macd_dea']  = df['macd_dif'].ewm(span=9).mean()
    df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2

    # KDJ
    for n in [9, 14]:
        low_n  = df['low'].rolling(n).min()
        high_n = df['high'].rolling(n).max()
        rsv = (df['close'] - low_n) / (high_n - low_n + 1e-9) * 100
        df[f'k_{n}'] = rsv.ewm(alpha=1/3).mean()
        df[f'd_{n}'] = df[f'k_{n}'].ewm(alpha=1/3).mean()
        df[f'j_{n}'] = 3 * df[f'k_{n}'] - 2 * df[f'd_{n}']

    # 布林带
    for w in [10, 20]:
        mid = df['close'].rolling(w).mean()
        std = df['close'].rolling(w).std()
        df[f'bb_upper_{w}'] = mid + 2 * std
        df[f'bb_lower_{w}'] = mid - 2 * std
        df[f'bb_position_{w}'] = (df['close'] - df[f'bb_lower_{w}']) / (
            df[f'bb_upper_{w}'] - df[f'bb_lower_{w}'] + 1e-9)

    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low']  - df['close'].shift(1))
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = df['tr'].rolling(14).mean()

    # 成交量
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)
    df['vol_change'] = df['volume'].pct_change(5)

    # 波动率
    df['volatility_5']  = df['return_1d'].rolling(5).std()
    df['volatility_20'] = df['return_1d'].rolling(20).std()

    # 价格位置
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20']  = df['low'].rolling(20).min()
    df['price_position'] = (df['close'] - df['low_20']) / (
        df['high_20'] - df['low_20'] + 1e-9)

    # 目标变量
    df['future_close'] = df['close'].shift(-predict_days)
    df['future_return'] = (df['future_close'] - df['close']) / df['close'] * 100
    df['target'] = (df['future_return'] >= up_threshold).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════
#  模型训练与预测
# ══════════════════════════════════════════════════════════════════

def train_and_predict(df: pd.DataFrame, predict_days: int = 10,
                      up_threshold: float = 2.0) -> dict:
    """
    训练三模型，对最新交易日做预测。
    返回：预测结果字典
    """
    feature_df = df[_FEATURE_COLS + ['target']].copy()
    feature_df = feature_df.dropna(subset=_FEATURE_COLS)
    valid = feature_df['target'].notna()
    X = feature_df[_FEATURE_COLS][valid]
    y = feature_df['target'][valid]

    if len(X) < 100:
        raise ValueError(f'有效数据不足（{len(X)}条），至少需要100条')
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        raise ValueError('正负样本数过少，模型无法训练')

    # 时序分割：保留最后60天测试
    split_idx = max(100, len(X) - 60)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # XGBoost
    try:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
            eval_metric='auc', random_state=42,
        )
        xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
        results['XGBoost'] = {
            'accuracy': accuracy_score(y_test, (xgb_proba >= 0.5).astype(int)),
            'auc':      roc_auc_score(y_test, xgb_proba),
            'prob_up':  float(xgb_clf.predict_proba(X[_FEATURE_COLS].iloc[-1:])[0, 1]),
            'feat_imp': dict(zip(_FEATURE_COLS, xgb_clf.feature_importances_)),
        }
    except Exception as e:
        pass

    # 随机森林
    try:
        rf_clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        rf_clf.fit(X_train, y_train)
        rf_proba = rf_clf.predict_proba(X_test)[:, 1]
        results['RandomForest'] = {
            'accuracy': accuracy_score(y_test, (rf_proba >= 0.5).astype(int)),
            'auc':      roc_auc_score(y_test, rf_proba),
            'prob_up':  float(rf_clf.predict_proba(X[_FEATURE_COLS].iloc[-1:])[0, 1]),
            'feat_imp': dict(zip(_FEATURE_COLS, rf_clf.feature_importances_)),
        }
    except Exception as e:
        pass

    # 逻辑回归
    try:
        lr_clf = LogisticRegression(C=0.1, class_weight='balanced',
                                     max_iter=1000, random_state=42)
        lr_clf.fit(X_train_sc, y_train)
        lr_proba = lr_clf.predict_proba(X_test_sc)[:, 1]
        last_scaled = scaler.transform(X[_FEATURE_COLS].iloc[-1:])
        results['LogisticRegression'] = {
            'accuracy': accuracy_score(y_test, (lr_proba >= 0.5).astype(int)),
            'auc':      roc_auc_score(y_test, lr_proba),
            'prob_up':  float(lr_clf.predict_proba(last_scaled)[0, 1]),
            'feat_imp': None,
        }
    except Exception as e:
        pass

    # 综合
    best_model = max(results.items(), key=lambda x: x[1]['auc'])
    avg_prob = sum(r['prob_up'] for r in results.values()) / len(results)

    return {
        'predictions': {
            name: {
                'prob_up':      round(r['prob_up'] * 100, 1),
                'prob_down':    round((1 - r['prob_up']) * 100, 1),
                'accuracy':     round(r['accuracy'], 3),
                'auc':          round(r['auc'], 3),
            }
            for name, r in results.items()
        },
        'avg_prob_up':    round(avg_prob * 100, 1),
        'best_model':     best_model[0],
        'best_model_auc': round(best_model[1]['auc'], 3),
        'last_close':     float(df['close'].iloc[-1]),
        'last_date':      df['date'].iloc[-1].strftime('%Y-%m-%d'),
        'data_count':     len(df),
        'train_count':    len(X_train),
        'test_count':     len(X_test),
        'positive_ratio': round(y_train.mean() * 100, 1),
        'feat_imp':       best_model[1].get('feat_imp'),
        'feature_df':     df[_FEATURE_COLS + ['target', 'close', 'date']].dropna(subset=_FEATURE_COLS),
    }


# ══════════════════════════════════════════════════════════════════
#  可视化
# ══════════════════════════════════════════════════════════════════

def plot_probability_bar(predictions: dict, avg_prob: float, best_model: str) -> None:
    """绘制概率对比条形图"""
    if not PLT_OK:
        return
    models = list(predictions.keys())
    prob_ups = [predictions[m]['prob_up'] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#e74c3c' if m == best_model else '#3498db' for m in models]
    bars = ax.barh(models, prob_ups, color=colors, height=0.5, edgecolor='white')

    # 添加数值标签
    for bar, val in zip(bars, prob_ups):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val}%', va='center', fontsize=12, fontweight='bold')

    # 参考线
    ax.axvline(x=50, color='#888', linestyle='--', linewidth=1.5, label='50%参考线')
    ax.axvline(x=avg_prob, color='#e67e22', linestyle='-', linewidth=2, label=f'均值 {avg_prob}%')

    ax.set_xlim(0, 100)
    ax.set_xlabel('T+10 上涨(≥2%)概率 (%)', fontsize=11)
    ax.set_title('三模型 T+10 涨跌概率对比', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


def plot_feature_importance(feat_imp: dict) -> None:
    """绘制特征重要性"""
    if not PLT_OK or not feat_imp:
        return
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    names  = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], values[::-1], color='#2c5aa0', height=0.5)
    ax.set_xlabel('重要性', fontsize=11)
    ax.set_title('Top 15 特征重要性（随机森林）', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def plot_kline(df: pd.DataFrame, name: str) -> None:
    """绘制K线图"""
    if not PLT_OK or df is None or len(df) < 20:
        return
    # 截取最近120天
    plot_df = df.tail(120).copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # 涨跌颜色
    colors = ['#e74c3c' if plot_df['close'].iloc[i] >= plot_df['close'].iloc[i-1] else '#27ae60'
              for i in range(1, len(plot_df))]
    colors = ['#e74c3c'] + colors  # 第一根用红色

    ax1.plot(plot_df['date'], plot_df['close'], color='#1a5490', linewidth=1.5, label='收盘价')
    ax1.fill_between(plot_df['date'], plot_df['close'], alpha=0.1, color='#1a5490')

    # MA5 / MA20
    ma5  = plot_df['close'].rolling(5).mean()
    ma20 = plot_df['close'].rolling(20).mean()
    ax1.plot(plot_df['date'], ma5,  color='#e67e22', linewidth=1, label='MA5',  alpha=0.8)
    ax1.plot(plot_df['date'], ma20, color='#9b59b6', linewidth=1, label='MA20', alpha=0.8)
    ax1.set_ylabel('价格', fontsize=11)
    ax1.set_title(f'{name} 近期走势（近120交易日）', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # 成交量
    ax2.bar(plot_df['date'], plot_df['volume'], color='#3498db', alpha=0.6, width=1)
    ax2.set_ylabel('成交量', fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  股票名称映射（常用A股）
# ══════════════════════════════════════════════════════════════════

_COMMON_STOCKS = {
    'sz002837': '英维克',    'sh600519': '贵州茅台',
    'sz000001': '平安银行',  'sh600036': '招商银行',
    'sh601318': '中国平安', 'sz300750': '宁德时代',
    'sh688256': '寒武纪',   'sh688041': '海光信息',
    'sh688981': '中芯国际', 'sz300474': '景嘉微',
    'sz002747': '埃斯顿',   'sh688169': '石头科技',
    'sh600547': '山东黄金', 'sh601899': '紫金矿业',
    'sz002466': '天齐锂业', 'sz002460': '赣锋锂业',
    'sz300124': '汇川技术', 'sh601138': '工业富联',
    'sh603160': '汇顶科技', 'sz002371': '北方华创',
    'sz002049': '紫光国微', 'sh603501': '韦尔股份',
}


# ══════════════════════════════════════════════════════════════════
#  Streamlit 页面布局
# ══════════════════════════════════════════════════════════════════

def main():
    # ── 标题 & 说明 ─────────────────────────────────────────────
    st.title("📈 股票趋势预测器")
    st.markdown(
        "基于 **XGBoost + 随机森林 + 逻辑回归** 三模型对比，"
        "预测 T+N 日涨跌概率（≥2% 判定为涨）。"
        "数据来源：腾讯财经 → 新浪财经（自动切换）。"
    )
    st.divider()

    # ── 侧边栏参数 ─────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 预测参数")
        predict_days   = st.slider("预测周期（T+N交易日）",  5, 20, 10, step=1,
                                    help="预测未来多少个交易日的走势")
        up_threshold   = st.slider("上涨判定阈值（%）",     1.0, 5.0, 2.0, step=0.5,
                                    help="T+N日后涨幅达到此阈值才算'涨'")
        lookback_days  = st.slider("历史数据范围（天）",    200, 600, 500, step=50,
                                    help="用于训练的历史K线天数（约2年=500天）")
        st.caption("参数修改后重新点击查询生效")
        st.divider()
        st.markdown("**📌 常用股票代码**")
        for code, name in list(_COMMON_STOCKS.items())[:8]:
            st.caption(f"`{code}` — {name}")

    # ── 输入框 ──────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        raw_input = st.text_input(
            "股票代码",
            placeholder="输入代码，如 sz002837（英维克）或 sh600519（茅台）",
            help="沪深股票代码格式：sh=上证 / sz=深证，如 sz002837",
            label_visibility="collapsed",
        )
    with col2:
        query_btn = st.button("🔍 查询", use_container_width=True, type="primary")

    # ── 默认展示（无输入时）────────────────────────────────────
    if not raw_input and not query_btn:
        st.info("👆 请输入股票代码后点击「查询」")
        st.stop()

    # ── 解析股票代码 ───────────────────────────────────────────
    user_input = raw_input.strip().lower()

    # 自动补充前缀
    if not user_input.startswith(('sh', 'sz', 'bj')):
        # 纯数字，尝试推断（6开头→sh，0/2/3开头→sz）
        if user_input.startswith('6'):
            user_input = 'sh' + user_input
        elif user_input.startswith(('0', '2', '3')):
            user_input = 'sz' + user_input
        else:
            user_input = 'sz' + user_input

    stock_name = _COMMON_STOCKS.get(user_input, user_input)

    # ── 依赖检查 ───────────────────────────────────────────────
    if not _deps_ok:
        st.error(f"⚠️ 缺少机器学习依赖库: `{_model_import_error}`")
        st.info("本地运行请执行: `pip install xgboost scikit-learn matplotlib`")
        st.stop()

    # ── 数据加载 & 训练 & 预测（带进度条）─────────────────────
    with st.spinner(f"正在获取 {stock_name}（{user_input}）历史数据..."):
        try:
            df = get_daily_kline(user_input, count=lookback_days)
        except Exception as e:
            st.error(f"❌ 数据获取失败: {e}")
            return

    with st.spinner("正在构建技术指标特征..."):
        df_feat = build_features(df, up_threshold=up_threshold, predict_days=predict_days)

    with st.spinner("正在训练 XGBoost / 随机森林 / 逻辑回归 三模型..."):
        try:
            result = train_and_predict(df_feat, predict_days=predict_days,
                                       up_threshold=up_threshold)
        except ValueError as e:
            st.error(f"❌ 训练失败: {e}")
            return
        except Exception as e:
            st.error(f"❌ 预测异常: {e}")
            return

    # ══════════════════════════════════════════════════════════════
    #  结果展示
    # ══════════════════════════════════════════════════════════════

    # 基本信息
    col_meta = st.columns(4)
    col_meta[0].metric("股票名称", stock_name)
    col_meta[1].metric("最新收盘", f"¥{result['last_close']:.2f}")
    col_meta[2].metric("数据截止", result['last_date'])
    col_meta[3].metric("数据条数", f"{result['data_count']} 条")
    st.divider()

    # ── 核心结论 ───────────────────────────────────────────────
    avg = result['avg_prob_up']
    if avg >= 60:
        verdict_icon = "🔴"
        verdict_text = "偏涨"
        verdict_color = "off"
    elif avg >= 40:
        verdict_icon = "🟡"
        verdict_text = "中性"
        verdict_color = "off"
    else:
        verdict_icon = "🟢"
        verdict_text = "偏跌"
        verdict_color = "off"

    st.markdown(
        f"## {verdict_icon} T+{predict_days} 综合预测结论\n"
        f"**{stock_name}** 未来 **{predict_days}** 个交易日上涨（≥{up_threshold}%）概率："
        f" ### {avg}%"
    )

    verdict_col, detail_col = st.columns([1, 2])

    with verdict_col:
        st.metric("综合涨跌判断", verdict_text,
                  delta=f"上涨概率 {avg}%")

    with detail_col:
        st.markdown(f"""
        **推荐参考模型**：{result['best_model']}（AUC={result['best_model_auc']}）
        - 训练样本：{result['train_count']} 条 | 测试样本：{result['test_count']} 条
        - 训练集正例（涨）占比：{result['positive_ratio']}%
        """)

    st.divider()

    # ── 模型对比 ───────────────────────────────────────────────
    st.subheader("📊 三模型 T+10 涨跌概率对比")

    # 表格
    pred_table = []
    for name, pred in result['predictions'].items():
        star = " ★" if name == result['best_model'] else ""
        pred_table.append({
            "模型": name + star,
            "T+上涨概率": f"{pred['prob_up']}%",
            "T+不涨概率": f"{pred['prob_down']}%",
            "测试准确率": f"{pred['accuracy']:.1%}",
            "AUC": f"{pred['auc']:.3f}",
        })
    st.dataframe(pd.DataFrame(pred_table), use_container_width=True, hide_index=True)

    # 条形图
    plot_probability_bar(result['predictions'], avg, result['best_model'])

    st.divider()

    # ── K线图 ─────────────────────────────────────────────────
    col_k, col_imp = st.columns(2)
    with col_k:
        st.subheader("📉 近期走势（近120交易日）")
        plot_kline(df, stock_name)

    with col_imp:
        st.subheader("🏆 Top 15 特征重要性")
        if result['feat_imp']:
            plot_feature_importance(result['feat_imp'])
        else:
            st.info("该模型无特征重要性输出")

    st.divider()

    # ── 风险提示 ──────────────────────────────────────────────
    st.warning(
        "⚠️ **风险提示**：本工具基于历史数据机器学习，模型 AUC 在 0.5~0.7 之间，"
        "预测能力有限。A股市场受政策、情绪、资金等多因素影响，历史规律未必延续。"
        "**本预测结果仅供参考，不构成任何投资建议。** 股市有风险，投资需谨慎！"
    )

    # ── 页脚 ──────────────────────────────────────────────────
    st.caption(
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | "
        f"数据: 腾讯财经 → 新浪财经 | 模型: XGBoost + RandomForest + LogisticRegression"
    )


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()

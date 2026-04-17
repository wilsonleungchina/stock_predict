# 股票趋势预测 Web 应用

基于 **Streamlit** 的 A 股走势预测工具，输入股票代码即可查看 T+N 日涨跌概率。

## 功能

- 输入任意沪深股票代码，查询未来走势预测
- 三模型对比：XGBoost、随机森林、逻辑回归
- K 线走势图 + 特征重要性图
- 可自定义预测周期（5~20日）和上涨阈值（1%~5%）

## 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

## 部署到 Streamlit Cloud（免费）

### 前提条件
- GitHub 账号
- 代码上传到公开仓库（Private 需付费）

### 步骤

#### 1. 把代码上传到 GitHub

在 GitHub 新建仓库（如 `stock-predictor`），把以下文件上传：

```
stock-predictor/
├── app.py              # Streamlit 主程序
├── stock_predictor.py   # 预测逻辑（app.py 会调用）
├── requirements.txt    # 依赖列表
└── README.md           # 本文件
```

#### 2. 连接 Streamlit Cloud

1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 用 GitHub 登录
3. 点击 **New app**
4. 选择：
   - **Repository**：你的仓库名（如 `yourname/stock-predictor`）
   - **Branch**：`main`
   - **Main file path**：`app.py`
5. 点击 **Deploy!**

#### 3. 完成

约 2~3 分钟后，应用上线：
```
https://yourname-stock-predictor.streamlit.app
```

每次 `git push` 到 main 分支，**自动重新部署**。

## 股票代码参考

| 代码 | 名称 |
|------|------|
| sz002837 | 英维克 |
| sh600519 | 贵州茅台 |
| sz300750 | 宁德时代 |
| sh688256 | 寒武纪 |
| sz300124 | 汇川技术 |
| sh688981 | 中芯国际 |
| sh601899 | 紫金矿业 |
| sz002466 | 天齐锂业 |

## 数据来源

- 腾讯财经日K（前复权）→ 首选
- 新浪财经日K → 备用（自动切换）

## 免责声明

本工具基于历史数据的机器学习预测，模型 AUC 通常在 0.5~0.7，预测能力有限。**本预测结果仅供参考，不构成任何投资建议**。股市有风险，投资需谨慎。

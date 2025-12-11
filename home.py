import streamlit as st

st.set_page_config(
    page_title="海洋工程波浪仿真综合平台",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 海洋工程波浪仿真综合平台")

st.markdown("""
### 欢迎使用
本平台集成了波浪生成、CFD 验证及后处理分析的全流程工具。请在 **左侧侧边栏** 选择您需要的功能模块。

---

### 📂 模块说明

#### 1. 🌊 **波浪生成 (Wave Generator)**
   - 基于 **JONSWAP 谱** 生成不规则波浪参数。
   - 支持生成 CFD 软件（STAR-CCM+, Fluent）所需的 **波面公式** 及 **速度场公式** (Sinh/Exp)。
   - 支持自动参数优化与 CSV 导出。

#### 2. ⚖️ **造波验证 (Verification)**
   - 用于对比 **理论波** 与 **CFD 仿真波**。
   - 内置 **Detrend (去趋势)**、**Zero-mean (去均值)** 算法。
   - 自动搜索最佳匹配时间窗口，计算 RMSE 及误差百分比。

#### 3. 📊 **参数分析 (Analysis)**
   - 针对单个波浪时间序列进行 **上跨零点法 (Zero-up Crossing)** 分析。
   - 自动统计 **H1/3 (三一波高)**、**Tz (平均周期)** 等关键指标。
   - 提供交互式时域图与波高分布直方图。

---
*Created with Streamlit & Python*
""")

# 隐藏主页面的侧边栏提示，只保留导航
st.sidebar.success("👆 请在上方选择功能模块")
import streamlit as st
from PIL import Image

# 1. 加载图片 (假设 1.jpg 在同级目录下)
# 建议将图片文件名改为更通用的名字，如 'logo.jpg'，这里先用 '1.jpg'
logo = Image.open("1.jpg")

st.set_page_config(
    page_title="海洋工程波浪仿真综合平台",
    page_icon=logo,  # 2. 这里直接传入图片对象，浏览器标签页图标就会变成你的 Logo
    layout="wide"
)

# 3. 使用列布局将 Logo 和 标题并排显示
# [1, 5] 是列宽比例，你可以根据 Logo 的实际显示效果调整这个比例
col1, col2 = st.columns([1, 7])

with col1:
    # 显示 Logo，width 控制大小
    st.image(logo, use_column_width=True)

with col2:
    # 垂直居中稍微有点难，Streamlit默认顶部对齐，
    # 我们可以直接显示标题，或者加一点 markdown 的空行让它视觉居中
    st.title("海洋工程波浪仿真综合平台1")
    st.caption("Marine Hydrodynamic Performance & Optimization Lab") # 既然Logo上有英文，这里配个英文副标题会更有质感

st.markdown("---") # 加个分割线区分头部和正文

st.markdown("""
### 欢迎使用
本平台集成了波浪生成、CFD 验证及后处理分析的全流程工具。请在 **左侧侧边栏** 选择您需要的功能模块。

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
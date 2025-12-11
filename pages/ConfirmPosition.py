import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# --- 页面配置 ---
st.set_page_config(page_title="CFD 规则波生成与验证工具", layout="wide", page_icon="🌊")


# ===========================
# 1. 核心算法库
# ===========================

def generate_theoretical_wave(t_series, param_df, x_pos):
    """
    根据参数文件和位置，在给定的时间序列上生成理论波
    公式: eta = A * cos(k*x - omega*t + phase)
    """
    eta = np.zeros_like(t_series, dtype=np.float64)

    # 支持列名容错
    cols = param_df.columns.str.lower()

    # 映射列名 (假设用户上传的CSV包含这些信息的变体)
    try:
        # 寻找对应的列名
        col_amp = param_df.columns[cols.str.contains('amp')][0]  # Amplitude
        col_omega = param_df.columns[cols.str.contains('freq') | cols.str.contains('omega')][0]  # AngularFrequency
        col_k = param_df.columns[cols.str.contains('wave') | cols.str.contains('k')][0]  # Wavenumber
        col_phase = param_df.columns[cols.str.contains('phase')][0]  # Phase

        for _, row in param_df.iterrows():
            A = row[col_amp]
            omega = row[col_omega]
            k = row[col_k]
            phi = row[col_phase]

            # 叠加分量
            eta += A * np.cos(k * x_pos - omega * t_series + phi)

        return eta, None
    except IndexError:
        return None, "参数文件列名识别失败。请确保CSV包含：Amplitude, AngularFrequency, Wavenumber, Phase"


def process_simulation_data(y_sim, do_detrend, do_zeromean):
    """
    处理仿真数据：去趋势、去均值
    """
    y_proc = y_sim.copy()

    if do_detrend:
        y_proc = signal.detrend(y_proc, type='linear')

    if do_zeromean:
        y_proc = y_proc - np.mean(y_proc)

    return y_proc


def read_file(uploaded_file):
    """通用的文件读取函数"""
    if uploaded_file is None: return None
    try:
        if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
            try:
                # 尝试读取带表头的
                df = pd.read_csv(uploaded_file)
                # 简单的检查，如果第一列不是数字，可能需要重新读取
                pd.to_numeric(df.iloc[:, 0])
            except:
                # 假如没有表头
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        return None


# ===========================
# 2. 前端界面构建
# ===========================

st.title("🌊 CFD 规则波生成与验证工具")
st.markdown("""
**模式说明**：上传 **仿真时间序列** 和 **波浪参数**，程序将自动生成理论波形并进行对比。
适用于规则波造波精度验证、相位校准及衰减分析。
""")

# --- 侧边栏：输入区域 ---
with st.sidebar:
    st.header("1. 数据输入")

    # 1.1 仿真数据
    st.subheader("A. 仿真数据 (CFD)")
    f_sim = st.file_uploader("上传水位监测数据 (.csv/.xlsx)", type=['csv', 'xlsx'])
    if f_sim:
        st.info("读取中...")
        df_sim_raw = read_file(f_sim)
        if df_sim_raw is not None:
            # 假设前两列是 Time 和 Elevation
            df_sim = df_sim_raw.iloc[:, :2].copy()
            df_sim.columns = ['Time', 'Elevation']
            # 确保是数值并排序
            df_sim = df_sim.apply(pd.to_numeric, errors='coerce').dropna().sort_values('Time')
            st.success(f"已加载: {len(df_sim)} 个时间点")
        else:
            st.error("文件格式错误")

    # 1.2 参数数据
    st.subheader("B. 波浪参数 (Theory)")
    f_param = st.file_uploader("上传波浪参数文件 (.csv)", type=['csv'])
    df_param = None
    if f_param:
        df_param = pd.read_csv(f_param)
        st.success(f"已加载 {len(df_param)} 组波浪分量")

    st.divider()

    # 2. 空间位置设置
    st.header("2. 探针位置设置")
    # 按照你的要求：位置步长为 1m
    x_probe = st.number_input("监测点 X 坐标 (m)", value=0.0, step=1.0, format="%.1f",
                              help="设置生成理论波的空间位置，步长为1m")

    st.divider()

    # 3. 修正选项
    st.header("3. 仿真数据修正")
    do_detrend = st.checkbox("去除线性趋势 (Detrend)", value=True, help="去除仿真数据的整体漂移")
    do_zeromean = st.checkbox("去除平均值 (Zero-mean)", value=True, help="强制将仿真数据静水面归零")

# --- 主界面：逻辑处理与展示 ---

if f_sim and f_param and df_sim is not None and df_param is not None:

    # 1. 获取时间轴 (完全依照仿真数据)
    t_sim = df_sim['Time'].values
    y_sim_raw = df_sim['Elevation'].values

    # 2. 生成理论波数据
    y_theo, err_msg = generate_theoretical_wave(t_sim, df_param, x_probe)

    if err_msg:
        st.error(err_msg)
        st.stop()

    # 3. 处理仿真数据 (清洗)
    y_sim_clean = process_simulation_data(y_sim_raw, do_detrend, do_zeromean)

    # 4. 计算误差指标
    # 截取中间段计算 RMSE (去除两端可能的不稳定)
    cut_ratio = 0.1
    n_points = len(t_sim)
    idx_start = int(n_points * cut_ratio)
    idx_end = int(n_points * (1 - cut_ratio))

    if idx_end > idx_start:
        rmse = np.sqrt(mean_squared_error(y_theo[idx_start:idx_end], y_sim_clean[idx_start:idx_end]))
        r2 = r2_score(y_theo[idx_start:idx_end], y_sim_clean[idx_start:idx_end])
    else:
        rmse = 0
        r2 = 0

    # --- 结果展示区 ---

    # A. 顶部指标
    col1, col2, col3 = st.columns(3)
    col1.metric("当前 X 位置", f"{x_probe:.1f} m")
    col2.metric("RMSE (均方根误差)", f"{rmse:.4f} m", help="数值越小越好")
    col3.metric("R² (拟合优度)", f"{r2:.4f}", help="越接近 1 越好")

    # B. 绘图
    st.subheader("📈 波形对比分析")

    fig = go.Figure()

    # 理论波 (实线)
    fig.add_trace(go.Scatter(
        x=t_sim, y=y_theo,
        name='理论值 (Theory)',
        line=dict(color='#ff7f0e', width=2.5)
    ))

    # 仿真波 (清洗后)
    fig.add_trace(go.Scatter(
        x=t_sim, y=y_sim_clean,
        name='仿真值 (CFD Clean)',
        line=dict(color='#2ca02c', width=2)
    ))

    # 仿真波 (原始 - 可选)
    fig.add_trace(go.Scatter(
        x=t_sim, y=y_sim_raw,
        name='仿真原始值 (CFD Raw)',
        line=dict(color='gray', width=1, dash='dot'),
        visible='legendonly'
    ))

    fig.update_layout(
        title=f'Wave Elevation Comparison at x = {x_probe} m',
        xaxis_title='Time (s)',
        yaxis_title='Elevation (m)',
        template="plotly_white",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # C. 数据导出
    st.subheader("💾 数据导出")

    df_export = pd.DataFrame({
        'Time': t_sim,
        'Theory_Elevation': y_theo,
        'CFD_Clean': y_sim_clean,
        'CFD_Raw': y_sim_raw,
        'Error': y_sim_clean - y_theo
    })

    csv_data = df_export.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="下载对比数据 (.csv)",
        data=csv_data,
        file_name=f"wave_validation_x{x_probe}_m.csv",
        mime="text/csv",
        type="primary"
    )

    st.info("提示：通过侧边栏调整 '监测点 X 坐标'，图表和误差计算会实时更新。")

else:
    # 欢迎页/空状态
    st.info("👈 请在左侧上传 [仿真数据文件] 和 [波浪参数文件] 以开始。")

    with st.expander("查看波浪参数文件 (.csv) 格式示例"):
        st.markdown("""
        CSV 文件应包含定义规则波（或不规则波分量）的列。程序会自动识别以下关键字：
        * **Amplitude** (振幅)
        * **AngularFrequency** (角频率 rad/s) 或 Omega
        * **Wavenumber** (波数 k)
        * **Phase** (相位 rad)

        | amplitude | angularFrequency | wavenumber | phase |
        | :--- | :--- | :--- | :--- |
        | 0.5 | 1.25 | 0.8 | 0.0 |
        """)
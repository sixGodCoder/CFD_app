import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import signal, interpolate
from scipy.signal import savgol_filter

# --- 页面配置 ---
st.set_page_config(page_title="Wave Analysis & Validation Tool", layout="wide")

st.title("Wave Analysis & Validation Tool ")
st.markdown("""
本工具整合了 **时域统计分析 (上跨零点法)** 与 **频域谱分析**。
""")


# --- 1. 核心算法函数 ---

def analyze_waves_zero_crossing(time, elevation):
    """
    移植自 H31andT0.py: 使用上跨零点法 (Zero Up-crossing) 计算统计特征
    """
    # 1. 去除平均值 (零均值化)
    elevation_zero_mean = elevation - np.mean(elevation)

    # 2. 寻找上跨零点 (sign从负变正的位置)
    # np.diff(np.sign) == 2 代表从 -1 变到 1
    crossings = np.where(np.diff(np.sign(elevation_zero_mean)) == 2)[0]

    waves_height = []
    waves_period = []

    # 3. 遍历提取波高和周期
    for i in range(len(crossings) - 1):
        idx_start = crossings[i]
        idx_end = crossings[i + 1]

        # 截取一个完整的波
        wave_segment = elevation_zero_mean[idx_start:idx_end]

        if len(wave_segment) > 0:
            # 波高 = 波峰 - 波谷
            h = np.max(wave_segment) - np.min(wave_segment)
            waves_height.append(h)

        # 周期 = 下一个零点时间 - 当前零点时间
        t = time[idx_end] - time[idx_start]
        waves_period.append(t)

    waves_height = np.array(waves_height)
    waves_period = np.array(waves_period)

    # 计算统计指标
    if len(waves_height) > 0:
        sorted_H = np.sort(waves_height)[::-1]
        n_third = int(len(sorted_H) / 3)
        # H1/3: 前1/3大波高的平均值
        h_1_3 = np.mean(sorted_H[:n_third]) if n_third > 0 else np.mean(sorted_H)
        t_z = np.mean(waves_period)
        h_max = np.max(waves_height)
    else:
        h_1_3, t_z, h_max = 0, 0, 0

    return h_1_3, t_z, h_max, len(waves_height)


def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file)
                    pd.to_numeric(df.iloc[:, 0])  # 尝试第一列转数值
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=None)  # 无表头回退
            else:
                df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"读取失败: {e}")
    return None


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


# --- 2. 界面与主逻辑 ---

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. CFD 仿真数据")
    cfd_file = st.file_uploader("上传 CFD 数据", type=['csv', 'xlsx', 'txt'], key="cfd")
with col2:
    st.subheader("2. 理论/目标数据")
    theory_file = st.file_uploader("上传 理论数据", type=['csv', 'xlsx', 'txt'], key="theory")

df_cfd = load_data(cfd_file)
df_theory = load_data(theory_file)

if df_cfd is not None and df_theory is not None:
    # --- 列映射与预处理 ---
    with st.expander("数据列映射与时间截断", expanded=True):
        c_cols = df_cfd.columns.tolist()
        t_cols = df_theory.columns.tolist()

        c1, c2, c3 = st.columns(3)
        with c1:
            c_t = st.selectbox("CFD 时间列", c_cols, index=0)
            c_z = st.selectbox("CFD 波高列", c_cols, index=1 if len(c_cols) > 1 else 0)
        with c2:
            t_t = st.selectbox("理论 时间列", t_cols, index=0)
            t_z = st.selectbox("理论 波高列", t_cols, index=1 if len(t_cols) > 1 else 0)
        with c3:
            start_time = st.number_input("截断起始时间 (s)", value=20.0, help="去除初始不稳定波")


    # 数据清洗
    def clean_series(df, col_t, col_z, t_start):
        df = df.sort_values(by=col_t)
        # 转数值防报错
        df[col_t] = pd.to_numeric(df[col_t], errors='coerce')
        df[col_z] = pd.to_numeric(df[col_z], errors='coerce')
        df.dropna(subset=[col_t, col_z], inplace=True)

        mask = df[col_t] >= t_start
        return df.loc[mask, col_t].values, df.loc[mask, col_z].values


    t_cfd, z_cfd = clean_series(df_cfd, c_t, c_z, start_time)
    t_theo, z_theo = clean_series(df_theory, t_t, t_z, start_time)

    # --- 3. 计算 H1/3 (使用上跨零点法) ---
    h13_cfd, tz_cfd, hmax_cfd, num_cfd = analyze_waves_zero_crossing(t_cfd, z_cfd)
    h13_theo, tz_theo, hmax_theo, num_theo = analyze_waves_zero_crossing(t_theo, z_theo)

    # 显示统计结果
    st.markdown("### 📊 统计分析结果 (时域上跨零点法)")
    met1, met2, met3, met4 = st.columns(4)

    err_h13 = (h13_cfd - h13_theo) / h13_theo * 100 if h13_theo != 0 else 0

    met1.metric("CFD H1/3", f"{h13_cfd:.4f} m", help="基于上跨零点统计")
    met2.metric("理论 H1/3", f"{h13_theo:.4f} m", help="基于上跨零点统计")
    met3.metric("误差 (H1/3)", f"{err_h13:.2f} %", delta_color="inverse")
    met4.metric("识别波数量 (CFD)", f"{num_cfd} 个")

    # --- 4. 频域分析与绘图 (Figure 6 复现) ---
    st.markdown("### 📈 频域谱分析 ")

    # 4.1 频谱计算
    # 为保证对比准确，将理论数据插值对齐到CFD采样率(仅用于频谱对比，不影响H1/3计算)
    # 计算 CFD 采样率
    dt_cfd = np.mean(np.diff(t_cfd))
    fs_cfd = 1 / dt_cfd

    # 对齐理论数据用于绘图对比 (可选，也可分别计算)
    # 这里分别计算更科学，不改变原始数据特性
    dt_theo = np.mean(np.diff(t_theo)) 
    fs_theo = 1 / dt_theo

    # Welch 参数：窗口越大分辨率越高(尖峰越准)，但噪点越多
    nperseg_cfd = len(z_cfd) // 2
    nperseg_theo = len(z_theo) // 2

    freq_c, psd_c_hz = signal.welch(z_cfd, fs=fs_cfd, nperseg=nperseg_cfd, scaling='density')
    freq_t, psd_t_hz = signal.welch(z_theo, fs=fs_theo, nperseg=nperseg_theo, scaling='density')

    # 4.2 单位转换
    # f (Hz) -> omega (rad/s)
    w_c = 2 * np.pi * freq_c
    w_t = 2 * np.pi * freq_t

    # S(f) -> S(omega)
    S_w_c = psd_c_hz / (2 * np.pi)
    S_w_t = psd_t_hz / (2 * np.pi)

    # 缩放至 10^4
    S_plot_c = S_w_c * 10000
    S_plot_t = S_w_t * 10000

    # 4.3 平滑处理 (Savitzky-Golay)
    # 侧边栏控制
    st.sidebar.markdown("---")
    st.sidebar.header("平滑参数")
    win_len = st.sidebar.slider("平滑窗口长度 (奇数)", 5, 99, 15, step=2)
    poly_order = 3

    try:
        S_smooth_c = savgol_filter(S_plot_c, win_len, poly_order)
        S_smooth_t = savgol_filter(S_plot_t, win_len, poly_order)
        # 去除负值
        S_smooth_c = np.maximum(S_smooth_c, 0)
        S_smooth_t = np.maximum(S_smooth_t, 0)
    except:
        S_smooth_c = S_plot_c
        S_smooth_t = S_plot_t

    # 4.4 修复后的 LaTeX 文本显示 (避免 f-string 报错)
    st.markdown(r"""
    **图表说明：**
    * **Y轴单位**：$S(\omega) \times 10^4 \ (m^2s)$，与文献 Figure 6 保持一致。
    * **X轴**：角频率 $\omega \ (rad/s)$。
    """)

    # 4.5 绘图
    fig = go.Figure()

    # 理论值 (红虚线)
    fig.add_trace(go.Scatter(
        x=w_t, y=S_smooth_t,
        mode='lines', name='Theory Spectrum',
        line=dict(color='red', width=2, dash='dash', shape='spline')
    ))

    # CFD值 (蓝实线)
    fig.add_trace(go.Scatter(
        x=w_c, y=S_smooth_c,
        mode='lines', name='CFD Spectrum',
        line=dict(color='blue', width=2, shape='spline')
    ))

    fig.update_layout(
        title="Wave Spectrum Comparison (Smoothed)",
        xaxis_title="Angular Frequency ω (rad/s)",
        yaxis_title="S (m²s) × 10⁴",
        template="plotly_white",
        xaxis=dict(range=[0, 15], showgrid=True),
        yaxis=dict(showgrid=True, rangemode="tozero"),
        hovermode="x unified",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- 5. 数据导出 ---
    st.subheader("📥 数据下载")

    # 为了导出对齐的数据，我们需要创建一个公共的 omega 轴
    # 简单起见，我们截取最短长度并导出（或者也可以插值对齐）
    min_len = min(len(w_c), len(w_t))

    df_download = pd.DataFrame({
        'Omega (rad/s)': w_c[:min_len],
        'CFD_S_x10e4 (Smoothed)': S_smooth_c[:min_len],
        'Theory_S_x10e4 (Smoothed)': S_smooth_t[:min_len]  # 注意：这里频率轴略有错位，仅供绘图参考
    })

    csv_data = convert_df(df_download)

    st.download_button(
        label="下载波谱数据 (CSV)",
        data=csv_data,
        file_name="fig6_spectrum_data.csv",
        mime="text/csv"
    )

else:
    st.info("👋 请在上方上传数据文件以开始分析。")
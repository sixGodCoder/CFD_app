import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import re

st.set_page_config(page_title="波浪FFT分析与误差校核", layout="wide")


def detect_columns(df):
    """智能识别时间列和波高列"""
    time_col, wave_col = None, None
    columns = df.columns.tolist()
    time_pattern = re.compile(r'(time|date|sec|t\b|时间|日期|秒)', re.IGNORECASE)
    wave_pattern = re.compile(r'(wave|height|amp|elev|eta|z\b|h\b|波高|水位|幅值|高程)', re.IGNORECASE)

    for col in columns:
        if time_pattern.search(str(col)) and not time_col:
            time_col = col
        elif wave_pattern.search(str(col)) and not wave_col:
            wave_col = col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not time_col and numeric_cols:
        for col in numeric_cols:
            diffs = df[col].dropna().diff().dropna()
            if len(diffs) > 0 and (diffs >= 0).all():
                time_col = col
                break
    if not wave_col and numeric_cols:
        candidates = [c for c in numeric_cols if c != time_col]
        if candidates: wave_col = df[candidates].var().idxmax()

    if not time_col and len(columns) > 0: time_col = columns[0]
    if not wave_col and len(columns) > 1: wave_col = columns[1] if columns[1] != time_col else columns[0]
    return time_col, wave_col


def compute_fft(t, y, fs):
    """执行 FFT 计算"""
    L = len(t)
    nfft = int(2 ** np.ceil(np.log2(L * 10)))
    Y = np.fft.fft(y - np.mean(y), n=nfft) / L  # 减去均值消除直流分量
    f = (fs / 2) * np.linspace(0, 1, nfft // 2 + 1)
    amp = 2 * np.abs(Y[0:nfft // 2 + 1])
    return f, amp


# ================= UI 侧边栏 =================
st.sidebar.title("🛠️ 分析参数设置")
theory_period = st.sidebar.number_input("输入理论周期 (s)", min_value=0.0, value=0.0, step=0.1,
                                        help="输入预期的波浪周期以计算误差")

# ================= UI 主界面 =================
st.title("🌊 波浪 FFT 分析与周期误差校核")

uploaded_file = st.file_uploader("上传数据文件 (*.csv, *.xlsx)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        guess_time_col, guess_wave_col = detect_columns(df)

        st.subheader("1. 数据配置")
        c1, c2 = st.columns(2)
        time_col = c1.selectbox("时间列", df.columns, index=df.columns.tolist().index(guess_time_col))
        wave_col = c2.selectbox("波高列", df.columns, index=df.columns.tolist().index(guess_wave_col))

        t_data, y_data = df[time_col].values, df[wave_col].values
        dt_array = np.diff(t_data)
        fs = 1.0 / np.median(dt_array[dt_array > 0])

        # 分析时间段
        t_min, t_max = float(t_data.min()), float(t_data.max())
        time_range = st.slider("分析时间范围 (s)", t_min, t_max, (t_min, t_max))
        mask = (t_data >= time_range[0]) & (t_data <= time_range[1])
        t_selected, y_selected = t_data[mask], y_data[mask]

        # FFT 计算
        f, amp = compute_fft(t_selected, y_selected, fs)
        max_idx = np.argmax(amp)
        main_freq = f[max_idx]
        main_period = 1.0 / main_freq if main_freq > 0 else 0
        sig_h = 4 * np.std(y_selected, ddof=1)

        # 2. 误差计算逻辑
        st.subheader("2. 分析结果与周期校核")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        res_col1.metric("实测主周期 (Tp)", f"{main_period:.3f} s")
        res_col2.metric("显著波高 (Hs)", f"{sig_h:.3f} m")

        if theory_period > 0:
            abs_error = main_period - theory_period
            rel_error = (abs_error / theory_period) * 100
            # 使用 delta 颜色表示误差方向
            res_col3.metric("理论周期 (Tt)", f"{theory_period:.3f} s")
            res_col4.metric("相对误差 (%)", f"{rel_error:.2f}%", delta=f"{abs_error:.3f} s", delta_color="inverse")
        else:
            res_col3.info("👈 在侧边栏输入理论周期以启用误差校核")

        # 3. 可视化
        # 频谱图增加理论线
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=f, y=amp, name='频谱幅值', line=dict(color='orange')))

        # 标注实测峰值点
        fig_freq.add_annotation(x=main_freq, y=amp[max_idx], text="实测峰值", showarrow=True, arrowhead=1)

        # 如果有理论值，画一条垂线对比
        if theory_period > 0:
            theory_freq = 1.0 / theory_period
            fig_freq.add_vline(x=theory_freq, line_dash="dash", line_color="red", annotation_text="理论频率")

        fig_freq.update_layout(title="波浪频谱分析 (对比图)", xaxis_title="频率 (Hz)", yaxis_title="幅值", height=450)
        st.plotly_chart(fig_freq, use_container_width=True)

        # 时域图
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t_selected, y=y_selected, name='波高', line=dict(color='#1f77b4')))
        fig_time.update_layout(title="波浪时域图", xaxis_title="时间 (s)", yaxis_title="波高 (m)", height=350)
        st.plotly_chart(fig_time, use_container_width=True)

        # 4. 导出
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_summary = pd.DataFrame({
                '参数': ['理论周期(s)', '实测周期(s)', '绝对误差(s)', '相对误差(%)', '显著波高(m)'],
                '数值': [theory_period, main_period,
                         main_period - theory_period if theory_period > 0 else 0,
                         ((main_period - theory_period) / theory_period * 100) if theory_period > 0 else 0,
                         sig_h]
            })
            df_summary.to_excel(writer, sheet_name='误差分析报告', index=False)
            pd.DataFrame({'Freq_Hz': f, 'Amp': amp}).to_excel(writer, sheet_name='频谱数据', index=False)

        st.download_button("📥 下载完整报告", output.getvalue(), "Wave_Report.xlsx", "application/vnd.ms-excel")

    except Exception as e:
        st.error(f"处理出错: {e}")
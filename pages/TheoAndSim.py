import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(page_title="CFD造波验证工作台 (All-in-One)", layout="wide")


# ===========================
# 1. 核心算法库
# ===========================

def zero_crossing_analysis(time, elevation):
    """
    使用上跨零点法计算 H1/3 和 Tz
    输入: 时间序列, 水位序列 (建议已去均值)
    输出: H1/3, Tz, 最大波高, 识别到的波数量
    """
    # 确保数据去均值 (Zero-mean)
    elev_zero_mean = elevation - np.mean(elevation)

    # 寻找上跨零点 indices
    # sign 变为 +1 的瞬间
    crossings = np.where(np.diff(np.sign(elev_zero_mean)) == 2)[0]

    waves_height = []
    waves_period = []

    for i in range(len(crossings) - 1):
        idx_start = crossings[i]
        idx_end = crossings[i + 1]

        # 提取单个波的数据段
        wave_segment = elev_zero_mean[idx_start:idx_end]

        if len(wave_segment) > 0:
            h = np.max(wave_segment) - np.min(wave_segment)
            waves_height.append(h)

        t = time[idx_end] - time[idx_start]
        waves_period.append(t)

    # 计算统计指标
    if len(waves_height) == 0:
        return 0, 0, 0, 0

    # H1/3
    sorted_H = np.sort(waves_height)[::-1]
    n_third = max(1, int(len(sorted_H) / 3))
    h_1_3 = np.mean(sorted_H[:n_third])

    # Tz
    t_z = np.mean(waves_period)

    # Hmax
    h_max = np.max(waves_height)

    return h_1_3, t_z, h_max, len(waves_height)


def process_data_pipeline(t_target, y_target, t_sim, y_sim,
                          do_detrend, do_zeromean, time_lag, window_duration=10.0):
    """
    综合处理管道：修正 -> 对齐 -> 匹配 -> 统计
    """
    # 1. 对仿真数据应用时间平移
    t_sim_shifted = t_sim + time_lag

    # 2. 确定公共时间轴 (20Hz采样)
    start_time = max(t_target.min(), t_sim_shifted.min())
    end_time = min(t_target.max(), t_sim_shifted.max())

    if end_time - start_time < window_duration:
        return None, "数据重叠时间太短，无法分析"

    dt_common = 0.003
    t_common = np.arange(start_time, end_time, dt_common)

    # 3. 插值同步
    f_target = interp1d(t_target, y_target, kind='linear', bounds_error=False, fill_value=0)
    f_sim = interp1d(t_sim_shifted, y_sim, kind='linear', bounds_error=False, fill_value=0)

    y_target_common = f_target(t_common)
    y_sim_raw_common = f_sim(t_common)

    # 4. 数据修正 (Correction)
    y_sim_corrected = y_sim_raw_common.copy()
    y_target_processed = y_target_common.copy()  # 理论值一般只做去均值，不做Detrend

    if do_detrend:
        # 去除线性漂移
        y_sim_corrected = signal.detrend(y_sim_corrected, type='linear')

    if do_zeromean:
        # 强制归零
        y_sim_corrected = y_sim_corrected - np.mean(y_sim_corrected)
        y_target_processed = y_target_processed - np.mean(y_target_processed)

    # 5. 寻找最佳匹配窗口 (RMSE最小化)
    window_points = int(window_duration / dt_common)
    step_points = int(0.1 / dt_common)
    limit = len(t_common) - window_points

    best_rmse = float('inf')
    best_start_idx = 0

    for i in range(0, limit, step_points):
        seg_target = y_target_processed[i: i + window_points]
        seg_sim = y_sim_corrected[i: i + window_points]
        mse = np.mean((seg_sim - seg_target) ** 2)
        if mse < best_rmse:
            best_rmse = mse
            best_start_idx = i

    best_start_time = t_common[best_start_idx]

    # 6. 计算全域统计参数 (使用修正后的全长时间序列，因为10s太短不足以统计H1/3)
    # 我们对比 "修正后的仿真全序列" vs "理论全序列"
    h13_tgt, tz_tgt, hmax_tgt, n_tgt = zero_crossing_analysis(t_common, y_target_processed)
    h13_sim, tz_sim, hmax_sim, n_sim = zero_crossing_analysis(t_common, y_sim_corrected)

    stats = {
        "Target": {"H1/3": h13_tgt, "Tz": tz_tgt, "Hmax": hmax_tgt, "Count": n_tgt},
        "CFD": {"H1/3": h13_sim, "Tz": tz_sim, "Hmax": hmax_sim, "Count": n_sim}
    }

    return {
        "t_common": t_common,
        "y_target": y_target_processed,
        "y_sim_raw": y_sim_raw_common,
        "y_sim_corr": y_sim_corrected,
        "best_start_time": best_start_time,
        "window_duration": window_duration,
        "rmse": np.sqrt(best_rmse),
        "stats": stats
    }, None


def read_file(uploaded_file):
    if uploaded_file is None: return None
    try:
        if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
            try:
                df = pd.read_csv(uploaded_file)
                pd.to_numeric(df.iloc[:, 0])
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None)
        else:
            df = pd.read_excel(uploaded_file)

        data = df.iloc[:, :2].copy()
        data.columns = ['Time', 'Elevation']
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        return data.sort_values(by='Time')
    except:
        return None


# ===========================
# 2. 前端界面构建
# ===========================

st.title("🌊 CFD 造波质量综合验证工作台")
st.markdown("""
集成 **数据清洗 (Detrend)**、**最佳区间匹配** 与 **波浪参数统计 (H1/3, Tz)**。
用于一站式解决仿真水位漂移、坐标原点偏差及精度验证问题。
""")

with st.sidebar:
    st.header("1. 数据导入")
    f_theo = st.file_uploader("理论波 (Target)", type=['csv', 'xlsx'])
    f_sim = st.file_uploader("仿真波 (CFD)", type=['csv', 'xlsx'])

    st.divider()
    st.header("2. 修正参数")
    do_detrend = st.checkbox("去除线性趋势 (Detrend)", value=True, help="消除水位随时间持续上升/漂移的问题")
    do_zeromean = st.checkbox("去除平均值 (Zero-mean)", value=True, help="消除坐标原点Z=0定义不同带来的固定偏差")
    time_shift = st.number_input("仿真时间平移 (s)", value=0.0, step=0.1, help="调整相位差")

    st.divider()
    st.markdown("Created by AI Assistant")

if f_theo and f_sim:
    df_theo = read_file(f_theo)
    df_sim = read_file(f_sim)

    if df_theo is not None and df_sim is not None:

        # --- 执行处理管道 ---
        with st.spinner("正在执行：趋势修正 -> 匹配搜索 -> 参数统计..."):
            res, err = process_data_pipeline(
                df_theo['Time'].values, df_theo['Elevation'].values,
                df_sim['Time'].values, df_sim['Elevation'].values,
                do_detrend, do_zeromean, time_shift
            )

        if err:
            st.error(err)
        else:
            # === 区域 1: 统计指标对比表格 ===
            st.subheader("1. 波浪统计参数对比 (基于全长公共数据)")

            stats = res['stats']


            # 计算误差百分比
            def calc_err(sim, tgt):
                return (sim - tgt) / tgt * 100 if tgt != 0 else 0


            err_h = calc_err(stats['CFD']['H1/3'], stats['Target']['H1/3'])
            err_t = calc_err(stats['CFD']['Tz'], stats['Target']['Tz'])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("理论 H1/3", f"{stats['Target']['H1/3']:.4f} m")
            col2.metric("仿真 H1/3 (修正后)", f"{stats['CFD']['H1/3']:.4f} m",
                        delta=f"{err_h:.2f}%", delta_color="inverse")

            col3.metric("理论 Tz", f"{stats['Target']['Tz']:.4f} s")
            col4.metric("仿真 Tz (修正后)", f"{stats['CFD']['Tz']:.4f} s",
                        delta=f"{err_t:.2f}%", delta_color="inverse")

            st.caption(
                f"*注：统计样本为重叠时间段内的所有完整波形。理论波识别到 {stats['Target']['Count']} 个波，仿真波识别到 {stats['CFD']['Count']} 个波。")

            # === 区域 2: 可视化图表 ===
            st.divider()
            st.subheader("2. 波形时域对比 & 最佳匹配段标注")

            best_t = res['best_start_time']
            duration = res['window_duration']

            fig = go.Figure()

            # 1. 原始 CFD (半透明蓝)
            fig.add_trace(go.Scatter(
                x=res['t_common'], y=res['y_sim_raw'],
                name='原始仿真 (含漂移)',
                line=dict(color='blue', width=2), opacity=0.8,
                visible='legendonly'  # 默认隐藏，点击图例可看
            ))

            # 2. 修正后 CFD (绿实线)
            fig.add_trace(go.Scatter(
                x=res['t_common'], y=res['y_sim_corr'],
                name='修正后仿真 (Clean)',
                line=dict(color='#2ca02c', width=2)
            ))

            # 3. 理论波 (橙实线)
            fig.add_trace(go.Scatter(
                x=res['t_common'], y=res['y_target'],
                name='理论波 (Target)',
                line=dict(color='#ff7f0e', width=2)
            ))

            # 4. 高亮最佳匹配区域 (矩形背景)
            fig.add_vrect(
                x0=best_t, x1=best_t + duration,
                fillcolor="rgba(44, 160, 44, 0.2)", layer="below", line_width=0,
                annotation_text="最佳匹配 10s", annotation_position="top left"
            )

            fig.update_layout(
                title=f"Time History Comparison (Best Match RMSE = {res['rmse']:.4f} m)",
                xaxis_title="Time (s)",
                yaxis_title="Elevation (m)",
                hovermode="x unified",
                template="plotly_white",
                height=550,
                legend=dict(orientation="h", y=1.1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # === 区域 3: 数据导出 ===
            st.divider()

            # 准备下载数据
            df_out = pd.DataFrame({
                "Time": res['t_common'],
                "Target_Theory": res['y_target'],
                "CFD_Raw": res['y_sim_raw'],
                "CFD_Corrected": res['y_sim_corr']
            })
            csv = df_out.to_csv(index=False).encode('utf-8')

            c_down1, c_down2 = st.columns([1, 4])
            with c_down1:
                st.download_button(
                    "📥 下载修正后数据 (.csv)",
                    data=csv,
                    file_name="validated_wave_data.csv",
                    mime="text/csv",
                    type="primary"
                )
            with c_down2:
                st.info("导出的 CSV 包含：时间、理论值、原始仿真值、去趋势修正后的仿真值。")

    else:
        st.warning("请在左侧上传两个数据文件以开始分析。")
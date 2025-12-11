import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go  # 引入 Plotly 图形对象库

# --- 页面设置 ---
st.set_page_config(page_title="交互式波浪分析工具", layout="wide")  # 使用 wide 布局让图表更宽


def analyze_waves(time, elevation):
    """
    使用上跨零点法 (Zero Up-crossing) 分析波浪数据
    """
    # 1. 去除平均值
    elevation_zero_mean = elevation - np.mean(elevation)

    # 2. 寻找上跨零点
    crossings = np.where(np.diff(np.sign(elevation_zero_mean)) == 2)[0]

    waves_height = []
    waves_period = []

    # 3. 遍历提取波高和周期
    for i in range(len(crossings) - 1):
        idx_start = crossings[i]
        idx_end = crossings[i + 1]

        wave_segment = elevation_zero_mean[idx_start:idx_end]

        if len(wave_segment) > 0:
            h = np.max(wave_segment) - np.min(wave_segment)
            waves_height.append(h)

        t = time[idx_end] - time[idx_start]
        waves_period.append(t)

    return np.array(waves_height), np.array(waves_period), elevation_zero_mean


# --- App 界面 ---

st.title("🌊 交互式不规则波参数分析")
st.markdown("上传数据后，下方的图表支持 **鼠标框选放大**、**双击复原** 和 **悬停查看数值**。")

# 侧边栏上传，节省主空间给图表
with st.sidebar:
    st.header("数据上传")
    uploaded_file = st.file_uploader("选择 CSV/Excel 文件", type=['csv', 'xlsx', 'txt'])
    st.info("格式要求：\n1. 第一列：时间 (s)\n2. 第二列：波高 (m)")

if uploaded_file is not None:
    try:
        # --- 数据读取 (保持不变) ---
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
        data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
        data['Elevation'] = pd.to_numeric(data['Elevation'], errors='coerce')
        data.dropna(inplace=True)

        # --- 计算核心参数 ---
        waves_H, waves_T, elev_zero_mean = analyze_waves(data['Time'].values, data['Elevation'].values)

        if len(waves_H) == 0:
            st.error("无法识别波浪周期，请检查数据。")
        else:
            # 计算统计值
            sorted_H = np.sort(waves_H)[::-1]
            n_third = int(len(sorted_H) / 3)
            h_1_3 = np.mean(sorted_H[:n_third]) if n_third > 0 else np.mean(sorted_H)
            t_z = np.mean(waves_T)
            h_max = np.max(waves_H)

            # --- 结果指标展示 ---
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("三一波高 (H1/3)", f"{h_1_3:.4f} m")
            c2.metric("平均过零周期 (Tz)", f"{t_z:.4f} s")
            c3.metric("最大波高 (Hmax)", f"{h_max:.4f} m")
            c4.metric("识别波数量", f"{len(waves_H)} 个")
            st.divider()

            # --- 交互式绘图 (Plotly) ---

            # 图表 1: 时域历程图
            st.subheader("1. 波浪时域历程 (可缩放)")

            fig_time = go.Figure()

            # 添加波面曲线
            fig_time.add_trace(go.Scatter(
                x=data['Time'],
                y=elev_zero_mean,
                mode='lines',
                name='波面 (去均值)',
                line=dict(color='#1f77b4', width=1.5)
            ))

            # 设置布局
            fig_time.update_layout(
                title='Wave Elevation Time History',
                xaxis_title='Time (s)',
                yaxis_title='Elevation (m)',
                hovermode="x unified",  # 鼠标悬停时显示X轴对应的所有数值
                template="plotly_white",
                height=500
            )

            # 渲染图表
            st.plotly_chart(fig_time, use_container_width=True)

            # 图表 2: 波高分布直方图
            st.subheader("2. 波高分布统计")

            fig_hist = go.Figure()

            # 添加直方图
            fig_hist.add_trace(go.Histogram(
                x=waves_H,
                nbinsx=30,
                name='波高计数',
                marker_color='#2ca02c',
                opacity=0.75
            ))

            # 添加 H1/3 竖线
            fig_hist.add_vline(
                x=h_1_3,
                line_width=3,
                line_dash="dash",
                line_color="red",
                annotation_text=f"H1/3 = {h_1_3:.2f}m",
                annotation_position="top right"
            )

            fig_hist.update_layout(
                title='Wave Height Distribution',
                xaxis_title='Wave Height (m)',
                yaxis_title='Count',
                template="plotly_white",
                bargap=0.1
            )

            st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"出错: {e}")
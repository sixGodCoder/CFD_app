import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- 页面配置 ---
st.set_page_config(page_title="不规则波生成器", layout="wide")

# --- Matplotlib 中文支持设置 (尝试解决乱码) ---
import platform

system_name = platform.system()
if system_name == "Windows":
    plt.rcParams["font.family"] = ["SimHei"]
elif system_name == "Darwin":  # Mac
    plt.rcParams["font.family"] = ["Arial Unicode MS"]
else:  # Linux (Streamlit Cloud 等)
    # 如果在服务器上运行，可能需要指定支持中文的字体文件，或者回退到默认
    plt.rcParams["font.family"] = ["sans-serif"]

plt.rcParams["axes.unicode_minus"] = False


# --- 核心函数 (保留原有逻辑) ---

def check_dataframe_columns(df):
    """检查上传的数据是否包含必要列"""
    required_cols = ['angularFrequency', 'Amplitude', 'Wavenumber', 'Phase']
    missing_cols = [col for col in required_cols if col not in df.columns]

    # 检查数值类型
    if not missing_cols:
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"列 {col} 必须为数值类型"

    if missing_cols:
        return False, f"缺少必要列：{missing_cols}，必须包含：{required_cols}"

    return True, "格式正确"


def generate_irregular_wave(param_df, x_probe, dt=0.01, total_time=100):
    """生成指定检测点的不规则波时域波高"""
    t = np.arange(0, total_time, dt)
    eta = np.zeros_like(t, dtype=np.float64)

    # 显示进度条 (Streamlit 特有优化)
    progress_text = "正在叠加简谐波分量..."
    my_bar = st.progress(0, text=progress_text)

    total_rows = len(param_df)

    for idx, row in param_df.iterrows():
        omega = row['angularFrequency']
        amp = row['Amplitude']
        k = row['Wavenumber']
        phase = row['Phase']

        eta_i = amp * np.cos(k * x_probe - omega * t + phase)
        eta += eta_i

        # 更新进度条 (每10%更新一次，避免太频繁)
        if idx % (max(1, total_rows // 10)) == 0:
            my_bar.progress(int((idx / total_rows) * 100), text=progress_text)

    my_bar.empty()  # 清除进度条
    return t, eta


def plot_waveform_mpl(t, eta, x_probe):
    """使用 Matplotlib 绘图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, eta, linewidth=0.8, color='#1f77b4')
    ax.set_xlabel('Time (s)')  # 使用英文以防服务器无中文字体
    ax.set_ylabel('Wave Height (m)')
    ax.set_title(f'Irregular Wave at x={x_probe:.2f}m')
    ax.grid(alpha=0.3)
    return fig


# --- 主程序逻辑 (UI 构建) ---

def main():
    st.title("🌊 不规则波时域生成器 (Online App)")
    st.markdown("上传包含简谐波分量的 CSV 文件，生成特定位置的时域波形数据。")

    # 1. 侧边栏：文件上传与参数设置
    with st.sidebar:
        st.header("1. 参数设置")

        uploaded_file = st.file_uploader("上传参数文件 (CSV)", type=["csv"])

        st.subheader("位置参数")
        x_probe = st.number_input("检测点位置 x (m)", value=0.0, step=1.0, format="%.2f")

        st.subheader("时间参数")
        dt = st.number_input("时间步长 dt (s)", value=0.01, step=0.001, format="%.3f")
        total_time = st.number_input("总时长 (s)", value=100.0, step=10.0)

        # 下载模板文件的辅助功能
        st.markdown("---")
        st.markdown("还没有文件？")
        sample_data = pd.DataFrame({
            'angularFrequency': [0.5, 0.6, 0.7],
            'amplitude': [0.1, 0.2, 0.15],
            'wavenumber': [0.1, 0.15, 0.2],
            'phase': [0, 1.5, 3.14]
        })
        csv_template = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下载示例模板 CSV",
            data=csv_template,
            file_name="template_wave_params.csv",
            mime="text/csv"
        )

    # 2. 主界面逻辑
    if uploaded_file is not None:
        try:
            param_df = pd.read_csv(uploaded_file)

            # 检查数据格式
            is_valid, message = check_dataframe_columns(param_df)

            if not is_valid:
                st.error(f"文件错误: {message}")
            else:
                st.success(f"✅ 文件读取成功，包含 {len(param_df)} 个波分量")

                # 数据预览 (可折叠)
                with st.expander("查看上传的参数数据"):
                    st.dataframe(param_df.head(10))

                # 3. 触发计算
                if st.button("开始生成波形", type="primary"):
                    # 计算逻辑
                    t, eta = generate_irregular_wave(param_df, x_probe, dt, total_time)

                    # 结果 DataFrame
                    result_df = pd.DataFrame({
                        'Time(s)': t,
                        'WaveHeight(m)': eta
                    })

                    # 4. 可视化
                    st.subheader(f"📊 波形可视化 (x = {x_probe}m)")

                    # 方式 A: 交互式图表 (推荐 Web 使用)
                    st.line_chart(result_df.set_index('Time(s)'), height=350)

                    # 方式 B: 传统 Matplotlib 图表 (保留你的原始风格)
                    # st.pyplot(plot_waveform_mpl(t, eta, x_probe))

                    # 5. 数据下载
                    st.subheader("💾 数据导出")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.info(f"数据点数: {len(result_df)}")
                    with col2:
                        csv_data = result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        output_filename = f"irregular_wave_x{x_probe:.2f}m.csv"

                        st.download_button(
                            label="下载生成的时域数据 (CSV)",
                            data=csv_data,
                            file_name=output_filename,
                            mime="text/csv",
                            type="primary"
                        )

        except Exception as e:
            st.error(f"读取或处理文件时发生错误: {e}")
    else:
        st.info("👈 请在左侧上传 CSV 参数文件以开始。")


if __name__ == "__main__":
    main()
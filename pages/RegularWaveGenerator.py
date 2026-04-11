import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import newton
import io

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="规则波生成器 (Airy Wave)",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 线性规则波仿真平台 (Airy Wave Theory)")
st.markdown("""
本工具基于 **Airy 线性波理论** 生成单一频率的规则波。
适用于水动力学计算、基础波浪力分析及 CFD 入口条件设置。
""")

# ---------------------- 1. 侧边栏：参数设置 ----------------------
st.sidebar.header("1. 波浪参数设置")

H_w = st.sidebar.number_input("波高 H (m)", min_value=0.01, value=2.0, step=0.1, format="%.3f")
T_w = st.sidebar.number_input("周期 T (s)", min_value=0.1, value=6.0, step=0.1, format="%.2f")
h_water = st.sidebar.number_input("水深 h (m)", min_value=0.1, value=50.0, step=1.0)
phase_init = st.sidebar.slider("初始相位 (deg)", 0, 360, 0)

st.sidebar.subheader("仿真与采样")
total_time = st.sidebar.number_input("总时长 (s)", min_value=1.0, value=60.0, step=10.0)
dt = st.sidebar.number_input("时间步长 (s)", min_value=0.001, value=0.05, step=0.01, format="%.4f")
x_probe = st.sidebar.number_input("检测点位置 x (m)", min_value=0.0, value=0.0, step=1.0)

# ---------------------- 核心计算逻辑 ----------------------
g = 9.81

def get_wavenumber(omega, h):
    """根据色散关系计算波数 k"""
    k0 = omega ** 2 / g # 深水估值
    def f(k):
        return omega ** 2 - g * k * np.tanh(k * h)
    try:
        return newton(f, k0, maxiter=100)
    except:
        return k0

# --- 计算派生参数 ---
omega = 2 * np.pi / T_w
k = get_wavenumber(omega, h_water)
wavelength = 2 * np.pi / k
amplitude = H_w / 2
eps = np.radians(phase_init)

# ---------------------- 结果展示 ----------------------

# 计算波时历
t = np.arange(0, total_time, dt)
# 公式: zeta = A * cos(kx - wt + eps)
zeta = amplitude * np.cos(k * x_probe - omega * t + eps)

# 1. 物理特性指标
c1, c2, c3, c4 = st.columns(4)
c1.metric("波长 L", f"{wavelength:.2f} m")
c2.metric("波数 k", f"{k:.4f} rad/m")
c3.metric("角频率 ω", f"{omega:.4f} rad/s")
kh = k * h_water
water_type = "深水波" if kh > np.pi else ("浅水波" if kh < 0.3 else "有限水深波")
c4.metric("水深性质", water_type)

# 2. 绘图
st.subheader(f"检测点 (x={x_probe}m) 波浪时历曲线")
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(t, zeta, color='#1f77b4', linewidth=1.5, label=f'H={H_w}m, T={T_w}s')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Wave Height (m)")
ax.set_ylim(-amplitude*1.5, amplitude*1.5)
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# 3. 导出公式文本 (用于 STAR-CCM+ 等)
st.divider()
st.subheader("📝 导出 CFD 字段函数 (Field Functions)")

col_v1, col_v2, col_v3 = st.columns(3)
var_t = col_v1.text_input("时间变量", value="${Time}")
var_x = col_v2.text_input("水平坐标", value="$$Position[0]")
var_z = col_v3.text_input("垂直坐标", value="$$Position[2]")

# 生成公式
coeff_uv = (g * amplitude * k) / omega
cosh_kh = np.cosh(k * h_water)

formula_eta = f"{amplitude:.10g} * cos({k:.10g}*{var_x} - {omega:.10g}*{var_t} + {eps:.10g})"
formula_u = f"({coeff_uv:.10g} * cosh({k:.10g}*({h_water} + {var_z})) / {cosh_kh:.10g}) * cos({k:.10g}*{var_x} - {omega:.10g}*{var_t} + {eps:.10g})"
formula_w = f"({coeff_uv:.10g} * sinh({k:.10g}*({h_water} + {var_z})) / {cosh_kh:.10g}) * sin({k:.10g}*{var_x} - {omega:.10g}*{var_t} + {eps:.10g})"

st.code(f"""
// 1. 波面高度 eta
eta = {formula_eta};

// 2. 水平速度 u (有限水深)
u = {formula_u};

// 3. 垂直速度 w (有限水深)
w = {formula_w};
""", language="cpp")

# 4. 数据下载
csv_data = pd.DataFrame({'Time(s)': t, 'WaveHeight(m)': zeta}).to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 下载时域波高数据 (CSV)", data=csv_data, file_name=f"RegularWave_H{H_w}_T{T_w}.csv")
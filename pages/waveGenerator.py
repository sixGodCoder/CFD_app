import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import newton
import io

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="不规则波浪生成器 (JONSWAP)",
    page_icon="🌊",
    layout="wide"
)

# 解决Matplotlib中文显示问题
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans", "Arial"]
except:
    pass

st.title("🌊 高精度不规则波浪仿真平台 (JONSWAP)")
st.markdown("""
本工具基于 **JONSWAP 谱** 生成不规则波浪。
**更新说明**：已修复公式导出时的精度问题，自动保留所有有效数字，防止微小振幅被截断为0。
""")

# ---------------------- 1. 侧边栏：参数设置 ----------------------
st.sidebar.header("1. 参数设置")

# 物理参数
st.sidebar.subheader("波浪参数")

H_s_input = st.sidebar.number_input(
    "理论三一波高 H₁/₃ (m)",
    min_value=0.01,
    value=2.00,
    step=0.001,
    format="%.3f"
)

T_z_input = st.sidebar.number_input(
    "理论过零周期 T_z (s)",
    min_value=0.1,
    value=6.00,
    step=0.01,
    format="%.2f"
)

h_water = st.sidebar.number_input("水深 h (m)", min_value=0.1, value=50.0, step=1.0)

# 仿真参数
st.sidebar.subheader("水槽与时间")
L_tank = st.sidebar.number_input("水池长度 (m)", min_value=10.0, value=200.0, step=10.0)

x_probe = st.sidebar.number_input(
    "检测点位置 x (m)",
    min_value=0.0,
    max_value=L_tank,
    value=L_tank / 2,
    step=0.1,
    format="%.2f"
)

total_time = st.sidebar.number_input("总时长 (s)", min_value=1.0, value=100.0, step=10.0)

dt = st.sidebar.number_input(
    "时间步长 (s)",
    min_value=0.0001,
    value=0.1,
    step=0.01,
    format="%.4f",
    help="越小的步长精度越高，但计算数据量越大。"
)

# 精度控制
st.sidebar.subheader("精度控制")
N_waves = st.sidebar.number_input("组成波数量 (N)", min_value=10, max_value=5000, value=200, step=50)
seed_val = st.sidebar.number_input("随机种子 (可选)", value=0)
run_optimization = st.sidebar.checkbox("启用参数自动迭代优化", value=True)

# ---------------------- 核心计算函数 ----------------------

g = 9.81


@st.cache_data
def jonswap_spectrum(omega, alpha, omega_p, gamma=3.3):
    omega = np.maximum(omega, 1e-6)
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    term1 = alpha * g ** 2 / (omega ** 5)
    term2 = np.exp(-1.25 * (omega_p / omega) ** 4)
    term3 = gamma ** np.exp(-((omega - omega_p) ** 2) / (2 * sigma ** 2 * omega_p ** 2))
    return term1 * term2 * term3


def dispersion_relation(omega, h):
    if omega == 0: return 0
    k0 = omega ** 2 / g
    if h < 100: k0 = omega / np.sqrt(g * h)

    def f(k):
        return omega ** 2 - g * k * np.tanh(k * h)

    try:
        return newton(f, k0, maxiter=50)
    except:
        return omega ** 2 / g


def calculate_wave_elevation_chunked(t_array, x_val, omega, k, zeta_An, eps, chunk_size=10000):
    zeta = np.zeros(len(t_array))
    N_steps = len(t_array)
    phase_space = k * x_val + eps
    for i in range(0, N_steps, chunk_size):
        end_idx = min(i + chunk_size, N_steps)
        t_chunk = t_array[i: end_idx]
        phase_matrix = phase_space[np.newaxis, :] - t_chunk[:, np.newaxis] * omega[np.newaxis, :]
        zeta[i: end_idx] = np.sum(zeta_An * np.cos(phase_matrix), axis=1)
    return zeta


def generate_wave_params(alpha, omega_p, h, N, seed=None):
    T_min = max(0.5, T_z_input * 0.2)
    T_max = min(50, T_z_input * 5)
    omega = np.linspace(2 * np.pi / T_max, 2 * np.pi / T_min, N)
    domega = omega[1] - omega[0]
    S_omega = jonswap_spectrum(omega, alpha, omega_p)
    zeta_An = np.sqrt(2 * S_omega * domega)
    if seed is not None and seed != 0:
        np.random.seed(seed)
    else:
        np.random.seed(None)
    eps = 2 * np.pi * np.random.rand(N)
    k = np.array([dispersion_relation(om, h) for om in omega])
    return omega, zeta_An, k, eps


def calculate_stats(t, zeta):
    zero_crossings = np.where(np.diff(np.signbit(zeta)))[0]
    up_crossings = []
    for idx in zero_crossings:
        if zeta[idx] < 0 and zeta[idx + 1] >= 0:
            t_c = t[idx] + (0 - zeta[idx]) * (t[idx + 1] - t[idx]) / (zeta[idx + 1] - zeta[idx])
            up_crossings.append(t_c)
    wave_heights = []
    if len(up_crossings) > 1:
        uc_indices = np.searchsorted(t, up_crossings)
        for i in range(len(uc_indices) - 1):
            s, e = uc_indices[i], uc_indices[i + 1]
            if e > s: wave_heights.append(np.max(zeta[s:e]) - np.min(zeta[s:e]))
    wave_heights = np.array(wave_heights)
    H_13 = np.nan
    if len(wave_heights) >= 5:
        wave_heights_sorted = np.sort(wave_heights)[::-1]
        n_13 = max(1, int(np.ceil(len(wave_heights) / 3)))
        H_13 = np.mean(wave_heights_sorted[:n_13])
    T_z = np.mean(np.diff(up_crossings)) if len(up_crossings) > 1 else np.nan
    return H_13, T_z


# ---------------------- 公式生成函数 (核心修改处) ----------------------
def generate_formula_text(df_components, depth, g, var_t, var_x, var_z):
    """
    生成 CFD 可用的公式文本
    修改说明：使用 .20g 格式化，保留20位有效数字，极小值自动转为科学计数法，防止截断为0。
    """
    buffer = io.StringIO()

    n_components = len(df_components)

    buffer.write(f"// 参数设置:\n// Water Depth h = {depth} m\n// Gravity g = {g} m/s^2\n")
    buffer.write(f"// Components N = {n_components}\n\n")

    # 1. 波面高度公式
    buffer.write("// ==========================================\n")
    buffer.write("// 1. 波面高度 (Wave Elevation) eta\n")
    buffer.write("// ==========================================\n")
    buffer.write(f"eta = \n")

    for i in range(n_components):
        row = df_components.iloc[i]
        a = row['Amplitude']
        w = row['angularFrequency']
        k = row['Wavenumber']
        e = row['Phase']

        # 修改点：使用 .20g 代替 .8f
        term = f"({a:.20g} * cos({k:.20g}*{var_x} - {w:.20g}*{var_t} + {e:.20g}))"

        if i < n_components - 1:
            buffer.write(f"  {term} +\n")
        else:
            buffer.write(f"  {term};\n\n")

    # 2. 速度势 (Finite Depth - Hyperbolic)
    buffer.write("// ==========================================\n")
    buffer.write("// 2. 速度分量 - 有限水深 (Finite Depth / Sinh-Cosh)\n")
    buffer.write("// 水平速度u = dPhi/dx, 垂直速度w = dPhi/dz\n")
    buffer.write("// ==========================================\n")

    # u_sinh
    buffer.write("水平速度u_finite_depth = \n")
    for i in range(n_components):
        row = df_components.iloc[i]
        a = row['Amplitude']
        w = row['angularFrequency']
        k = row['Wavenumber']
        e = row['Phase']

        coeff = (g * a * k) / w
        # 修改点：使用 .20g 代替 .8f
        term = (f"({coeff:.20g} * "
                f"(cosh({k:.20g}*({depth} + {var_z})) / cosh({k:.20g}*{depth})) * "
                f"cos({k:.20g}*{var_x} - {w:.20g}*{var_t} + {e:.20g}))")

        if i < n_components - 1:
            buffer.write(f"  {term} +\n")
        else:
            buffer.write(f"  {term};\n\n")

    # w_sinh
    buffer.write("垂直速度w_finite_depth = \n")
    for i in range(n_components):
        row = df_components.iloc[i]
        a = row['Amplitude']
        w = row['angularFrequency']
        k = row['Wavenumber']
        e = row['Phase']

        coeff = (g * a * k) / w
        # 修改点：使用 .20g 代替 .8f
        term = (f"({coeff:.20g} * "
                f"(sinh({k:.20g}*({depth} + {var_z})) / cosh({k:.20g}*{depth})) * "
                f"sin({k:.20g}*{var_x} - {w:.20g}*{var_t} + {e:.20g}))")

        if i < n_components - 1:
            buffer.write(f"  {term} +\n")
        else:
            buffer.write(f"  {term};\n\n")

    # 3. 速度势 (Deep Water - Exponential Approximation)
    buffer.write("// ==========================================\n")
    buffer.write("// 3. 速度分量 - 深水近似 (Deep Water / Exponential)\n")
    buffer.write("// 适用于 kh >> 1, 近似 cosh(k(h+z))/cosh(kh) -> exp(kz)\n")
    buffer.write("// ==========================================\n")

    # u_exp
    buffer.write("水平速度u_exp_approx = \n")
    for i in range(n_components):
        row = df_components.iloc[i]
        a = row['Amplitude']
        w = row['angularFrequency']
        k = row['Wavenumber']
        e = row['Phase']

        coeff = (g * a * k) / w
        # 修改点：使用 .20g 代替 .8f
        term = (f"({coeff:.20g} * "
                f"exp({k:.20g}*{var_z}) * "
                f"cos({k:.20g}*{var_x} - {w:.20g}*{var_t} + {e:.20g}))")

        if i < n_components - 1:
            buffer.write(f"  {term} +\n")
        else:
            buffer.write(f"  {term};\n\n")

    # w_exp
    buffer.write("垂直速度w_exp_approx = \n")
    for i in range(n_components):
        row = df_components.iloc[i]
        a = row['Amplitude']
        w = row['angularFrequency']
        k = row['Wavenumber']
        e = row['Phase']

        coeff = (g * a * k) / w
        # 修改点：使用 .20g 代替 .8f
        term = (f"({coeff:.20g} * "
                f"exp({k:.20g}*{var_z}) * "
                f"sin({k:.20g}*{var_x} - {w:.20g}*{var_t} + {e:.20g}))")

        if i < n_components - 1:
            buffer.write(f"  {term} +\n")
        else:
            buffer.write(f"  {term};\n")

    return buffer.getvalue()


# ---------------------- 主逻辑 ----------------------

if st.sidebar.button("开始生成波浪", type="primary"):

    status_text = st.empty()
    progress_bar = st.progress(0)
    status_text.text("正在计算...请稍候")

    # --- 计算逻辑 ---
    omega_char = 2 * np.pi / T_z_input
    k_char = dispersion_relation(omega_char, h_water)
    lambda_char = 2 * np.pi / k_char if k_char > 0 else 100.0
    dx = lambda_char / 20
    num_x = int(np.ceil(L_tank / dx))
    x_points = np.linspace(0, L_tank, num_x)

    target_H13 = H_s_input
    target_Tz = T_z_input
    omega_p = 2 * np.pi / (target_Tz * 0.78)
    alpha = 5.061 * (target_H13 / 4) ** 2 * omega_p ** 4 / g ** 2

    best_alpha, best_omega_p = alpha, omega_p

    if run_optimization:
        status_text.text("正在优化参数 (Step 1/2)...")
        tolerance = 0.05
        max_iter = 15
        dt_opt = max(dt, 0.05)
        t_opt_dur = max(300, 20 * target_Tz)
        t_test = np.arange(0, t_opt_dur, dt_opt)

        for i in range(max_iter):
            omega, zeta_An, k, eps = generate_wave_params(alpha, omega_p, h_water, N_waves, seed=42)
            zeta_test = calculate_wave_elevation_chunked(t_test, L_tank / 2, omega, k, zeta_An, eps)
            curr_H13, curr_Tz = calculate_stats(t_test, zeta_test)
            h_err = (curr_H13 - target_H13) / target_H13
            t_err = (curr_Tz - target_Tz) / target_Tz
            progress_bar.progress(int((i + 1) / max_iter * 50))
            if abs(h_err) < tolerance and abs(t_err) < tolerance:
                best_alpha, best_omega_p = alpha, omega_p
                break
            if not np.isnan(curr_H13) and curr_H13 > 0: alpha *= (target_H13 / curr_H13) ** 2
            if not np.isnan(curr_Tz) and curr_Tz > 0: omega_p *= (curr_Tz / target_Tz)
            alpha = max(1e-5, alpha)
            omega_p = np.clip(omega_p, 0.1, 10.0)
            best_alpha, best_omega_p = alpha, omega_p

    status_text.text("生成全场数据 (Step 2/2)...")
    progress_bar.progress(70)

    t = np.arange(0, total_time, dt)
    omega, zeta_An, k, eps = generate_wave_params(best_alpha, best_omega_p, h_water, N_waves,
                                                  seed=seed_val if seed_val != 0 else None)
    zeta_probe = calculate_wave_elevation_chunked(t, x_probe, omega, k, zeta_An, eps, chunk_size=50000)
    final_H13, final_Tz = calculate_stats(t, zeta_probe)

    progress_bar.progress(100)
    status_text.empty()

    # 存入 Session State
    st.session_state['has_data'] = True
    st.session_state['results'] = {
        't': t,
        'zeta': zeta_probe,
        'final_H13': final_H13,
        'final_Tz': final_Tz,
        'omega': omega,
        'zeta_An': zeta_An,
        'k': k,
        'eps': eps,
        'x_probe': x_probe,
        'params': {
            'Hs': H_s_input,
            'Tz': T_z_input,
            'dt': dt
        }
    }

# ---------------------- 结果展示 ----------------------

if st.session_state.get('has_data'):
    res = st.session_state['results']
    params = res['params']

    file_prefix = f"JONSWAP_Hs{params['Hs']:.3f}_Tz{params['Tz']:.2f}_dt{params['dt']:.4f}"

    st.divider()

    # 1. 指标展示
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目标 H₁/₃", f"{H_s_input:.3f} m")
    c2.metric("实际 H₁/₃", f"{res['final_H13']:.3f} m",
              delta=f"{(res['final_H13'] - H_s_input) / H_s_input * 100:.1f}%")
    c3.metric("目标 T_z", f"{T_z_input:.3f} s")
    c4.metric("实际 T_z", f"{res['final_Tz']:.3f} s", delta=f"{(res['final_Tz'] - T_z_input) / T_z_input * 100:.1f}%")

    # 2. 绘图
    st.subheader(f"检测点 (x={res['x_probe']:.2f}m) 波浪时历曲线")
    display_limit = 10000
    t_disp = res['t']
    z_disp = res['zeta']
    if len(t_disp) > display_limit:
        step_disp = len(t_disp) // display_limit
        t_disp = t_disp[::step_disp]
        z_disp = z_disp[::step_disp]
        st.caption(f"注：当前数据点过多，图表已降采样显示。")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t_disp, z_disp, linewidth=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Wave Height (m)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(res['final_H13'] / 2, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(-res['final_H13'] / 2, color='r', linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    # 数据准备
    df_components = pd.DataFrame({
        'angularFrequency': res['omega'],
        'Amplitude': res['zeta_An'],
        'Wavenumber': res['k'],
        'Phase': res['eps']
    })

    # 3. 下载区域
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        csv_probe = pd.DataFrame({'Time(s)': res['t'], 'WaveHeight(m)': res['zeta']}).to_csv(index=False).encode(
            'utf-8-sig')
        st.download_button(
            label="📥 下载检测点数据 (CSV)",
            data=csv_probe,
            file_name=f"{file_prefix}_probe.csv",
            mime="text/csv"
        )
    with c_d2:
        csv_components = df_components.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载组成波参数 (CSV)",
            data=csv_components,
            file_name=f"{file_prefix}_components.csv",
            mime="text/csv"
        )

    # ---------------------- 新增功能：公式导出 ----------------------
    st.divider()
    st.subheader("📝 导出数学公式 (CFD Field Functions)")
    st.markdown("自定义变量名，并生成可直接复制到 STAR-CCM+、Fluent 或代码中的公式文本。")

    col_v1, col_v2, col_v3 = st.columns(3)
    var_t_name = col_v1.text_input("时间变量名 (Time)", value="${Time}")
    var_x_name = col_v2.text_input("水平坐标变量名 (X)", value="$${Position}[0]")
    var_z_name = col_v3.text_input("垂直坐标变量名 (Z)", value="${Position}[2]")

    # 生成文本
    formula_txt = generate_formula_text(
        df_components,
        depth=h_water,
        g=g,
        var_t=var_t_name,
        var_x=var_x_name,
        var_z=var_z_name
    )

    st.download_button(
        label="📥 下载完整公式文件 (.txt)",
        data=formula_txt,
        file_name=f"{file_prefix}_formulas.txt",
        mime="text/plain",
        type="primary"
    )

    with st.expander("👁️ 预览公式前10项"):
        # 只显示前10行预览，避免卡顿
        preview_lines = formula_txt.split('\n')[:50]
        st.code('\n'.join(preview_lines) + "\n...", language="c")

else:
    st.info("👈 请在左侧设置参数，然后点击“开始生成波浪”。")
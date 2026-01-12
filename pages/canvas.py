import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re

import kaleido
# --- 1. 页面设置 ---
st.set_page_config(page_title="论文绘图 - 全能导出版", layout="wide", page_icon="🎓")

st.title("🎓 论文数据绘图 (含多格式导出)")
st.markdown("支持自定义标题、智能 X 轴识别、0 轴高亮，并提供 **SVG/PNG/PDF** 多种高清格式导出。")

uploaded_file = st.file_uploader("📂 上传 Excel/CSV 文件", type=["xlsx", "csv"])


# --- 核心：智能解析与清洗 (保持不变) ---
@st.cache_data
def parse_and_clean_smart(file, keyword="探针"):
    try:
        if file.name.endswith('.csv'):
            df_raw = pd.read_csv(file, header=None, dtype=str)
        else:
            df_raw = pd.read_excel(file, header=None, dtype=str)
    except Exception as e:
        return None, f"文件读取失败: {e}"

    # 1. 定位表头
    header_indices = df_raw.index[
        df_raw.apply(lambda row: row.astype(str).str.contains(keyword).any(), axis=1)].tolist()

    if not header_indices:
        return None, f"未在文件中找到关键词 '{keyword}'，请检查文件或更改默认关键词。"

    parsed_data = []

    # 2. 智能分块策略
    if len(header_indices) > 1:
        # === 模式 A: 多重表头 ===
        for i, start_row in enumerate(header_indices):
            cond_name = f"工况_{i + 1}"
            if start_row > 0:
                val = str(df_raw.iloc[start_row - 1, 0]).strip()
                if val and val.lower() not in ['nan', 'none', '']:
                    cond_name = val

            end_row = header_indices[i + 1] - 1 if i < len(header_indices) - 1 else len(df_raw)

            chunk = df_raw.iloc[start_row + 1: end_row].copy()
            chunk.columns = df_raw.iloc[start_row].tolist()
            chunk["工况"] = cond_name
            parsed_data.append(chunk)

    else:
        # === 模式 B: 分割行 ===
        header_row_idx = header_indices[0]
        headers = df_raw.iloc[header_row_idx].tolist()

        chunk_buffer = []
        current_condition = "默认工况"

        data_rows = df_raw.iloc[header_row_idx + 1:].copy()
        for idx, row in data_rows.iterrows():
            first_val = str(row.iloc[0]).strip()
            non_empty_cnt = row.count()
            if non_empty_cnt <= 2 and first_val not in ['nan', 'None', '']:
                current_condition = first_val
            elif first_val not in ['nan', 'None', '']:
                chunk_buffer.append(row.tolist() + [current_condition])

        if chunk_buffer:
            parsed_data.append(pd.DataFrame(chunk_buffer, columns=headers + ["工况"]))

    if not parsed_data:
        return None, "未提取到任何数据行。"

    # 3. 合并
    df_final = pd.concat(parsed_data, ignore_index=True)

    # 4. 智能清洗
    df_final.columns = df_final.columns.str.strip()

    for col in df_final.columns:
        if col == "工况": continue

        original_series = df_final[col].astype(str).str.strip()
        clean_series = original_series.str.replace('%', '', regex=False)
        numeric_series = pd.to_numeric(clean_series, errors='coerce')

        non_na_count_before = original_series[original_series != 'nan'].count()
        non_na_count_after = numeric_series.count()

        if non_na_count_before > 0:
            loss_rate = 1 - (non_na_count_after / non_na_count_before)
            if loss_rate > 0.5:
                df_final[col] = original_series  # 保留文本
            else:
                df_final[col] = numeric_series
        else:
            df_final[col] = numeric_series

    df_final.dropna(how='all', inplace=True)

    return df_final, None


# --- 主逻辑 ---
if uploaded_file:
    with st.expander("🛠️ 解析设置 (如列找不到请点这里)", expanded=False):
        keyword = st.text_input("定位关键词 (数据中第一列的列名)", value="探针")

    df, error = parse_and_clean_smart(uploaded_file, keyword)

    if error:
        st.error(error)
    else:
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        default_x = next((c for c in cols if keyword in c), cols[0])
        default_y = next((c for c in numeric_cols if "误差" in c or "Error" in c),
                         numeric_cols[0] if numeric_cols else None)

        if not numeric_cols:
            st.error("❌ 未检测到有效数值列，无法绘图。")
            st.stop()

        # --- 2. 绘图配置面板 ---
        st.markdown("### ⚙️ 图表设置")
        tab_data, tab_text, tab_style = st.tabs(["📊 数据映射", "📝 标题与标签", "🎨 样式与导出"])

        with tab_data:
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("X 轴 (探针/位置)", cols, index=cols.index(default_x) if default_x in cols else 0)
            with c2:
                y_col = st.selectbox("Y 轴 (数值指标)", numeric_cols,
                                     index=numeric_cols.index(default_y) if default_y else 0)

        with tab_text:
            c3, c4 = st.columns(2)
            with c3:
                auto_title = f"{y_col}随{x_col}变化对比"
                custom_title = st.text_input("主标题内容", value=auto_title)
                title_size = st.number_input("主标题字号", 10, 40, 20)
                title_align = st.radio("标题对齐", ["居中 (Center)", "居左 (Left)"], horizontal=True)
            with c4:
                custom_x_label = st.text_input("X 轴标签", value=x_col)
                custom_y_label = st.text_input("Y 轴标签", value=y_col)
                label_size = st.number_input("轴标签字号", 10, 30, 16)

        with tab_style:
            c5, c6, c7 = st.columns(3)
            with c5:
                font_family = st.selectbox("字体", ["Times New Roman", "Arial", "SimSun"], index=0)
                show_zero_line = st.toggle("✨ 突出显示 y=0 基准线", value=True)
            with c6:
                marker_size = st.slider("标记点大小", 4, 15, 8)
                line_width = st.slider("线条宽度", 1.0, 5.0, 2.0)
            with c7:
                legend_pos = st.selectbox("图例位置", ["图表上方", "图表内部", "图表右侧"], index=0)
                show_grid = st.toggle("显示网格", value=True)

        # --- 3. 绘图执行 ---
        fig = go.Figure()

        groups = df["工况"].unique()
        symbols = ['circle', 'square', 'triangle-up', 'diamond', 'x', 'cross']
        colors = ['#000000', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']

        x_order = df[x_col].unique().tolist()

        for idx, group in enumerate(groups):
            sub_df = df[df["工况"] == group]
            sub_df = sub_df.dropna(subset=[y_col])

            if sub_df.empty: continue

            fig.add_trace(go.Scatter(
                x=sub_df[x_col],
                y=sub_df[y_col],
                mode='lines+markers',
                name=str(group),
                marker=dict(symbol=symbols[idx % len(symbols)], size=marker_size, line=dict(width=1, color='white')),
                line=dict(width=line_width, color=colors[idx % len(colors)])
            ))

        # 绘制 y=0 基准线
        if show_zero_line:
            fig.add_hline(
                y=0,
                line_width=2,
                line_color="black",
                line_dash="dash",
                opacity=0.7
            )

        # 布局应用
        font_cfg = dict(family=font_family, size=label_size, color="black")

        legend_cfg = dict(font=dict(family=font_family, size=14))
        if legend_pos == "图表内部":
            legend_cfg.update(dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.8)",
                                   bordercolor="black", borderwidth=1))
        elif legend_pos == "图表上方":
            legend_cfg.update(dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))

        title_x = 0.5 if "居中" in title_align else 0.02

        fig.update_layout(
            title=dict(text=custom_title, x=title_x, font=dict(family=font_family, size=title_size)),
            xaxis=dict(
                title=custom_x_label,
                title_font=font_cfg, tickfont=font_cfg,
                showline=True, mirror=True, linecolor='black', linewidth=2,
                showgrid=show_grid, gridcolor='lightgrey',
                type='category',
                categoryorder='array',
                categoryarray=x_order
            ),
            yaxis=dict(
                title=custom_y_label,
                title_font=font_cfg, tickfont=font_cfg,
                showline=True, mirror=True, linecolor='black', linewidth=2,
                showgrid=show_grid, gridcolor='lightgrey',
                zeroline=False
            ),
            legend=legend_cfg,
            plot_bgcolor='white',
            hovermode="x unified",
            height=600,
            margin=dict(l=60, r=40, t=80, b=60)
        )

        # 渲染交互式图表 (保留右上角 SVG 下载)
        st.plotly_chart(fig, use_container_width=True,
                        config={'toImageButtonOptions': {'format': 'svg', 'filename': 'plot_svg', 'scale': 2}})

        # --- 新增：多格式导出区域 ---
        st.markdown("---")
        with st.expander("📤 导出其他格式 (PNG / PDF 高清大图)", expanded=False):
            st.info("提示：SVG 矢量图请直接点击上方图表右上角的相机图标 📷。下方按钮用于导出高分辨率位图或 PDF 文档。")
            col_exp1, col_exp2 = st.columns(2)

            # 定义导出参数
            export_width = 1200
            export_height = 800
            export_scale = 3  # 3倍缩放，保证极高清晰度

            with col_exp1:
                # 生成 PNG
                try:
                    # 使用 kaleido 引擎生成静态图片
                    img_bytes_png = fig.to_image(format="png", width=export_width, height=export_height,
                                                 scale=export_scale)
                    st.download_button(
                        label="🖼️ 下载高分辨率 PNG (位图)",
                        data=img_bytes_png,
                        file_name="academic_plot.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"PNG 生成失败。请检查是否已安装 kaleido 库 (pip install kaleido)。错误: {e}")

            with col_exp2:
                # 生成 PDF
                try:
                    img_bytes_pdf = fig.to_image(format="pdf", width=export_width, height=export_height,
                                                 scale=export_scale)
                    st.download_button(
                        label="📄 下载高分辨率 PDF (矢量文档)",
                        data=img_bytes_pdf,
                        file_name="academic_plot.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF 生成失败。请检查是否已安装 kaleido 库。错误: {e}")

else:
    st.info("👆 请上传文件。")
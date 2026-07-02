import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ============================================================
# 0) CẤU HÌNH TRANG (LỆNH STREAMLIT ĐẦU TIÊN)
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Dự báo Rủi ro Tín dụng 5C",
    page_icon="🏦",
)

# ============================================================
# 1) HẰNG SỐ & HÀM DÙNG CHUNG
# ============================================================
TARGET_COL = "PD"

# Tập biến đầu vào X trích từ notebook (mô hình 5C: Tính cách - Năng lực -
# Điều kiện - Vốn - Tài sản đảm bảo)
FEATURE_COLS = [
    "TC1", "TC2", "TC3", "TC4", "TC5",
    "NL1", "NL2", "NL3", "NL4",
    "DK1", "DK2", "DK3", "DK4", "DK5",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "TS1", "TS2", "TS3", "TS4",
]

CATEGORY_LABELS = {
    "TC": "Tính cách (Character)",
    "NL": "Năng lực (Capacity)",
    "DK": "Điều kiện (Conditions)",
    "V": "Vốn (Capital)",
    "TS": "Tài sản đảm bảo (Collateral)",
}


@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Nạp dữ liệu từ bytes của file CSV (giống notebook: pd.read_csv)."""
    df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def feature_category(col: str) -> str:
    for prefix, label in CATEGORY_LABELS.items():
        if col.startswith(prefix):
            return label
    return col


def plot_distribution(series: pd.Series, title: str):
    """Tự chọn loại biểu đồ theo kiểu biến. Toàn bộ biến trong notebook là
    số nguyên rời rạc (thang Likert 1-5) hoặc biến mục tiêu nhị phân, nên
    dùng biểu đồ cột theo value_counts."""
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 8:
        fig = px.histogram(series, x=series, nbins=20, title=title)
    else:
        counts = series.value_counts().sort_index().reset_index()
        counts.columns = ["Giá trị", "Số lượng"]
        counts["Giá trị"] = counts["Giá trị"].astype(str)
        fig = px.bar(counts, x="Giá trị", y="Số lượng", title=title, text="Số lượng")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ============================================================
# 2) SIDEBAR — VÙNG CẤU HÌNH (THÀNH PHẦN 1)
# ============================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải lên file dữ liệu (.csv)",
        type=["csv"],
        help="File CSV chứa các biến khảo sát 5C (TC, NL, DK, V, TS) và biến "
        "mục tiêu PD (Probability of Default) — giống dữ liệu mẫu 5c.csv.",
    )

    # Chỉ có 1 mô hình trong notebook (LogisticRegression) -> không cần selectbox chọn mô hình

    st.subheader("Tham số mô hình AI")
    test_size = st.slider(
        "Tỷ lệ tập kiểm tra (test_size)",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Tỷ lệ dữ liệu dành cho kiểm định mô hình. Giá trị mặc định lấy đúng "
        "theo notebook: test_size=0.2.",
    )
    random_state = st.number_input(
        "random_state",
        min_value=0,
        max_value=9999,
        value=23,
        step=1,
        help="Hạt giống ngẫu nhiên để tái lập kết quả chia tập train/test. Giá trị "
        "mặc định lấy đúng theo notebook: random_state=23.",
    )

    with st.expander("Tham số nâng cao (Logistic Regression)"):
        C = st.number_input(
            "C (nghịch đảo độ mạnh regularization)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Notebook không chỉ định tham số này khi khởi tạo mô hình "
            "(LogisticRegression()) — dùng giá trị mặc định của scikit-learn.",
        )
        max_iter = st.number_input(
            "max_iter",
            min_value=50,
            max_value=2000,
            value=100,
            step=50,
            help="Notebook không chỉ định tham số này — dùng giá trị mặc định của "
            "scikit-learn.",
        )
        solver = st.selectbox(
            "solver",
            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            index=0,
            help="Notebook không chỉ định tham số này — dùng giá trị mặc định của "
            "scikit-learn.",
        )

    st.divider()
    train_button = st.button(
        "🚀 Huấn luyện mô hình", type="primary", use_container_width=True
    )

# ============================================================
# 3) HEADER — VÙNG ĐỊNH HƯỚNG (THÀNH PHẦN 2)
# ============================================================
st.title("🏦 Dự báo Rủi ro Tín dụng theo Mô hình 5C")
st.caption(
    "Ứng dụng huấn luyện mô hình Hồi quy Logistic (Logistic Regression) để dự báo "
    "rủi ro tín dụng (PD - Probability of Default) của khách hàng dựa trên 24 biến "
    "khảo sát theo mô hình 5C: Tính cách (TC), Năng lực (NL), Điều kiện (DK), "
    "Vốn (V), Tài sản đảm bảo (TS). Tải lên file CSV cùng cấu trúc với dữ liệu mẫu "
    "để bắt đầu."
)

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên file dữ liệu CSV ở thanh bên trái để bắt đầu.")
    st.stop()

file_bytes = uploaded_file.getvalue()
df = load_data(file_bytes)

missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing_cols:
    st.error(
        "❌ File dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing_cols)
        + ". Vui lòng kiểm tra lại file, cần đầy đủ 24 biến đầu vào (TC1-TS4) và "
        "biến mục tiêu PD."
    )
    st.stop()

if df.empty:
    st.error("❌ File dữ liệu rỗng, không có dòng dữ liệu nào.")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(
    f"🔢 Tóm tắt nhanh: {df.shape[0]} dòng · {df.shape[1]} cột · "
    f"tỷ lệ rủi ro (PD=1): {df[TARGET_COL].mean() * 100:.1f}%"
)
st.divider()

# ============================================================
# 4) KHỐI HUẤN LUYỆN — CHỈ CHẠY KHI BẤM NÚT Ở SIDEBAR
# ============================================================
if train_button:
    with st.spinner("Đang huấn luyện mô hình..."):
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = LogisticRegression(C=C, max_iter=int(max_iter), solver=solver)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        st.session_state["model"] = model
        # Không có scaler/encoder trong notebook (dữ liệu đã ở dạng số) -> bộ tiền
        # xử lý là hàm chọn đúng thứ tự cột X
        st.session_state["preprocessor_cols"] = FEATURE_COLS
        st.session_state["results"] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        st.session_state["train_data_ref"] = df[FEATURE_COLS]

    st.success(
        "✅ Huấn luyện xong! Xem kết quả ở tab 'Kết quả huấn luyện & kiểm định mô hình'."
    )

# ============================================================
# 5) CÁC TAB NỘI DUNG (THÀNH PHẦN 3-6)
# ============================================================
tab_overview, tab_viz, tab_result, tab_predict = st.tabs(
    [
        "📋 Tổng quan dữ liệu",
        "📊 Trực quan hóa dữ liệu",
        "🧪 Kết quả huấn luyện & kiểm định mô hình",
        "🔮 Sử dụng mô hình",
    ]
)

# ---------- THÀNH PHẦN 3: TỔNG QUAN DỮ LIỆU ----------
with tab_overview:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", f"{df.shape[0]:,}")
    c2.metric("Số cột", f"{df.shape[1]:,}")
    c3.metric("Dung lượng file", f"{file_size_mb:.3f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=300):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả (biến của mô hình)")
    st.dataframe(
        df[FEATURE_COLS + [TARGET_COL]].describe().T, use_container_width=True
    )

# ---------- THÀNH PHẦN 4: TRỰC QUAN HÓA DỮ LIỆU ----------
with tab_viz:
    st.caption(
        "Biểu đồ bắt đầu bằng biến mục tiêu PD (mô hình có giám sát), sau đó là các "
        "biến đầu vào được chọn bên dưới."
    )

    default_features = ["TC1", "NL1", "V1"]
    default_features = [f for f in default_features if f in FEATURE_COLS]
    selected_features = st.multiselect(
        "Chọn biến đầu vào muốn trực quan hóa (mặc định 3 biến đại diện các nhóm 5C)",
        options=FEATURE_COLS,
        default=default_features,
        max_selections=3,
        help="Chọn tối đa 3 biến để ghép cùng biến mục tiêu PD thành lưới 2x2.",
    )

    charts = [(TARGET_COL, "Phân phối biến mục tiêu: PD")] + [
        (f, f"{f} — {feature_category(f)}") for f in selected_features
    ]

    row1 = st.columns(2)
    row2 = st.columns(2)
    slots = row1 + row2

    for slot, (col, title) in zip(slots, charts):
        with slot:
            st.plotly_chart(
                plot_distribution(df[col], title),
                use_container_width=True,
            )

    for slot in slots[len(charts):]:
        with slot:
            st.caption("Chọn thêm biến ở trên để hiển thị biểu đồ.")

# ---------- THÀNH PHẦN 5: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH ----------
with tab_result:
    if "results" not in st.session_state:
        st.info(
            "Vui lòng bấm nút '🚀 Huấn luyện mô hình' ở thanh bên trái để xem kết quả."
        )
    else:
        res = st.session_state["results"]
        y_test = res["y_test"]
        y_pred = res["y_pred"]
        y_proba = res["y_proba"]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-score", f"{f1:.3f}")
        m5.metric("ROC-AUC", f"{auc:.3f}" if auc is not None else "N/A")

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=[f"Thực tế: {c}" for c in sorted(y_test.unique())],
                columns=[f"Dự báo: {c}" for c in sorted(y_test.unique())],
            )
            fig_cm = px.imshow(
                cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto"
            )
            fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            if y_proba is not None:
                st.subheader("Đường cong ROC")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})")
                )
                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(dash="dash", color="gray"), name="Ngẫu nhiên",
                    )
                )
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=350,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("Báo cáo phân loại chi tiết (classification report)")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

# ---------- THÀNH PHẦN 6: SỬ DỤNG MÔ HÌNH ----------
with tab_predict:
    if "model" not in st.session_state:
        st.info(
            "Vui lòng bấm nút '🚀 Huấn luyện mô hình' ở thanh bên trái trước khi sử "
            "dụng mô hình để dự báo."
        )
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state["preprocessor_cols"]
        ref_data = st.session_state["train_data_ref"]

        mode = st.radio(
            "Chọn chế độ dự báo",
            options=["Nhập trực tiếp", "Tải file hàng loạt"],
            horizontal=True,
        )

        if mode == "Nhập trực tiếp":
            st.caption(
                "Nhập giá trị cho từng biến khảo sát (thang điểm 1-5) để dự báo rủi "
                "ro tín dụng của một khách hàng."
            )
            with st.form("predict_form"):
                cols_layout = st.columns(4)
                input_values = {}
                for i, col in enumerate(feature_cols):
                    col_min = int(ref_data[col].min())
                    col_max = int(ref_data[col].max())
                    col_median = int(ref_data[col].median())
                    with cols_layout[i % 4]:
                        input_values[col] = st.number_input(
                            col,
                            min_value=col_min,
                            max_value=col_max,
                            value=col_median,
                            step=1,
                            help=f"Nhóm: {feature_category(col)}. Khoảng dữ liệu gốc: "
                            f"[{col_min}, {col_max}].",
                        )
                submitted = st.form_submit_button("Dự báo")

            if submitted:
                x_new = pd.DataFrame([input_values])[feature_cols]
                pred = model.predict(x_new)[0]
                try:
                    proba = model.predict_proba(x_new)[0]
                except Exception:
                    proba = None

                if pred == 1:
                    st.error(f"⚠️ Kết quả dự báo: **CÓ rủi ro tín dụng** (PD = {pred})")
                else:
                    st.success(f"✅ Kết quả dự báo: **KHÔNG có rủi ro tín dụng** (PD = {pred})")

                if proba is not None:
                    p1, p2 = st.columns(2)
                    p1.metric("Xác suất không rủi ro (PD=0)", f"{proba[0] * 100:.1f}%")
                    p2.metric("Xác suất có rủi ro (PD=1)", f"{proba[1] * 100:.1f}%")

        else:
            st.caption(
                "Tải lên file CSV có đúng các cột biến đầu vào (giống x_test trong "
                "notebook) để dự báo hàng loạt."
            )
            batch_file = st.file_uploader(
                "Tải file dữ liệu mới (.csv)", type=["csv"], key="batch_predict_uploader"
            )

            if batch_file is not None:
                new_df = load_data(batch_file.getvalue())
                missing_batch_cols = [c for c in feature_cols if c not in new_df.columns]
                if missing_batch_cols:
                    st.error(
                        "❌ File thiếu các cột bắt buộc: " + ", ".join(missing_batch_cols)
                    )
                else:
                    X_batch = new_df[feature_cols]
                    preds = model.predict(X_batch)
                    try:
                        probas = model.predict_proba(X_batch)[:, 1]
                    except Exception:
                        probas = None

                    result_df = new_df.copy()
                    result_df["PD_du_bao"] = preds
                    if probas is not None:
                        result_df["Xac_suat_rui_ro"] = probas

                    st.subheader("Kết quả dự báo hàng loạt")
                    with st.container(height=350):
                        st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Tải kết quả dự báo (CSV)",
                        data=csv_bytes,
                        file_name="ket_qua_du_bao_PD.csv",
                        mime="text/csv",
                    )

          

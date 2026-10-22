# -*- coding: utf-8 -*-
"""
===================================================================================
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 2: MACHINE LEARNING & XGBOOST
NHIỆM VỤ CV06: TỔNG HỢP PIPELINE, TRẢ LỜI CÂU HỎI LÝ LUẬN & XUẤT BÁO CÁO TỔNG KẾT
===================================================================================

[THÔNG TIN DÀNH CHO THÀNH VIÊN & AI AGENT]:
- Mã công việc: CV06
- Thành viên phụ trách: Trần Thị Thanh Phương
- Nhánh Git khuyến nghị: feat/cv06-report-theory
- Tệp thực thi: src/cv06_report_synthesis.py
- Dữ liệu đầu vào:
  + Tất cả các bảng chỉ số tại: outputs/tables/*.csv
  + Tất cả các hình ảnh trực quan tại: outputs/figures/*.png
- Sản phẩm đầu ra:
  + docs/BAO_CAO_LAB2_NHOM7.md (Báo cáo kỹ thuật toàn diện nộp Giảng viên)
  + outputs/tables/final_model_comparison.csv (Bảng so sánh tổng hợp tất cả các kịch bản)
  + outputs/figures/cv06_summary_dashboard.png (Bảng điều khiển tổng hợp kết quả đồ thị)

[QUY TẮC KỸ THUẬT BẮT BUỘC CHO AI AGENT]:
1. KIỂM TOÁN TÍNH ĐẦY ĐỦ CỦA DỰ ÁN (PROJECT READINESS AUDIT):
   - Tự động kiểm tra sự tồn tại của các artifacts từ CV01 đến CV05 trước khi tổng hợp.
2. TRẢ LỜI ĐẦY ĐỦ VÀ CHẶT CHẼ 3 CÂU HỎI LÝ LUẬN CỦA ĐỀ BÀI:
   - Câu 1: So sánh XGBoost vs Linear Regression trong bài toán chuỗi thời gian:
     + Quan hệ phi tuyến & tương tác bậc cao giữa các lags.
     + Hiện tượng đa cộng tuyến (Multicollinearity).
     + Khả năng ngoại suy xu hướng ngoài miền dữ liệu huấn luyện.
   - Câu 2: Tác động khi tăng số lượng bậc trễ lags p:
     + Lợi ích tiếp nhận tín hiệu dài hạn (chu kỳ ngày 24h, tuần 168h).
     + Rủi ro quá khớp (Overfitting), chiều dữ liệu và mất mát quan sát đầu chuỗi.
   - Câu 3: Khi nào XGBoost phù hợp hơn ARIMA/SARIMA:
     + Chuỗi phi tuyến, cấu trúc mùa vụ đa chu kỳ (Intra-day, Weekly, Annual).
     + Có nhiều biến ngoại sinh và dữ liệu kích thước lớn.
3. TỰ ĐỘNG XUẤT BÁO CÁO MARKDOWN ĐẦY ĐỦ ĐỒ THỊ VÀ BẢNG SỐ LIỆU.
===================================================================================
"""
"""CV06 - Audit pipeline, consolidate results and compile Lab 2 report."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = DOCS_DIR / "BAO_CAO_LAB2_NHOM7.md"
FINAL_TABLE_PATH = TABLES_DIR / "final_model_comparison.csv"
DASHBOARD_PATH = FIGURES_DIR / "cv06_summary_dashboard.png"

EXPECTED_TABLES = [
    "cv01_data_summary.csv",
    "cv02_train_history.csv",
    "cv03_evaluation_metrics.csv",
    "cv03_feature_importance.csv",
    "cv04_multistep_metrics.csv",
    "cv05_best_hyperparameters.csv",
    "cv05_benchmark_summary.csv",
]

EXPECTED_FIGURES = [
    "cv01_raw_and_lags.png",
    "cv02_learning_curve.png",
    "cv03_actual_vs_predicted.png",
    "cv03_feature_importance.png",
    "cv04_rolling_features_impact.png",
    "cv04_multistep_forecast.png",
    "cv05_benchmark_comparison.png",
]


def step1_audit_team_deliverables():
    """Check expected CV01-CV05 artifacts without assuming they exist."""
    found_tables = [
        name for name in EXPECTED_TABLES
        if (TABLES_DIR / name).is_file()
    ]
    missing_tables = [
        name for name in EXPECTED_TABLES
        if not (TABLES_DIR / name).is_file()
    ]
    found_figures = [
        name for name in EXPECTED_FIGURES
        if (FIGURES_DIR / name).is_file()
    ]
    missing_figures = [
        name for name in EXPECTED_FIGURES
        if not (FIGURES_DIR / name).is_file()
    ]

    print(f"[CV06] Tables found: {len(found_tables)}/{len(EXPECTED_TABLES)}")
    print(f"[CV06] Figures found: {len(found_figures)}/{len(EXPECTED_FIGURES)}")

    return {
        "found_tables": found_tables,
        "missing_tables": missing_tables,
        "found_figures": found_figures,
        "missing_figures": missing_figures,
    }


def step2_consolidate_comparison_table():
    """Combine available evaluation and benchmark CSVs, preserving source."""
    source_names = [
        "cv03_evaluation_metrics.csv",
        "cv05_benchmark_summary.csv",
    ]
    frames = []

    for name in source_names:
        path = TABLES_DIR / name
        if not path.is_file():
            print(f"[CV06] Missing input: {name}")
            continue

        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"[CV06] Empty input: {name}")
                continue
            df.insert(0, "source_file", name)
            frames.append(df)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            print(f"[CV06] Cannot read {name}: {exc}")

    if frames:
        final_df = pd.concat(frames, ignore_index=True, sort=False)
    else:
        # Explicitly mark that no actual comparison data was available.
        final_df = pd.DataFrame(columns=["source_file", "status"])
        final_df.loc[0] = [
            "",
            "No evaluation or benchmark CSV available; run upstream tasks first.",
        ]

    final_df.to_csv(FINAL_TABLE_PATH, index=False, encoding="utf-8-sig")
    print(f"[CV06] Wrote comparison table: {FINAL_TABLE_PATH}")
    return final_df


def step3_create_dashboard(audit):
    """Create an artifact-readiness dashboard, with no fabricated metrics."""
    labels = (
        [f"Table: {x}" for x in EXPECTED_TABLES]
        + [f"Figure: {x}" for x in EXPECTED_FIGURES]
    )
    present = (
        [int(x in audit["found_tables"]) for x in EXPECTED_TABLES]
        + [int(x in audit["found_figures"]) for x in EXPECTED_FIGURES]
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(labels))
    ax.barh(y, present)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 1], ["Missing", "Present"])
    ax.set_title("CV06 - Project Readiness Audit")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(DASHBOARD_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[CV06] Wrote dashboard: {DASHBOARD_PATH}")


def _markdown_table(df):
    if df is None or df.empty:
        return "_Chưa có dữ liệu để trình bày._"
    def _markdown_table(df):
    if df is None or df.empty:
        return "_Chưa có dữ liệu để trình bày._"

    df = df.head(30).fillna("")
    columns = [str(c) for c in df.columns]

    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [
        "| " + " | ".join(clean(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for _, row in df.iterrows():
        rows.append(
            "| " + " | ".join(clean(v) for v in row.tolist()) + " |"
        )

    return "\n".join(rows)


def step4_compile_report(audit, final_df):
    """Write the technical report and three theory answers."""
    lines = [
        "# BÁO CÁO KỸ THUẬT LAB 2 - NHÓM 7",
        "",
        "## 1. Mục tiêu",
        "",
        "Tổng hợp kết quả các công đoạn xử lý dữ liệu, huấn luyện, "
        "đánh giá và mở rộng mô hình dự báo chuỗi thời gian.",
        "",
        "## 2. Kiểm toán mức độ sẵn sàng của dự án",
        "",
        f"- Bảng CSV hiện có: {len(audit['found_tables'])}/{len(EXPECTED_TABLES)}.",
        f"- Biểu đồ hiện có: {len(audit['found_figures'])}/{len(EXPECTED_FIGURES)}.",
        "",
        "### 2.1. Bảng đã tìm thấy",
        "",
    ]

    lines.extend([f"- `{x}`" for x in audit["found_tables"]] or ["- Chưa tìm thấy bảng nào."])
    lines += ["", "### 2.2. Bảng còn thiếu", ""]
    lines.extend([f"- `{x}`" for x in audit["missing_tables"]] or ["- Không thiếu bảng nào trong danh sách kiểm tra."])
    lines += ["", "### 2.3. Biểu đồ đã tìm thấy", ""]
    lines.extend([f"- `{x}`" for x in audit["found_figures"]] or ["- Chưa tìm thấy biểu đồ nào."])
    lines += ["", "### 2.4. Biểu đồ còn thiếu", ""]
    lines.extend([f"- `{x}`" for x in audit["missing_figures"]] or ["- Không thiếu biểu đồ nào trong danh sách kiểm tra."])

    lines += [
        "",
        "### 2.5. Dashboard kiểm toán",
        "",
        "![CV06 Project Readiness Dashboard](../outputs/figures/cv06_summary_dashboard.png)",
        "",
        "## 3. Bảng so sánh mô hình",
        "",
        "Bảng dưới đây ghép các hàng từ những file đánh giá và benchmark "
        "đọc được. Cột `source_file` cho biết nguồn gốc của từng hàng. "
        "Do cấu trúc các CSV có thể khác nhau, cần kiểm tra tên cột và "
        "đơn vị trước khi so sánh trực tiếp.",
        "",
        _markdown_table(final_df),
        "",
        "## 4. Trả lời câu hỏi lý luận",
        "",
        "### Câu 1. So sánh XGBoost và Linear Regression",
        "",
        "**Linear Regression** mô hình hóa quan hệ tuyến tính giữa biến "
        "mục tiêu và các đặc trưng trễ. Mô hình đơn giản, dễ diễn giải, "
        "nhưng khó biểu diễn trực tiếp quan hệ phi tuyến nếu không bổ sung "
        "biến biến đổi hoặc tương tác.",
        "",
        "**XGBoost** kết hợp nhiều cây quyết định tăng cường theo gradient, "
        "có thể học quan hệ phi tuyến và tương tác giữa các lag mà không "
        "cần chỉ định trước dạng hàm. Tuy vậy, mô hình có nhiều tham số "
        "hơn và cần kiểm soát overfitting.",
        "",
        "Các lag liên quan thường tương quan mạnh với nhau, gây đa cộng "
        "tuyến trong Linear Regression. Điều này có thể làm hệ số hồi quy "
        "kém ổn định, dù dự báo vẫn có thể tốt. XGBoost không có hệ số "
        "tuyến tính theo cách đó nhưng vẫn có thể bị ảnh hưởng bởi các "
        "đặc trưng dư thừa và dữ liệu ít.",
        "",
        "Cả hai mô hình đều không tự động bảo đảm ngoại suy xu hướng dài "
        "hạn. Linear Regression có thể tiếp tục xu hướng tuyến tính nếu "
        "được xây dựng với biến thời gian, trong khi XGBoost thường dự báo "
        "theo các mẫu đã học và không nên mặc định sẽ ngoại suy tốt ngoài "
        "miền huấn luyện.",
        "",
        "### Câu 2. Tác động khi tăng số lượng lag p",
        "",
        "Tăng số lag giúp mô hình tiếp nhận thông tin lịch sử xa hơn. "
        "Với dữ liệu theo giờ, lag 24 có thể đại diện cho cùng giờ ngày "
        "trước và lag 168 cho cùng giờ tuần trước. Các lag này hữu ích "
        "nếu chuỗi có chu kỳ ngày hoặc tuần.",
        "",
        "Tuy nhiên, thêm lag làm tăng số đặc trưng, chi phí tính toán và "
        "nguy cơ overfitting. Nhiều lag tương quan cũng có thể bổ sung "
        "ít thông tin mới. Việc tạo lag p thường làm mất p quan sát đầu "
        "tiên nếu không có dữ liệu lịch sử trước đó. Vì vậy, nên chọn p "
        "bằng kiểm định theo thời gian và chỉ dùng dữ liệu quá khứ để "
        "tránh rò rỉ dữ liệu.",
        "",
        "### Câu 3. Khi nào nên dùng XGBoost thay vì ARIMA/SARIMA?",
        "",
        "XGBoost phù hợp khi quan hệ giữa mục tiêu và các đặc trưng có "
        "tính phi tuyến, có nhiều biến ngoại sinh như thời tiết hoặc lịch, "
        "có lượng dữ liệu đủ lớn và cần kết hợp nhiều loại đặc trưng trễ, "
        "rolling statistics hay đặc trưng lịch.",
        "",
        "ARIMA/SARIMA là lựa chọn đáng cân nhắc khi cấu trúc tự tương quan "
        "và mùa vụ có thể mô hình hóa bằng các thành phần thống kê, đặc "
        "biệt khi cần mô hình gọn và dễ phân tích. XGBoost không tự hiểu "
        "thứ tự thời gian: người làm cần tạo đặc trưng phù hợp và đánh giá "
        "bằng cách chia tập theo thời gian. Với mùa vụ nhiều chu kỳ, cần "
        "mã hóa chu kỳ và đánh giá thực nghiệm; không thể khẳng định "
        "XGBoost luôn tốt hơn ARIMA/SARIMA.",
        "",
        "## 5. Kết luận và giới hạn",
        "",
        "Báo cáo này ghi nhận tình trạng artifact tại thời điểm chạy script. "
        "Các chỉ số hiệu năng chỉ có thể kết luận sau khi các CSV đánh giá "
        "được tạo từ mô hình thực tế và được kiểm tra về cột, đơn vị, "
        "cách chia dữ liệu và phương pháp đánh giá. Không suy diễn rằng "
        "một mô hình tốt hơn nếu thiếu số liệu kiểm chứng.",
        "",
        "## 6. Tệp đầu ra",
        "",
        "- `outputs/tables/final_model_comparison.csv`",
        "- `outputs/figures/cv06_summary_dashboard.png`",
        "- `docs/BAO_CAO_LAB2_NHOM7.md`",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[CV06] Wrote report: {REPORT_PATH}")
    return REPORT_PATH


def main():
    print("=" * 70)
    print("CV06 - AUDIT, MODEL COMPARISON AND REPORT SYNTHESIS")
    print("=" * 70)

    audit = step1_audit_team_deliverables()
    final_df = step2_consolidate_comparison_table()
    step3_create_dashboard(audit)
    step4_compile_report(audit, final_df)

    complete = not audit["missing_tables"] and not audit["missing_figures"]
    if complete:
        print("[CV06] All expected artifacts are present.")
    else:
        print("[CV06] Report generated, but upstream artifacts are still missing.")
        print("[CV06] Review the report before submission.")


if __name__ == "__main__":
    main()

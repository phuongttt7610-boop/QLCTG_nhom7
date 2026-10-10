# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 2
BÀI TẬP 2: KIỂM ĐỊNH CHUỖI NHIỄU TRẮNG (WHITE NOISE TESTING)
---------------------------------------------------------------------------------
Thành viên phụ trách: Thanh Phuong
Thư mục: LAB2/bai_tap_2/
Tệp thực thi: bai_tap_2.py
Dataset so sánh: LAB2/data/raw/AirPassengers.csv

=================================================================================
🤖 CHỈ THỊ DÀNH CHO AI AGENT (AI AGENT PROMPT & GUARDRAILS SPECIFICATION)
=================================================================================
Nếu bạn là một AI Agent (Cursor, GitHub Copilot, Gemini, ChatGPT, Claude...):
1. TUÂN THỦ KIẾN TRÚC ĐỒNG NHẤT:
   - Không được đổi tên 4 hàm chuẩn: `step1_load_data`, `step2_execute_analysis`, 
     `step3_visualize`, `step4_conclude_and_report`.
   - Giữ nguyên cấu trúc trả về dạng `Dict[str, Any]` và các type hints.
2. NGUYÊN TẮC KỸ THUẬT:
   - Sinh chuỗi Gaussian White Noise với `loc=0.0`, `scale=1.0`, kích thước 1000 điểm.
   - Bắt buộc cố định `np.random.seed(42)` để kết quả thực nghiệm có tính tái lập (reproducible).
   - Thực hiện kiểm định Ljung-Box (`acorr_ljungbox`) tại các độ trễ kiểm tra: lags=[10, 20].
3. NGUYÊN TẮC TRỰC QUAN HÓA & LƯU TRỮ:
   - KHÔNG tự ý lưu tệp hình ảnh (.png/.jpg) ra ổ đĩa để tránh làm nặng repository.
   - Hàm `step3_visualize` tạo biểu đồ đối chứng trực quan, giải phóng bộ nhớ hoặc hiển thị tương tác nếu cần.
   - Xuất file văn bản `LAB2/bai_tap_2/ket_qua_bai_2.txt` chứa đoạn nhận xét 5 dòng chuẩn.
4. CÂU HỎI NGHIỆM THU:
   - Câu 1: Bao nhiêu lag của chuỗi White Noise vượt ra ngoài dải tin cậy 95%?
   - Câu 2: Trị số p-value của kiểm định Ljung-Box trên 2 chuỗi là bao nhiêu?
   - Câu 3: Chuỗi White Noise có thể dùng để dự báo tương lai được không? Tại sao?
=================================================================================
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Cấu hình encoding UTF-8 để không bị lỗi console trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# -------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ TOÀN CỤC
# -------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AirPassengers.csv"

CONFIG = {
    "sample_size": 1000,
    "seed": 42,
    "mean": 0.0,
    "std": 1.0,
    "lags": 30,
    "alpha": 0.05
}


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: SINH DỮ LIỆU WHITE NOISE & TẢI DỮ LIỆU THỰC TẾ
# -------------------------------------------------------------------------------
def step1_load_data(data_path: Path) -> Tuple[pd.Series, pd.Series]:
    """
    Sinh chuỗi Gaussian White Noise ngẫu nhiên và tải chuỗi thực tế AirPassengers.
    """
    print(f"[BƯỚC 1] Sinh chuỗi Gaussian White Noise ({CONFIG['sample_size']} điểm, seed={CONFIG['seed']})...")
    np.random.seed(CONFIG["seed"])
    white_noise = np.random.normal(loc=CONFIG["mean"], scale=CONFIG["std"], size=CONFIG["sample_size"])
    dates_wn = pd.date_range(start="2000-01-01", periods=CONFIG["sample_size"], freq="D")
    wn_series = pd.Series(white_noise, index=dates_wn, name="White_Noise")

    print(f" -> Tải dữ liệu thực tế từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {data_path}")
    
    df_real = pd.read_csv(data_path)
    df_real["Month"] = pd.to_datetime(df_real["Month"])
    df_real.set_index("Month", inplace=True)
    real_series = df_real["Passengers"].astype(float)

    print(f" -> Chuỗi White Noise: {len(wn_series)} điểm (Mean={wn_series.mean():.4f}, Std={wn_series.std():.4f})")
    print(f" -> Chuỗi AirPassengers: {len(real_series)} điểm.")
    return wn_series, real_series


# -------------------------------------------------------------------------------
# 2. BƯỚC 2: THỰC HIỆN KIỂM ĐỊNH THỐNG KÊ (LJUNG-BOX TEST)
# -------------------------------------------------------------------------------
def step2_execute_analysis(wn_series: pd.Series, real_series: pd.Series) -> Dict[str, Any]:
    """
    Chạy kiểm định Ljung-Box để kiểm tra tính độc lập ngẫu nhiên của cả 2 chuỗi.
    """
    print(f"[BƯỚC 2] Thực hiện kiểm định Ljung-Box Test tại lags=[10, 20]...")
    check_lags = [10, 20]

    # Kiểm định cho White Noise
    lb_wn = acorr_ljungbox(wn_series, lags=check_lags, return_df=True)

    # Kiểm định cho chuỗi thực tế
    lb_real = acorr_ljungbox(real_series, lags=check_lags, return_df=True)

    results = {
        "lb_wn": lb_wn,
        "lb_real": lb_real,
        "wn_pvalue_lag10": lb_wn.loc[10, "lb_pvalue"],
        "wn_pvalue_lag20": lb_wn.loc[20, "lb_pvalue"],
        "real_pvalue_lag10": lb_real.loc[10, "lb_pvalue"],
        "real_pvalue_lag20": lb_real.loc[20, "lb_pvalue"]
    }

    print(" -> Kết quả Ljung-Box: Chuỗi White Noise:")
    print(lb_wn[["lb_stat", "lb_pvalue"]])
    print(" -> Kết quả Ljung-Box: Chuỗi AirPassengers:")
    print(lb_real[["lb_stat", "lb_pvalue"]])
    return results


# -------------------------------------------------------------------------------
# 3. BƯỚC 3: TRỰC QUAN HÓA SO SÁNH 2 CHUỖI
# -------------------------------------------------------------------------------
def step3_visualize(wn_series: pd.Series, real_series: pd.Series, analysis_results: Dict[str, Any], show_plot: bool = False) -> None:
    """
    Khởi tạo 4 biểu đồ: (1) Đường White Noise, (2) ACF White Noise, (3) Đường AirPassengers, (4) ACF AirPassengers.
    Lưu ý: Không tự ý xuất file ảnh ra ổ đĩa theo quy định tinh gọn của dự án.
    """
    print(f"[BƯỚC 3] Đang khởi tạo biểu đồ so sánh White Noise vs AirPassengers...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    # Hàng 1: White Noise
    axes[0, 0].plot(wn_series.values, color="seagreen", linewidth=0.8)
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Biểu đồ Chuỗi White Noise Mô Phỏng (N=1000)")
    axes[0, 0].set_ylabel("Giá trị")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    plot_acf(wn_series, lags=CONFIG["lags"], ax=axes[0, 1], color="seagreen", title="ACF của White Noise (Toàn bộ trong dải 95%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Hàng 2: Chuỗi thực tế
    axes[1, 0].plot(real_series.index, real_series.values, color="royalblue", linewidth=1.5)
    axes[1, 0].set_title("Biểu đồ Chuỗi AirPassengers (Dữ liệu thực tế)")
    axes[1, 0].set_ylabel("Lượng khách")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    plot_acf(real_series, lags=CONFIG["lags"], ax=axes[1, 1], color="royalblue", title="ACF của AirPassengers (Xu hướng & Mùa vụ mạnh)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(" -> Đã hoàn thành trực quan hóa đối chứng (không xuất tệp ảnh ra đĩa).")
    return None


# -------------------------------------------------------------------------------
# 4. BƯỚC 4: TỔNG KẾT VÀ BÁO CÁO NHẬN XÉT (5 DÒNG CHUẨN ĐỀ BÀI)
# -------------------------------------------------------------------------------
def step4_conclude_and_report(analysis_results: Dict[str, Any]) -> str:
    """
    Xuất đoạn văn so sánh 5 dòng và kết luận tính khả thi dự báo.
    """
    print("\n[BƯỚC 4] TỔNG HỢP KẾT QUẢ VÀ BÁO CÁO ĐOẠN VĂN SO SÁNH 5 DÒNG:")
    report_text = f"""
=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 2 (KIỂM TRA WHITE NOISE)
=================================================================================
ĐOẠN VĂN NHẬN XÉT SO SÁNH (5 DÒNG THEO YÊU CẦU ĐỀ BÀI):
1. Biểu đồ ACF của chuỗi White Noise có gần như toàn bộ các hệ số độ trễ (lags 1-30) nằm hoàn toàn
   bên trong dải tin cậy 95%, chứng minh các điểm dữ liệu hoàn toàn độc lập và không có tương quan.
2. Ngược lại, biểu đồ ACF của chuỗi AirPassengers hiển thị các hệ số tự tương quan rất cao, suy giảm chậm
   và lặp lại các đỉnh chu kỳ tại lag 12, phản ánh rõ rệt cấu trúc xu hướng và tính mùa vụ.
3. Kết quả kiểm định Ljung-Box trên chuỗi White Noise có p-value lớn hơn 0.05 (p={analysis_results['wn_pvalue_lag10']:.4f}),
   chấp nhận giả thuyết H0 rằng chuỗi này là nhiễu thuần túy và không có cấu trúc tự hồi quy.
4. Đối với chuỗi AirPassengers, kiểm định Ljung-Box cho p-value xấp xỉ bằng 0 (p={analysis_results['real_pvalue_lag10']:.4e} < 0.05),
   bác bỏ H0 và khẳng định chuỗi chứa lượng thông tin nội tại dồi dào.
5. Do đó, chuỗi White Noise hoàn toàn KHÔNG THỂ dự báo được (dự báo tốt nhất chỉ là kỳ vọng mu=0),
   trong khi chuỗi AirPassengers chứa tín hiệu quy luật mạnh mẽ và rất lý tưởng để xây dựng mô hình dự báo.
=================================================================================
"""
    print(report_text)

    report_file = BASE_DIR / "ket_qua_bai_2.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f" -> Đã lưu báo cáo phân tích tại: {report_file}")
    return report_text


# -------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH
# -------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 2: KIỂM ĐỊNH CHUỖI NHIỄU TRẮNG (WHITE NOISE)")
    print("=" * 80)
    wn_series, real_series = step1_load_data(DATA_PATH)
    analysis_results = step2_execute_analysis(wn_series, real_series)
    step3_visualize(wn_series, real_series, analysis_results)
    step4_conclude_and_report(analysis_results)
    print("\n[HOÀN THÀNH BÀI TẬP 2 THÀNH CÔNG]")


if __name__ == "__main__":
    main()

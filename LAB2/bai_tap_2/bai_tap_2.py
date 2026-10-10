# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 2
BÀI TẬP 2: KIỂM ĐỊNH CHUỖI NHIỄU TRẮNG (WHITE NOISE TESTING)
---------------------------------------------------------------------------------
Thành viên phụ trách: Thành viên 2
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
from statsmodels.tsa.stattools import acf
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
    real_series = df_real["Passengers"].astype(float).sort_index().asfreq("MS")
    if real_series.isna().any():
        raise ValueError("Dữ liệu AirPassengers bị thiếu tháng hoặc có giá trị NaN")

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

    # Đếm lag 1..30 vượt dải tham chiếu 95% ±1.96/sqrt(N), không tính lag 0.
    threshold = 1.96 / np.sqrt(len(wn_series))
    acf_wn = acf(wn_series, nlags=CONFIG["lags"], fft=True)
    outside_lags = np.flatnonzero(np.abs(acf_wn[1:]) > threshold).tolist()
    outside_lags = [lag + 1 for lag in outside_lags]

    # Kiểm định cho White Noise
    lb_wn = acorr_ljungbox(wn_series, lags=check_lags, return_df=True)

    # Kiểm định cho chuỗi thực tế
    lb_real = acorr_ljungbox(real_series, lags=check_lags, return_df=True)

    results = {
        "lb_wn": lb_wn,
        "acf_wn": acf_wn,
        "threshold": float(threshold),
        "outside_lags": outside_lags,
        "lb_real": lb_real,
        "wn_pvalue_lag10": lb_wn.loc[10, "lb_pvalue"],
        "wn_pvalue_lag20": lb_wn.loc[20, "lb_pvalue"],
        "real_pvalue_lag10": lb_real.loc[10, "lb_pvalue"],
        "real_pvalue_lag20": lb_real.loc[20, "lb_pvalue"]
    }

    print(f" -> ACF: {len(outside_lags)}/{CONFIG['lags']} lag vượt ±{threshold:.4f}: {outside_lags}")
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

    # Dùng cùng ngưỡng với phép đếm lag để đồ thị và báo cáo nhất quán.
    plot_acf(wn_series, lags=CONFIG["lags"], alpha=None, ax=axes[0, 1], color="seagreen",
             title=f"ACF White Noise ({len(analysis_results['outside_lags'])} lag vượt dải 95%)")
    ci = analysis_results["threshold"]
    axes[0, 1].axhspan(-ci, ci, color="seagreen", alpha=0.15, label="Dải tham chiếu 95%")
    axes[0, 1].axhline(ci, color="seagreen", linestyle="--", linewidth=0.9)
    axes[0, 1].axhline(-ci, color="seagreen", linestyle="--", linewidth=0.9)
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
    # Đoạn nhận xét chính giữ đúng 5 ý; câu hỏi và bảng số liệu nằm ở mục riêng.
    lag_list = ", ".join(map(str, analysis_results["outside_lags"])) or "không có"
    p10_wn = float(analysis_results["wn_pvalue_lag10"])
    p20_wn = float(analysis_results["wn_pvalue_lag20"])
    p10_air = float(analysis_results["real_pvalue_lag10"])
    p20_air = float(analysis_results["real_pvalue_lag20"])
    count = len(analysis_results["outside_lags"])
    report_text = f"""=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 2 (KIỂM ĐỊNH WHITE NOISE)
=================================================================================
I. ĐOẠN NHẬN XÉT SO SÁNH (ĐÚNG 5 Ý THEO PHÂN CÔNG)
1. Chuỗi White Noise dao động ngẫu nhiên quanh trung bình 0, không có xu hướng hoặc mùa vụ rõ rệt.
2. ACF White Noise có {count}/30 lag (1-30) vượt dải tham chiếu 95% ±{analysis_results['threshold']:.4f}, tại lag: {lag_list}; một vài lag vượt ngưỡng là có thể xảy ra do ngẫu nhiên.
3. ACF AirPassengers có tương quan dương mạnh, giảm chậm và dấu hiệu mùa vụ chu kỳ 12 tháng; khác rõ so với White Noise.
4. Ljung-Box cho White Noise: p(10)={p10_wn:.6f}, p(20)={p20_wn:.6f}, đều > 0.05 nên chưa đủ bằng chứng bác bỏ H0 (không có tự tương quan tới lag kiểm định).
5. Ljung-Box cho AirPassengers: p(10)={p10_air:.6e}, p(20)={p20_air:.6e}, đều < 0.05 nên bác bỏ H0; chuỗi có cấu trúc thời gian hữu ích để xem xét dự báo.

II. BẢNG KẾT QUẢ KIỂM ĐỊNH LJUNG-BOX (alpha=0.05)
Chuỗi                    Lag       LB Statistic         p-value          Kết luận
White Noise               10       {analysis_results['lb_wn'].loc[10, 'lb_stat']:>12.6f}       {p10_wn:.6f}          Không bác bỏ H0
White Noise               20       {analysis_results['lb_wn'].loc[20, 'lb_stat']:>12.6f}       {p20_wn:.6f}          Không bác bỏ H0
AirPassengers             10       {analysis_results['lb_real'].loc[10, 'lb_stat']:>12.6f}       {p10_air:.6e}     Bác bỏ H0
AirPassengers             20       {analysis_results['lb_real'].loc[20, 'lb_stat']:>12.6f}       {p20_air:.6e}     Bác bỏ H0
Ghi chú: H0 là không có tự tương quan tại các độ trễ từ 1 đến lag kiểm định.

III. TRẢ LỜI 3 CÂU HỎI NGHIỆM THU GIỮ NGUYÊN TỪ ĐỀ
Câu 1: Bao nhiêu lag của chuỗi White Noise vượt ra ngoài dải tin cậy 95%?
Trả lời: {count}/30 lag; lag vượt ngưỡng: {lag_list} (không tính lag 0).
Câu 2: Trị số p-value của kiểm định Ljung-Box trên 2 chuỗi là bao nhiêu?
Trả lời: White Noise p(10)={p10_wn:.6f}, p(20)={p20_wn:.6f}; AirPassengers p(10)={p10_air:.6e}, p(20)={p20_air:.6e}.
Câu 3: Chuỗi White Noise có thể dùng để dự báo tương lai được không? Tại sao?
Trả lời: Không dự báo được biến động riêng lẻ tốt hơn dự báo theo kỳ vọng mu=0
nếu chỉ dùng lịch sử chuỗi, vì các giá trị Gaussian White Noise được sinh độc lập.

IV. CÂU HỎI LÝ THUYẾT LIÊN QUAN (3, 4, 11)
Câu 3: White Noise là chuỗi có kỳ vọng và phương sai ổn định, tự tương quan bằng 0 ở các lag dương;
ACF và Ljung-Box giúp kiểm tra dấu hiệu tự tương quan, nhưng không chứng minh được tính độc lập hoàn toàn.
Câu 4: White Noise không có tín hiệu trong lịch sử để cải thiện dự báo giá trị tương lai;
với Gaussian White Noise N(0,1), dự báo theo sai số bình phương tối ưu là 0.
Câu 11: Nếu residual gần White Noise, mô hình có thể đã giải thích phần lớn cấu trúc tự tương quan;
đây là dấu hiệu tốt, không tự động chứng minh mô hình tối ưu hay phần dư độc lập hoàn toàn.
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

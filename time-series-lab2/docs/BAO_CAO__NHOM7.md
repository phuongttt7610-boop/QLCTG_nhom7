# BÁO CÁO TỔNG KẾT LAB  – NHÓM 7

## CV06 – Tổng hợp kết quả và đánh giá mô hình

## 1. Mục tiêu

CV06 thực hiện tổng hợp kết quả từ các công việc trước đó trong bài Lab 3, kiểm tra tính nhất quán của pipeline và xây dựng báo cáo kỹ thuật cuối cùng. Nội dung tập trung vào việc so sánh các phương pháp dự báo chuỗi thời gian, đánh giá ảnh hưởng của số lượng độ trễ (lags) và xác định trường hợp phù hợp để sử dụng XGBoost hoặc ARIMA.

## 2. Trả lời câu hỏi lý thuyết

### 2.1. XGBoost khác Linear Regression như thế nào trong dự báo chuỗi thời gian?

Linear Regression mô hình hóa mối quan hệ tuyến tính giữa các biến đầu vào và biến mục tiêu. Khi sử dụng dữ liệu chuỗi thời gian, các giá trị trễ (lags) có thể được đưa vào làm biến độc lập để dự báo giá trị tương lai. Ưu điểm của phương pháp này là dễ triển khai, dễ giải thích và có chi phí tính toán thấp. Tuy nhiên, mô hình có thể không mô tả tốt các quan hệ phi tuyến phức tạp.

XGBoost là thuật toán học máy dựa trên việc kết hợp nhiều cây quyết định theo phương pháp boosting. Mô hình có khả năng học các quan hệ phi tuyến và tương tác giữa nhiều biến đầu vào. Khi được cung cấp các đặc trưng phù hợp như lags, đặc trưng mùa vụ và đặc trưng lịch, XGBoost có thể khai thác những mẫu dữ liệu phức tạp hơn Linear Regression.

Tuy nhiên, XGBoost không mặc nhiên tốt hơn Linear Regression. Hiệu quả thực tế phụ thuộc vào dữ liệu, cách tạo đặc trưng, tham số mô hình và phương pháp đánh giá. Vì vậy, cần so sánh hai mô hình trên cùng tập kiểm tra và cùng thước đo sai số trước khi kết luận.

### 2.2. Tăng số lượng lags ảnh hưởng như thế nào đến chất lượng dự báo?

Lags là các giá trị quan sát trong quá khứ được sử dụng làm đầu vào để dự báo hiện tại hoặc tương lai. Ví dụ, lag 1 là giá trị ở thời điểm ngay trước đó, còn lag 12 có thể đại diện cho giá trị của 12 kỳ trước đối với dữ liệu theo tháng.

Tăng số lượng lags giúp mô hình tiếp cận nhiều thông tin lịch sử hơn. Điều này có thể cải thiện dự báo nếu chuỗi dữ liệu có tính tự tương quan hoặc chu kỳ dài. Tuy nhiên, việc thêm quá nhiều lags cũng làm tăng số lượng đặc trưng, tăng chi phí tính toán và có thể khiến mô hình học cả nhiễu trong dữ liệu.

Do đó, không nên mặc định rằng càng nhiều lags thì kết quả càng tốt. Cần thử nghiệm các cấu hình khác nhau, giữ nguyên cách chia dữ liệu theo thời gian và so sánh các chỉ số MAE, RMSE hoặc MAPE nếu phù hợp. Cấu hình tốt nhất là cấu hình cho kết quả đánh giá ngoài mẫu tốt và ổn định, không chỉ là cấu hình có nhiều lags nhất.

### 2.3. Khi nào nên sử dụng XGBoost thay vì ARIMA?

ARIMA là mô hình thống kê dự báo chuỗi thời gian dựa trên các thành phần tự hồi quy, sai phân và trung bình trượt. Mô hình phù hợp khi cấu trúc chuỗi có thể được mô tả tương đối tốt bằng các quan hệ thống kê theo thời gian và không cần quá nhiều biến giải thích bên ngoài.

XGBoost phù hợp để cân nhắc khi dữ liệu có các quan hệ phi tuyến, có nhiều biến đầu vào hoặc có những tương tác phức tạp mà mô hình ARIMA cơ bản khó thể hiện. Tuy nhiên, XGBoost thường cần xây dựng đặc trưng đầu vào, chẳng hạn lags và đặc trưng mùa vụ, đồng thời phải tránh sử dụng thông tin từ tương lai.

Việc lựa chọn giữa hai mô hình cần dựa trên kết quả kiểm tra thực tế. Nếu ARIMA có sai số thấp và ổn định thì không nhất thiết phải thay thế bằng XGBoost. Nếu XGBoost cải thiện đáng kể các chỉ số đánh giá trên tập kiểm tra và quy trình thử nghiệm được thực hiện công bằng, mô hình này có thể là lựa chọn phù hợp hơn. Trong một số trường hợp, có thể xem xét kết hợp nhiều phương pháp để tận dụng ưu điểm của từng mô hình.

## 3. Kiểm toán pipeline

Việc kiểm toán pipeline nhằm đảm bảo quá trình xử lý dữ liệu, huấn luyện, đánh giá và tổng hợp kết quả được thực hiện nhất quán.

Các nội dung cần kiểm tra gồm:

1. **Dữ liệu đầu vào:** xác định đúng tệp dữ liệu, kiểm tra giá trị thiếu, kiểu dữ liệu và thứ tự thời gian.
2. **Tiền xử lý:** bảo đảm các bước xử lý dữ liệu được thực hiện nhất quán giữa các mô hình.
3. **Chia tập dữ liệu:** ưu tiên chia theo thứ tự thời gian, không xáo trộn ngẫu nhiên làm rò rỉ thông tin tương lai vào tập huấn luyện.
4. **Tạo đặc trưng lags:** chỉ sử dụng dữ liệu có sẵn tại thời điểm dự báo; kiểm tra việc dịch chuyển dữ liệu để tránh dùng nhầm giá trị mục tiêu tương lai.
5. **Huấn luyện và đánh giá:** các mô hình cần được đánh giá trên cùng giai đoạn kiểm tra và bằng cùng định nghĩa chỉ số sai số.
6. **Tổng hợp kết quả:** kiểm tra các tệp kết quả do CV03, CV04 và CV05 tạo ra, thống nhất tên cột và định dạng trước khi ghép dữ liệu.
7. **Tái lập kết quả:** ghi rõ đường dẫn đầu vào, đầu ra và cách chạy chương trình để thành viên khác có thể kiểm tra lại.

Nếu thiếu tệp kết quả từ một công đoạn, cần ghi nhận trạng thái thiếu và bổ sung sau khi chạy lại công đoạn đó. Không nên tự điền số liệu hoặc đưa ra kết luận mô hình tốt nhất khi chưa có dữ liệu đánh giá đầy đủ.

## 4. Hợp nhất số liệu và xuất báo cáo

CV06 có nhiệm vụ thu thập kết quả từ các công đoạn trước, chuẩn hóa dữ liệu và hợp nhất vào tệp:

`outputs/tables/final_model_comparison.csv`

Tệp này dùng để lưu kết quả so sánh mô hình theo cấu trúc thống nhất. Các trường thông tin cụ thể cần phù hợp với dữ liệu thực tế mà các công đoạn trước xuất ra, chẳng hạn tên mô hình, cấu hình lags và các chỉ số sai số nếu có.

Bên cạnh bảng số liệu, CV06 tạo báo cáo Markdown tại:

`docs/BAO_CAO_LAB3_NHOM7.md`

Báo cáo trình bày mục tiêu, cơ sở lý thuyết, quy trình kiểm toán và nhận xét dựa trên kết quả thực tế. Nếu có dữ liệu và biểu đồ phù hợp, có thể sử dụng thêm hình tổng hợp tại:

`outputs/figures/cv06_summary_dashboard.png`

Các đường dẫn trên là vị trí dự kiến của sản phẩm CV06; cần kiểm tra lại cấu trúc thư mục thực tế của repository trước khi xác nhận chương trình đã tạo đầy đủ các tệp.

## 5. Đánh giá và kết luận

Qua nhiệm vụ CV06, nhóm hướng đến việc xây dựng một quy trình tổng hợp kết quả có thể kiểm tra và tái lập. Ba câu hỏi lý thuyết cho thấy việc lựa chọn mô hình dự báo cần dựa trên đặc điểm dữ liệu, cách xây dựng đặc trưng và kết quả đánh giá ngoài mẫu.

Linear Regression có ưu điểm đơn giản và dễ giải thích; XGBoost có khả năng học các quan hệ phi tuyến; còn ARIMA là phương pháp thống kê chuyên biệt cho chuỗi thời gian. Không có mô hình nào luôn tốt nhất trong mọi trường hợp. Việc lựa chọn cần dựa trên các chỉ số đánh giá được tính toán nhất quán trên cùng tập kiểm tra.

Đối với phần triển khai, việc hợp nhất kết quả chỉ được xem là hoàn tất khi các tệp đầu ra đã được tạo thành công, số liệu có nguồn gốc rõ ràng và báo cáo phản ánh đúng kết quả thực nghiệm. Những công đoạn còn thiếu cần được bổ sung trước khi đưa ra kết luận cuối cùng về hiệu quả của các mô hình.

## 6. Danh sách kiểm tra trước khi nộp

- [ ] Đã kiểm tra kết quả đầu ra từ CV03, CV04 và CV05.
- [ ] Đã tạo và kiểm tra `final_model_comparison.csv`.
- [ ] Đã chạy chương trình CV06 và xác nhận báo cáo Markdown được tạo đúng.
- [ ] Đã kiểm tra hình tổng hợp nếu pipeline có tạo hình.
- [ ] Các số liệu và nhận xét trong báo cáo khớp với kết quả thực nghiệm.
- [ ] Đã kiểm tra các thay đổi trên GitHub và gửi bài theo quy trình của nhóm.

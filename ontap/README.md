1. Kernel Linear:
+ Đơn giản và nhanh nhất trong huấn luyện
+ Phù hợp với dữ liệu có thể tách biệt tuyến tính
+ Thường cho độ chính xác khá tốt trên bộ dữ liệu Digits
2. Kernel Polynomial:
+ Độ chính xác thường nằm giữa linear và RBF
+ Thời gian huấn luyện có thể khá lâu
+ Dễ bị overfitting nếu bậc đa thức cao
3. Kernel RBF (Radial Basis Function):
+ Thường cho độ chính xác cao nhất
+ Thời gian huấn luyện lâu hơn kernel tuyến tính
+ Có khả năng tổng quát hóa tốt
4. Kết luận
+ Nếu muốn độ chính xác cao nhất thì sử dụng kernel RBF
+ Nếu muốn tốc độ huấn luyện nhanh thì sử dụng kernel linear
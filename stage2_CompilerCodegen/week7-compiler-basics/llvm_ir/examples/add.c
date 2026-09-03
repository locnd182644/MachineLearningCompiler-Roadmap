// Hàm đơn giản: 2 tham số, trả về tổng
// Mục tiêu: xem LLVM IR cơ bản nhất
int add(int a, int b) {
    int result = a + b;
    return result;
}

// Hàm phức tạp hơn: constant folding observable
int add_with_const(int x) {
    int a = 10;
    int b = 20;
    int c = a + b;  // Should be folded to 30 at -O2
    return x + c;
}

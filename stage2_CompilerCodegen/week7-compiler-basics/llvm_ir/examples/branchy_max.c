// Max đơn giản — branch thường bị chuyển thành select tại -O2
int max(int a, int b) {
    if (a > b)
        return a;
    else
        return b;
}

// Clamp function — nhiều branches
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// Absolute value
int abs_val(int x) {
    if (x < 0)
        return -x;
    return x;
}

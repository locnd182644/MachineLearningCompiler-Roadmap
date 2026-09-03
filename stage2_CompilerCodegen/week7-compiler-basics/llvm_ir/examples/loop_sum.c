// Sum mảng — để xem loop optimization
// -O0: loop đầy đủ với alloca, load, store mỗi iteration
// -O2: có thể vectorize, loop invariant code motion, strength reduction
int loop_sum(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Nested loop — reduction pattern rõ ràng hơn
int dot_product(int* a, int* b, int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

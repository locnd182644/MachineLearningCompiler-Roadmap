# Build Notes: LLVM/MLIR từ Source

> Log quá trình build + troubleshooting cho Bài tập 8.1

## 1. Prerequisites

### Hardware tối thiểu
- RAM: 16GB (khuyến nghị 32GB)
- Disk: ~30GB cho Release build, ~100GB cho Debug build
- CPU: multi-core (build time tỉ lệ nghịch với số cores)
- Thời gian build: ~30 phút (Release, 8 cores) đến 2+ giờ (Debug, 4 cores)

### Software cần có
```bash
# Ubuntu/Debian
sudo apt-get install -y \
  cmake ninja-build gcc g++ python3 \
  git lld ccache

# Fedora/RHEL
sudo dnf install -y \
  cmake ninja-build gcc gcc-c++ python3 \
  git lld ccache

# macOS
brew install cmake ninja ccache
```

- CMake ≥ 3.20
- Ninja (khuyến nghị, nhanh hơn Make nhiều)
- GCC ≥ 7.4 hoặc Clang ≥ 5.0
- Python ≥ 3.6 (cho lit test runner)
- ccache (QUAN TRỌNG: tiết kiệm thời gian rebuild rất nhiều)
- lld (linker nhanh hơn ld/gold rất nhiều — giảm link time từ phút xuống giây)

## 2. Clone Repository

```bash
# Clone (shallow để nhanh, đủ cho build)
git clone --depth 1 https://github.com/llvm/llvm-project.git

# Hoặc full history nếu muốn đọc git log
git clone https://github.com/llvm/llvm-project.git

# Kiểm tra commit
cd llvm-project
git log -1 --oneline
```

Note: repo rất lớn (~2GB shallow, 5+ GB full). Clone ở đâu có mạng nhanh.

## 3. CMake Configure

### Config chuẩn cho tuần 8 (Release + Assertions)

```bash
cd llvm-project
mkdir -p build && cd build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install
```

### Giải thích từng flag:

| Flag | Ý nghĩa | Tại sao |
|------|---------|--------|
| `-G Ninja` | Dùng Ninja build system | Nhanh hơn Make, parallel tốt hơn |
| `-DLLVM_ENABLE_PROJECTS=mlir` | Chỉ build MLIR (+ LLVM core) | Không cần clang, libc++, etc |
| `-DLLVM_BUILD_EXAMPLES=ON` | Build examples (Toy tutorial) | Cần cho tuần 9 |
| `-DLLVM_TARGETS_TO_BUILD="Native"` | Chỉ build backend cho CPU hiện tại | Giảm build time, ko cần ARM/RISC-V |
| `-DCMAKE_BUILD_TYPE=Release` | Tối ưu hóa, không debug info | Nhanh + nhỏ (30GB vs 100GB) |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | Giữ assert checks dù Release | BẮT BUỘC cho development — bắt bugs sớm |
| `-DLLVM_CCACHE_BUILD=ON` | Cache compiled objects | Rebuild nhanh hơn 10x+ |
| `-DLLVM_USE_LINKER=lld` | Dùng lld thay GNU ld | Link nhanh hơn nhiều |

### Config cho máy ít RAM (< 16GB)

```bash
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_PARALLEL_LINK_JOBS=2 \
  -DLLVM_PARALLEL_COMPILE_JOBS=4
```

Lưu ý: linking là phase tốn RAM nhất. `-DLLVM_PARALLEL_LINK_JOBS=2` giới hạn song song link.

## 4. Build

```bash
# Build toàn bộ MLIR
ninja -j$(nproc)

# Hoặc giới hạn jobs nếu RAM ít
ninja -j4

# Chỉ build mlir-opt (nhanh hơn, đủ cho bài tập 8.2)
ninja mlir-opt

# Build + chạy test suite
ninja check-mlir
```

## 5. Verify Installation

```bash
# Thêm vào PATH
export PATH=$PWD/bin:$PATH

# Verify
mlir-opt --version
mlir-opt --help | head -20

# List các pass có sẵn
mlir-opt --help | grep -c '\-\-'  # đếm số passes

# Test nhanh: parse 1 file MLIR
echo 'func.func @test() { return }' | mlir-opt
```

## 6. Thêm vào shell profile

```bash
# Thêm vào ~/.bashrc hoặc ~/.zshrc
export LLVM_BUILD_DIR=$HOME/path/to/llvm-project/build
export PATH=$LLVM_BUILD_DIR/bin:$PATH

# Verify sau khi source lại
source ~/.bashrc
which mlir-opt
mlir-opt --version
```

## 7. Troubleshooting

### Lỗi thường gặp

#### OOM (Out of Memory) khi link
```
Error: collect2: fatal error: ld terminated with signal 9 [Killed]
```
Giải pháp:
- Thêm `-DLLVM_PARALLEL_LINK_JOBS=1`
- Dùng `-DLLVM_USE_LINKER=lld` (tốn ít RAM hơn gold/ld)
- Thêm swap: `sudo fallocate -l 8G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`

#### ccache not found
```
CMake Warning: LLVM_CCACHE_BUILD is ON but ccache is not found
```
Giải pháp: `sudo apt install ccache` hoặc bỏ flag `-DLLVM_CCACHE_BUILD=ON`

#### Ninja version too old
```
CMake Error: CMake was unable to find a build program corresponding to "Ninja"
```
Giải pháp: `pip3 install ninja` hoặc download từ https://ninja-build.org/

#### check-mlir failures
Một vài test có thể fail do environment (missing libs, etc). Nếu >95% pass, OK để tiếp tục. Báo cáo specific failures trên LLVM Discourse.

## 8. Build Log Template

```
Date: ____
Commit: ____
OS: ____
CPU: ____ cores
RAM: ____ GB
Disk used: ____ GB

Configure time: ____ seconds
Build time: ____ minutes  
check-mlir: ____/____  tests pass

Issues encountered:
- ____
- ____

Resolutions:
- ____
```

## 9. Build targets hữu ích

```bash
# Chỉ build tools cần cho bài tập
ninja mlir-opt           # CLI tool chính
ninja mlir-cpu-runner    # Chạy MLIR trên CPU
ninja mlir-translate     # MLIR ↔ LLVM IR
ninja toyc-ch1           # Toy tutorial chapter 1 (tuần 9)
ninja check-mlir         # Chạy toàn bộ MLIR test suite

# List tất cả targets
ninja -t targets | grep mlir | head -30
```

## 10. Tips cho việc học

- **Không cần build Debug**: Release + Assertions đủ cho bài tập. Debug chỉ cần khi step through MLIR source bằng gdb/lldb.
- **ccache là bạn**: sau lần build đầu, mọi rebuild nhanh rất nhiều.
- **Build incremental**: sau khi sửa 1 file, `ninja mlir-opt` chỉ rebuild files liên quan.
- **Đọc source code**: sau khi build, dùng IDE mở `llvm-project/` — code search rất giá trị.
- **mlir-opt --help**: dành 10 phút đọc — list tất cả pass có sẵn, là bản đồ của MLIR.

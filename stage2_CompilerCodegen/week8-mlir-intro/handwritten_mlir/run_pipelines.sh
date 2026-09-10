#!/bin/bash
# =============================================================================
# Bài tập 8.2 — Pass Pipeline Explorer
#
# Script này chạy nhiều pass pipelines khác nhau trên các file MLIR,
# lưu output để so sánh IR trước/sau mỗi pass.
#
# Cách dùng: ./run_pipelines.sh [path/to/mlir-opt]
#
# Mục tiêu:
# - Thấy IR biến đổi qua từng pass
# - Hiểu thứ tự pass quan trọng
# - Làm quen với --mlir-print-ir-after-all
# =============================================================================

set -euo pipefail

MLIR_OPT="${1:-mlir-opt}"
MLIR_RUNNER="${MLIR_RUNNER:-mlir-cpu-runner}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/pipeline_outputs"

mkdir -p "$OUTPUT_DIR"

echo "================================================================="
echo "  MLIR Pass Pipeline Explorer"
echo "  Tool: $MLIR_OPT"
echo "  Output: $OUTPUT_DIR/"
echo "================================================================="
echo

# ─────────────────────────────────────────────────────
# 1. Arith basic: verify parse
# ─────────────────────────────────────────────────────
echo ">>> [1] Parsing 01_arith_basic.mlir"
$MLIR_OPT "$SCRIPT_DIR/01_arith_basic.mlir" -o "$OUTPUT_DIR/01_parsed.mlir"
echo "    ✓ Parsed and verified"
echo

# ─────────────────────────────────────────────────────
# 2. Canonicalize arith — constant folding!
# ─────────────────────────────────────────────────────
echo ">>> [2] Canonicalize 01_arith_basic.mlir"
$MLIR_OPT "$SCRIPT_DIR/01_arith_basic.mlir" --canonicalize \
  -o "$OUTPUT_DIR/01_canonicalized.mlir"
echo "    ✓ Saved canonicalized output"
echo "    ➤ So sánh: diff $SCRIPT_DIR/01_arith_basic.mlir $OUTPUT_DIR/01_canonicalized.mlir"
echo "    ➤ Chú ý: @with_constants() nên được fold thành return 11!"
echo

# ─────────────────────────────────────────────────────
# 3. Canonicalize + CSE (Common Subexpression Elimination)
# ─────────────────────────────────────────────────────
echo ">>> [3] Canonicalize + CSE 01_arith_basic.mlir"
$MLIR_OPT "$SCRIPT_DIR/01_arith_basic.mlir" --canonicalize --cse \
  -o "$OUTPUT_DIR/01_canon_cse.mlir"
echo "    ✓ Canonicalize + CSE"
echo

# ─────────────────────────────────────────────────────
# 4. SCF loop: structured → unstructured CF
# ─────────────────────────────────────────────────────
echo ">>> [4] Convert scf → cf (02_scf_loop.mlir)"
$MLIR_OPT "$SCRIPT_DIR/02_scf_loop.mlir" --convert-scf-to-cf \
  -o "$OUTPUT_DIR/02_cf.mlir"
echo "    ✓ scf.for → cf.br + cf.cond_br + block arguments"
echo "    ➤ Mở $OUTPUT_DIR/02_cf.mlir — thấy scf.for biến mất, thay bằng blocks"
echo

# ─────────────────────────────────────────────────────
# 5. SCF loop: full pipeline xuống LLVM dialect
# ─────────────────────────────────────────────────────
echo ">>> [5] Full lowering 02_scf_loop.mlir → LLVM dialect"
$MLIR_OPT "$SCRIPT_DIR/02_scf_loop.mlir" \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o "$OUTPUT_DIR/02_llvm.mlir"
echo "    ✓ Mọi op đã là llvm dialect"
echo "    ➤ So sánh $SCRIPT_DIR/02_scf_loop.mlir (scf/arith) với $OUTPUT_DIR/02_llvm.mlir (llvm)"
echo

# ─────────────────────────────────────────────────────
# 6. IR after ALL passes (debug mode)
# ─────────────────────────────────────────────────────
echo ">>> [6] IR sau từng pass (--mlir-print-ir-after-all)"
$MLIR_OPT "$SCRIPT_DIR/02_scf_loop.mlir" \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  --mlir-print-ir-after-all \
  2>"$OUTPUT_DIR/02_ir_after_all.log" \
  -o /dev/null || true
echo "    ✓ Saved: $OUTPUT_DIR/02_ir_after_all.log"
echo "    ➤ Đọc file này để thấy IR biến đổi qua từng pass"
echo "    ➤ Tìm: 'IR Dump After ...' headers"
echo

# ─────────────────────────────────────────────────────
# 7. Dot product: lower
# ─────────────────────────────────────────────────────
echo ">>> [7] Dot product: lower → LLVM dialect"
$MLIR_OPT "$SCRIPT_DIR/03_dot_product.mlir" \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o "$OUTPUT_DIR/03_lowered.mlir"
echo "    ✓ Lowered to LLVM dialect"
echo

# ─────────────────────────────────────────────────────
# 8. Dot product: IR dump
# ─────────────────────────────────────────────────────
echo ">>> [8] Dot product: IR sau từng pass"
$MLIR_OPT "$SCRIPT_DIR/03_dot_product.mlir" \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  --mlir-print-ir-after-all \
  2>"$OUTPUT_DIR/03_ir_after_all.log" \
  -o /dev/null || true
echo "    ✓ Saved: $OUTPUT_DIR/03_ir_after_all.log"
echo

# ─────────────────────────────────────────────────────
# 9. Dot product: chạy (nếu mlir-cpu-runner có)
# ─────────────────────────────────────────────────────
echo ">>> [9] Dot product: chạy qua mlir-cpu-runner"
if command -v "$MLIR_RUNNER" &>/dev/null; then
  $MLIR_OPT "$SCRIPT_DIR/03_dot_product.mlir" \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --finalize-memref-to-llvm \
    --convert-cf-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
  | $MLIR_RUNNER -e main -entry-point-result=void \
      --shared-libs=libmlir_runner_utils.so,libmlir_c_runner_utils.so \
  2>&1 | tee "$OUTPUT_DIR/03_run_output.txt"
  echo "    ✓ Expected: 30.0 (= 1*1 + 2*2 + 3*3 + 4*4)"
else
  echo "    ⚠ mlir-cpu-runner not found — skip execution"
  echo "    ➤ Add LLVM build/bin to PATH, then re-run"
fi
echo

# ─────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────
echo "================================================================="
echo "  Done! Output files:"
echo "================================================================="
ls -la "$OUTPUT_DIR/"
echo
echo "Bước tiếp theo:"
echo "  1. So sánh 01_arith_basic.mlir với 01_canonicalized.mlir (constant folding)"
echo "  2. Đọc 02_ir_after_all.log (IR sau từng pass)"
echo "  3. So sánh 02_scf_loop.mlir với 02_cf.mlir (structured → unstructured)"
echo "  4. So sánh 02_scf_loop.mlir với 02_llvm.mlir (high-level → low-level)"
echo "  5. Nếu có mlir-cpu-runner: verify dot product = 30.0"

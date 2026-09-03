"""
Optimization Passes cho ToyCalc Compiler
==========================================

Pass là gì?
    Một pass là một bước biến đổi (transformation) hoặc phân tích (analysis)
    chạy trên IR. Mỗi pass đọc IR, thực hiện thay đổi, trả về IR mới.

Pass Infrastructure trong thực tế:
    - LLVM: PassManager quản lý thứ tự chạy passes (FunctionPassManager, ModulePassManager)
    - MLIR: PassManager + OpPassManager, chạy trên từng Operation level
    - TVM: Sequential pass, Module pass, Function pass

Passes có sẵn:
    const_fold — Constant Folding: tính toán biểu thức hằng tại compile time
    dce        — Dead Code Elimination: xóa instructions không ai dùng
"""

from .const_fold import ConstantFoldingPass
from .dce import DeadCodeEliminationPass

__all__ = ['ConstantFoldingPass', 'DeadCodeEliminationPass']

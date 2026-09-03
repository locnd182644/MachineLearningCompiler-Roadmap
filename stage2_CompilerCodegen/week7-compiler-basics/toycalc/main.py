"""
ToyCalc Compiler — Main Entry Point
====================================

Module này là điểm khởi đầu chính (main entry point) kết nối toàn bộ pipeline của
ToyCalc compiler lại với nhau:

Pipeline:
    Source text
        ↓  (Phase 1: Lexer)
    Token stream
        ↓  (Phase 2: Parser)
    AST (Abstract Syntax Tree)
        ↓  (Phase 3: IR Lowering)
    3-Address Code IR (SSA form)
        ↓  (Phase 4: Optimization passes - ConstFold + DCE)
    Optimized IR
        ↓  (Phase 5: Codegen)
    Stack Machine Bytecode
        ↓  (Phase 6: VM Interpreter)
    Program Output (Execution)

Đây là mô hình thu nhỏ hoàn chỉnh của một modern compiler pipeline:
    - Clang/LLVM: C/C++ → AST → LLVM IR → LLVM Opt Passes → Machine Code
    - MLIR: Dialect AST → High-level IR → Progressive Lowering → LLVM IR / PTX
    - TVM: Relay/Relax graph → TIR → TensorIR schedules → Target assembly/CUDA
    - XLA: JAX/TF HLO → HLO Passes (fusion, DCE) → LLVM IR → Binary

Cách sử dụng CLI:
    python -m toycalc.main --demo
    python -m toycalc.main 'let x = 3 + 4; print(x)'
    python -m toycalc.main --verbose 'let x = 3 + 4; print(x)'
    python -m toycalc.main --dump-ir 'let x = 3 + 4 * 2; print(x)'
    python -m toycalc.main --dump-ast 'let x = (1 + 2) * 3; print(x)'
    python -m toycalc.main --dump-stack 'let x = 42; print(x)'
    python -m toycalc.main --trace 'let x = 5; print(x * 2)'
    echo 'let x = 10; print(x * 3)' | python -m toycalc.main
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Any

try:
    from .lexer import Lexer, LexerError
    from .parser import Parser, parse_source, ParseError
    from .ast_printer import ASTPrinter, CompactPrinter
    from .ir import ASTToIR
    from .passes.const_fold import ConstantFoldingPass
    from .passes.dce import DeadCodeEliminationPass
    from .codegen import IRToStack
    from .vm import VM, VMError
except ImportError:
    # Hỗ trợ chạy trực tiếp: python stage2_CompilerCodegen/.../toycalc/main.py
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from toycalc.lexer import Lexer, LexerError
    from toycalc.parser import Parser, parse_source, ParseError
    from toycalc.ast_printer import ASTPrinter, CompactPrinter
    from toycalc.ir import ASTToIR
    from toycalc.passes.const_fold import ConstantFoldingPass
    from toycalc.passes.dce import DeadCodeEliminationPass
    from toycalc.codegen import IRToStack
    from toycalc.vm import VM, VMError


# =============================================================================
# DANH SÁCH CHƯƠNG TRÌNH MẪU CHO DEMO MODE
# =============================================================================

programs = [
    ("let x = 3 + 4 * 2; print(x)", "Basic arithmetic with precedence"),
    ("let x = (3 + 4) * 2; print(x)", "Parenthesized expression"),
    ("let a = 5; let b = a * 2; print(b)", "Variable chaining"),
    ("let x = 2 * 3; let y = x + 4; let z = y * 2; print(z)", "Multi-step computation"),
    ("let x = -5; let y = x * x; print(y - 1)", "Negative numbers"),
    ("let a = 100; let b = a / 5; let c = a - b; print(c)", "Division"),
    ("let pi = 3.14; print(pi * 2)", "Floating point"),
    ("let unused = 999; let x = 42; print(x)", "Dead code elimination demo"),
]


# =============================================================================
# CORE PIPELINE FUNCTION
# =============================================================================

def compile_and_run(
    source: str,
    verbose: bool = False,
    dump_ir: bool = False,
    dump_ast: bool = False,
    dump_stack: bool = False,
    trace: bool = False,
) -> list:
    """
    Full pipeline: source → tokens → AST → IR → optimized IR → stack code → VM execution.
    
    Quy trình thực thi qua từng phase:
    1. Lexer:         Source text → Token stream
    2. Parser:        Tokens → Abstract Syntax Tree (AST)
    3. IR Gen:        AST → 3-Address Code IR (SSA form)
    4. Optimization:  Constant Folding Pass + Dead Code Elimination (DCE) Pass
    5. Codegen:       Optimized IR → Stack Machine instructions
    6. VM Execution:  Chạy bytecode trên Stack Machine Interpreter
    
    Args:
        source:      Chuỗi mã nguồn ToyCalc.
        verbose:     Nếu True, in chi tiết kết quả qua từng phase biên dịch.
        dump_ir:     Nếu True, in IR trước và sau khi tối ưu hóa.
        dump_ast:    Nếu True, in cấu trúc cây AST.
        dump_stack:  Nếu True, in mã máy ngăn xếp (stack bytecode).
        trace:       Nếu True, in trace từng lệnh thực thi và trạng thái stack của VM.
        
    Returns:
        Danh sách các giá trị output thu được từ các lệnh print().
    """
    if verbose:
        print("=" * 70)
        print(" TOYCALC COMPILER PIPELINE")
        print("=" * 70)
        print(f"Source:\n  {source.strip()}\n")

    # -------------------------------------------------------------------------
    # Phase 1: Lexer (Tokenization)
    # Chuyển đổi mã nguồn dạng văn bản thành chuỗi tokens.
    # -------------------------------------------------------------------------
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    if verbose:
        print("─" * 70)
        print(f"[Phase 1: Lexer] Tokenization ({len(tokens)} tokens)")
        for i, tok in enumerate(tokens):
            print(f"  [{i:3d}] {tok}")
        print()

    # -------------------------------------------------------------------------
    # Phase 2: Parser (AST Construction)
    # Phân tích cú pháp đệ quy (Recursive Descent) và dựng cây cú pháp trừu tượng.
    # -------------------------------------------------------------------------
    parser = Parser(tokens)
    ast = parser.parse()

    ast_formatted = ASTPrinter().format(ast)
    ast_compact = CompactPrinter().format(ast)

    if verbose:
        print("─" * 70)
        print("[Phase 2: Parser] Abstract Syntax Tree (AST)")
        print(ast_formatted)
        print(f"\n  Compact S-expression: {ast_compact}\n")
    elif dump_ast:
        print("--- AST (Abstract Syntax Tree) ---")
        print(ast_formatted)
        print("--- End AST ---\n")

    # -------------------------------------------------------------------------
    # Phase 3: IR Generation (AST → 3-Address Code SSA)
    # Hạ cấp (lower) AST thành IR tuyến tính 3 địa chỉ với tính chất SSA.
    # -------------------------------------------------------------------------
    ast_to_ir = ASTToIR()
    ir_raw = ast_to_ir.lower(ast)

    if verbose:
        print("─" * 70)
        print("[Phase 3: IR Generation] 3-Address Code (SSA)")
        print(ir_raw.dump("IR Gốc (Before Optimization)"))
        print()

    # -------------------------------------------------------------------------
    # Phase 4: Optimization Passes
    # Tối ưu hóa trên IR:
    #   Pass 1: Constant Folding (tính hằng số tại compile-time đến fixpoint)
    #   Pass 2: Dead Code Elimination (loại bỏ instructions không sử dụng)
    # -------------------------------------------------------------------------
    const_folder = ConstantFoldingPass()
    ir_folded = const_folder.run(ir_raw)

    dce = DeadCodeEliminationPass()
    ir_opt = dce.run(ir_folded)

    if verbose:
        print("─" * 70)
        print("[Phase 4: Optimization Passes]")
        print("• Pass 1: Constant Folding")
        print(ir_folded.dump("IR sau Constant Folding"))
        print()
        print("• Pass 2: Dead Code Elimination (DCE)")
        print(ir_opt.dump("IR sau DCE (Final Optimized IR)"))
        print()
    elif dump_ir:
        print(ir_raw.dump("IR Trước Tối Ưu (Before Optimization)"))
        print()
        print(ir_opt.dump("IR Sau Tối Ưu (After Optimization)"))
        print()

    # -------------------------------------------------------------------------
    # Phase 5: Code Generation (IR → Stack Machine Bytecode)
    # Chuyển đổi mã 3 địa chỉ sang tập lệnh máy ngăn xếp (PUSH, LOAD, STORE, ADD, ...)
    # -------------------------------------------------------------------------
    codegen = IRToStack()
    stack_prog = codegen.lower(ir_opt)

    if verbose:
        print("─" * 70)
        print("[Phase 5: Codegen] Stack Machine Instructions")
        print(stack_prog.dump("Stack Machine Code"))
        print()
    elif dump_stack:
        print(stack_prog.dump("Stack Machine Code"))
        print()

    # -------------------------------------------------------------------------
    # Phase 6: Virtual Machine Execution
    # Thực thi bytecode với fetch-decode-execute cycle trên Stack VM.
    # -------------------------------------------------------------------------
    if verbose:
        print("─" * 70)
        print("[Phase 6: VM Execution] Virtual Machine Interpreter")

    if trace:
        print("--- VM Execution Trace ---")

    vm = VM(verbose=trace)
    result = vm.run(stack_prog)

    if trace:
        print("--- End Trace ---\n")

    if verbose:
        formatted_outputs = [int(x) if x == int(x) else x for x in result.outputs]
        formatted_vars = {k: int(v) if v == int(v) else v for k, v in result.variables.items()}
        print(f"Output:    {formatted_outputs}")
        print(f"Variables: {formatted_vars}")
        print("=" * 70 + "\n")

    return result.outputs


# =============================================================================
# DEMO MODE
# =============================================================================

def run_demo(
    verbose: bool = False,
    dump_ir: bool = False,
    dump_ast: bool = False,
    dump_stack: bool = False,
    trace: bool = False,
):
    """Chạy danh sách chương trình mẫu để minh họa các khả năng của ToyCalc."""
    print("=" * 70)
    print(" TOYCALC DEMO SUITE — Minh Họa Toàn Bộ Compiler Pipeline")
    print("=" * 70)

    for i, (source, desc) in enumerate(programs, 1):
        print(f"\n{'─' * 70}")
        print(f"Demo {i}/{len(programs)}: {desc}")
        print(f"Source: {source}")
        print(f"{'─' * 70}")

        outputs = compile_and_run(
            source=source,
            verbose=verbose,
            dump_ir=dump_ir,
            dump_ast=dump_ast,
            dump_stack=dump_stack,
            trace=trace,
        )

        if not verbose:
            formatted = [int(x) if x == int(x) else x for x in outputs]
            print(f"Output: {formatted}")

    print(f"\n{'=' * 70}")
    print(" Đã hoàn thành toàn bộ demo!")
    print("=" * 70)


# =============================================================================
# ARGUMENT PARSING & CLI
# =============================================================================

def parse_args() -> argparse.ArgumentParser:
    """Tạo bộ phân tích đối số dòng lệnh."""
    parser = argparse.ArgumentParser(
        prog="python -m toycalc.main",
        description="ToyCalc — Trình biên dịch biểu thức số học với pipeline hoàn chỉnh.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Mã nguồn ToyCalc cần biên dịch (hoặc '-' để nhận từ stdin).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="In chi tiết kết quả qua từng phase trong compiler pipeline.",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="In IR trước và sau khi tối ưu hóa (Constant Folding + DCE).",
    )
    parser.add_argument(
        "--dump-ast",
        action="store_true",
        help="In cây cú pháp trừu tượng (Abstract Syntax Tree).",
    )
    parser.add_argument(
        "--dump-stack",
        action="store_true",
        help="In mã máy ngăn xếp (stack machine bytecode).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="In trace từng lệnh thực thi và trạng thái stack của Virtual Machine.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Chạy danh sách các chương trình mẫu minh họa compiler pipeline.",
    )
    return parser


def main():
    """Hàm main thực thi khi chạy từ dòng lệnh."""
    parser = parse_args()
    args = parser.parse_args()

    # 1. Xử lý demo mode
    if args.demo:
        run_demo(
            verbose=args.verbose,
            dump_ir=args.dump_ir,
            dump_ast=args.dump_ast,
            dump_stack=args.dump_stack,
            trace=args.trace,
        )
        return

    # 2. Thu thập mã nguồn từ đối số dòng lệnh hoặc stdin
    source = None
    if args.source:
        if args.source == "-":
            source = sys.stdin.read()
        else:
            source = args.source
    elif not sys.stdin.isatty():
        source = sys.stdin.read()

    # Kiểm tra nếu không có mã nguồn hợp lệ
    if not source or not source.strip():
        parser.print_help()
        print(
            "\nLỗi: Vui lòng cung cấp mã nguồn qua đối số, stdin, hoặc sử dụng cờ --demo.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Biên dịch và chạy mã nguồn
    try:
        outputs = compile_and_run(
            source=source,
            verbose=args.verbose,
            dump_ir=args.dump_ir,
            dump_ast=args.dump_ast,
            dump_stack=args.dump_stack,
            trace=args.trace,
        )

        # Nếu không ở chế độ verbose, in giá trị output ra màn hình chuẩn
        if not args.verbose:
            for val in outputs:
                formatted = int(val) if val == int(val) else val
                print(formatted)

    except (LexerError, ParseError, VMError) as err:
        print(f"Lỗi: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"Lỗi không mong muốn: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

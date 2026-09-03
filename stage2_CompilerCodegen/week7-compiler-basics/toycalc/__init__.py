"""
toycalc — Toy Calculator Compiler
=================================

Bài tập 7.1: Compiler cho ngôn ngữ biểu thức số học đơn giản.

Pipeline:
    Source → Lexer → Parser (AST) → IR (3-address code, SSA)
    → Optimization passes (const folding + DCE)
    → Codegen (stack machine) → VM interpreter

Đây là phiên bản thu nhỏ của pipeline mà mọi AI compiler (XLA, TVM, MLIR-based)
đều tuân theo. Mỗi module trong package tương ứng với một phase trong compiler.

Modules:
    lexer       — Tokenization (source text → token stream)
    parser      — Parsing + AST construction (tokens → Abstract Syntax Tree)
    ast_printer — Pretty-print AST (debug tool #1)
    ir          — Intermediate Representation (AST → 3-address code SSA)
    passes/     — Optimization passes (const_fold, dce)
    codegen     — Code generation (IR → stack machine instructions)
    vm          — Virtual Machine / interpreter (execute stack machine)
    main        — Entry point tying everything together
"""

__version__ = "0.1.0"
__author__ = "Week 7 - ML Compiler Engineering"

from .main import compile_and_run, programs

__all__ = ["compile_and_run", "programs"]


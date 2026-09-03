"""
AST Pretty Printer — Công cụ Debug #1 của Compiler Engineer
=============================================================

"DUMP ĐƯỢC LÀ DEBUG ĐƯỢC" — Quy tắc vàng #1

    Kỹ năng nghề nghiệp số 1 của compiler engineer là nhìn IR
    trước/sau mỗi pass và giải thích sự khác biệt.
    
    --mlir-print-ir-after-all là bạn thân.
    
    AST Printer là bước đầu tiên xây dựng thói quen này.
    Mỗi phase trong compiler PHẢI có pretty printer.

MODULE NÀY LÀM GÌ:
    1. In AST dạng cây có indentation (dễ đọc)
    2. In AST dạng compact 1 dòng (cho logs)
    3. (Optional) Xuất DOT format cho Graphviz visualization

SO SÁNH VỚI AI COMPILERS:
    - MLIR: mọi op đều có print() method, dump() cho debug
    - PyTorch FX: graph.print_tabular(), graph.python_code
    - TVM: relay.astext(), tir.astext()  
    - XLA: HloModule::ToString()
    → MỌI compiler framework đều đầu tư vào pretty printing
"""

from __future__ import annotations
from typing import Any
from .parser import (
    ASTNode, ASTVisitor, Program, LetStatement, PrintStatement,
    BinaryOp, UnaryOp, NumberLiteral, Identifier
)


class ASTPrinter(ASTVisitor):
    """
    Pretty printer cho AST — in dạng cây có indentation.
    
    Output ví dụ:
        Program
        ├── LetStatement (x)
        │   └── BinaryOp (+)
        │       ├── NumberLiteral (3.0)
        │       └── BinaryOp (*)
        │           ├── NumberLiteral (4.0)
        │           └── NumberLiteral (2.0)
        └── PrintStatement
            └── Identifier (x)
    """
    
    def __init__(self):
        self._lines: list[str] = []
    
    def format(self, node: ASTNode) -> str:
        """Format AST thành string có indentation."""
        self._lines = []
        self._print_node(node, prefix="", is_last=True, is_root=True)
        return "\n".join(self._lines)
    
    def _print_node(self, node: ASTNode, prefix: str, is_last: bool, is_root: bool = False):
        """
        In một node và đệ quy xuống children.
        
        Sử dụng Unicode box-drawing characters cho đẹp:
        ├── cho node giữa, └── cho node cuối
        """
        # Connector line
        if is_root:
            connector = ""
            child_prefix = ""
        elif is_last:
            connector = "└── "
            child_prefix = prefix + "    "
        else:
            connector = "├── "
            child_prefix = prefix + "│   "
        
        # Dispatch theo node type
        if isinstance(node, Program):
            self._lines.append(f"{prefix}{connector}Program")
            for i, stmt in enumerate(node.statements):
                self._print_node(stmt, child_prefix, i == len(node.statements) - 1)
        
        elif isinstance(node, LetStatement):
            self._lines.append(f"{prefix}{connector}LetStatement ({node.name})")
            self._print_node(node.value, child_prefix, is_last=True)
        
        elif isinstance(node, PrintStatement):
            self._lines.append(f"{prefix}{connector}PrintStatement")
            self._print_node(node.expr, child_prefix, is_last=True)
        
        elif isinstance(node, BinaryOp):
            self._lines.append(f"{prefix}{connector}BinaryOp ({node.op})")
            self._print_node(node.left, child_prefix, is_last=False)
            self._print_node(node.right, child_prefix, is_last=True)
        
        elif isinstance(node, UnaryOp):
            self._lines.append(f"{prefix}{connector}UnaryOp ({node.op})")
            self._print_node(node.operand, child_prefix, is_last=True)
        
        elif isinstance(node, NumberLiteral):
            # Format: bỏ ".0" nếu là số nguyên
            val = int(node.value) if node.value == int(node.value) else node.value
            self._lines.append(f"{prefix}{connector}NumberLiteral ({val})")
        
        elif isinstance(node, Identifier):
            self._lines.append(f"{prefix}{connector}Identifier ({node.name})")
        
        else:
            self._lines.append(f"{prefix}{connector}<unknown node: {type(node).__name__}>")
    
    # --- Visitor methods (alternative interface) ---
    
    def visit_Program(self, node: Program) -> str:
        return self.format(node)
    
    def visit_LetStatement(self, node: LetStatement) -> str:
        return self.format(node)
    
    def visit_PrintStatement(self, node: PrintStatement) -> str:
        return self.format(node)
    
    def visit_BinaryOp(self, node: BinaryOp) -> str:
        return self.format(node)
    
    def visit_UnaryOp(self, node: UnaryOp) -> str:
        return self.format(node)
    
    def visit_NumberLiteral(self, node: NumberLiteral) -> str:
        return self.format(node)
    
    def visit_Identifier(self, node: Identifier) -> str:
        return self.format(node)


class CompactPrinter:
    """
    In AST dạng compact 1 dòng — hữu ích cho logs và test assertions.
    
    Output ví dụ:
        (let x (+ 3 (* 4 2)))
        (print (- (identifier x) 1))
    
    Dạng S-expression, giống Lisp. Đơn giản, rõ cấu trúc.
    """
    
    def format(self, node: ASTNode) -> str:
        if isinstance(node, Program):
            stmts = " ".join(self.format(s) for s in node.statements)
            return f"(program {stmts})"
        
        elif isinstance(node, LetStatement):
            return f"(let {node.name} {self.format(node.value)})"
        
        elif isinstance(node, PrintStatement):
            return f"(print {self.format(node.expr)})"
        
        elif isinstance(node, BinaryOp):
            return f"({node.op} {self.format(node.left)} {self.format(node.right)})"
        
        elif isinstance(node, UnaryOp):
            return f"({node.op} {self.format(node.operand)})"
        
        elif isinstance(node, NumberLiteral):
            val = int(node.value) if node.value == int(node.value) else node.value
            return str(val)
        
        elif isinstance(node, Identifier):
            return node.name
        
        return f"<unknown:{type(node).__name__}>"


class DOTPrinter:
    """
    Xuất AST dạng Graphviz DOT — cho visualization.
    
    Dùng:
        dot_str = DOTPrinter().format(ast)
        # Ghi ra file .dot, rồi: dot -Tpng ast.dot -o ast.png
    
    Visualization giúp hiểu cấu trúc AST phức tạp — đặc biệt hữu ích
    khi debug parser cho biểu thức lồng nhiều tầng.
    """
    
    def __init__(self):
        self._node_id = 0
        self._lines: list[str] = []
    
    def _new_id(self) -> str:
        self._node_id += 1
        return f"n{self._node_id}"
    
    def format(self, node: ASTNode) -> str:
        self._node_id = 0
        self._lines = [
            "digraph AST {",
            '  node [shape=box, style="rounded,filled", fillcolor="#E8F5E9"];',
            '  edge [color="#424242"];',
        ]
        self._visit(node)
        self._lines.append("}")
        return "\n".join(self._lines)
    
    def _visit(self, node: ASTNode) -> str:
        nid = self._new_id()
        
        if isinstance(node, Program):
            self._lines.append(f'  {nid} [label="Program", fillcolor="#C8E6C9"];')
            for stmt in node.statements:
                child_id = self._visit(stmt)
                self._lines.append(f'  {nid} -> {child_id};')
        
        elif isinstance(node, LetStatement):
            self._lines.append(f'  {nid} [label="let {node.name}"];')
            child_id = self._visit(node.value)
            self._lines.append(f'  {nid} -> {child_id};')
        
        elif isinstance(node, PrintStatement):
            self._lines.append(f'  {nid} [label="print"];')
            child_id = self._visit(node.expr)
            self._lines.append(f'  {nid} -> {child_id};')
        
        elif isinstance(node, BinaryOp):
            self._lines.append(f'  {nid} [label="{node.op}", fillcolor="#BBDEFB"];')
            left_id = self._visit(node.left)
            right_id = self._visit(node.right)
            self._lines.append(f'  {nid} -> {left_id} [label="L"];')
            self._lines.append(f'  {nid} -> {right_id} [label="R"];')
        
        elif isinstance(node, UnaryOp):
            self._lines.append(f'  {nid} [label="unary {node.op}"];')
            child_id = self._visit(node.operand)
            self._lines.append(f'  {nid} -> {child_id};')
        
        elif isinstance(node, NumberLiteral):
            val = int(node.value) if node.value == int(node.value) else node.value
            self._lines.append(f'  {nid} [label="{val}", fillcolor="#FFF9C4"];')
        
        elif isinstance(node, Identifier):
            self._lines.append(f'  {nid} [label="{node.name}", fillcolor="#F3E5F5"];')
        
        return nid


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from toycalc.parser import parse_source
    
    print("=" * 60)
    print(" AST PRETTY PRINTER — 'Dump được là debug được'")
    print("=" * 60)
    
    source = "let x = 3 + 4 * 2; let y = x * x; print(y - 1)"
    print(f"\nSource: {source}\n")
    
    ast = parse_source(source)
    
    # Tree format
    print("--- Tree Format ---")
    printer = ASTPrinter()
    print(printer.format(ast))
    
    # Compact format
    print("\n--- Compact (S-expression) Format ---")
    compact = CompactPrinter()
    print(compact.format(ast))
    
    # Thêm ví dụ
    print("\n" + "=" * 60)
    source2 = "let result = (10 + 20) * 3; print(result)"
    print(f"\nSource: {source2}\n")
    ast2 = parse_source(source2)
    print("--- Tree Format ---")
    print(printer.format(ast2))
    print("\n--- Compact Format ---")
    print(compact.format(ast2))
    
    # DOT format
    print("\n--- DOT Format (cho Graphviz) ---")
    dot = DOTPrinter()
    print(dot.format(ast2))
    print("\n💡 Ghi ra file .dot rồi chạy: dot -Tpng ast.dot -o ast.png")

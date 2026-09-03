"""
Intermediate Representation (IR) — Phase 3 của Compiler Pipeline
=================================================================

IR LÀ GÌ?
    IR (Intermediate Representation) là cách compiler biểu diễn chương trình
    ở dạng TRUNG GIAN — giữa source code (gần người) và machine code (gần máy).
    
    Tại sao cần IR?
    1. SOURCE CODE quá cao cấp để optimize trực tiếp (cú pháp phức tạp)
    2. MACHINE CODE quá thấp cấp để phân tích ngữ nghĩa
    3. IR = sweet spot: đủ đơn giản để analyze + transform, đủ giàu để giữ ngữ nghĩa

    AST (cây)              →    IR tuyến tính (3-address code)
    BinaryOp(+)                 %t0 = CONST 3
    ├── 3                       %t1 = CONST 4
    └── BinaryOp(*)             %t2 = CONST 2
        ├── 4                   %t3 = MUL %t1, %t2
        └── 2                   %t4 = ADD %t0, %t3

3-ADDRESS CODE:
    Mỗi instruction có nhiều nhất 3 operand: dest = op src1, src2
    Đây là format cổ điển nhất cho IR, LLVM IR cũng là dạng 3-address.
    
    Ví dụ LLVM IR:
        %3 = add i32 %1, %2     ; 3-address: dest=%3, src1=%1, src2=%2

SSA — STATIC SINGLE ASSIGNMENT (KHÁI NIỆM QUAN TRỌNG NHẤT):
    Quy tắc: MỖI BIẾN CHỈ ĐƯỢC GÁN ĐÚNG 1 LẦN.
    
    Không SSA:                  SSA:
        x = 1                     %x.1 = 1
        x = x + 2                %x.2 = ADD %x.1, 2
        x = x * 3                %x.3 = MUL %x.2, 3
    
    Tại sao SSA quan trọng?
    1. Def-use chains HIỂN NHIÊN: mỗi use chỉ đến đúng 1 def
    2. Mọi dataflow analysis đơn giản hóa đáng kể
    3. Constant propagation, DCE, copy propagation trivially correct
    4. LLVM IR là SSA. MLIR là SSA. TVM TIR là SSA.
       → KHÔNG HIỂU SSA THÌ KHÔNG ĐỌC ĐƯỢC MLIR.
    
    MLIR dùng SSA values:
        %result = arith.addf %a, %b : f32
        ^^ SSA value, gán đúng 1 lần, dùng nhiều lần

PROGRESSIVE LOWERING:
    AST (tree) → IR (linear) là bước lowering đầu tiên.
    Trong MLIR, concept tương tự:
        High-level dialect → Low-level dialect
        linalg.matmul → affine.for loops → llvm.call
    Mỗi bước lowering MẤT thông tin nhưng GẠNH thêm chi tiết.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from .parser import (
    ASTNode, ASTVisitor, Program, LetStatement, PrintStatement,
    BinaryOp, UnaryOp, NumberLiteral, Identifier
)


# =============================================================================
# IR INSTRUCTION SET
# =============================================================================
# Instruction set nhỏ gọn cho toy compiler.
# So sánh:
#   LLVM IR: ~60 instruction types (add, sub, mul, load, store, br, phi, ...)
#   MLIR: mỗi dialect có op set riêng (arith: ~30 ops, linalg: ~20 ops, ...)
#   XLA HLO: ~100 ops (add, multiply, dot, convolution, ...)

class IROpcode(Enum):
    """Opcodes cho IR instructions."""
    
    # Constant loading
    CONST = auto()      # %t = CONST value      → load hằng số
    
    # Arithmetic operations (binary)
    ADD = auto()        # %t = ADD %a, %b       → cộng
    SUB = auto()        # %t = SUB %a, %b       → trừ  
    MUL = auto()        # %t = MUL %a, %b       → nhân
    DIV = auto()        # %t = DIV %a, %b       → chia
    
    # Unary operations
    NEG = auto()        # %t = NEG %a           → đảo dấu
    
    # Data movement
    COPY = auto()       # %t = COPY %a          → copy value (cho variable lookup)
    
    # Side effects
    PRINT = auto()      # PRINT %a              → in giá trị
    
    # No-operation (placeholder, sẽ bị DCE xóa)
    NOP = auto()        # NOP                   → không làm gì


# Map từ AST operator sang IR opcode
OP_TO_OPCODE = {
    '+': IROpcode.ADD,
    '-': IROpcode.SUB,
    '*': IROpcode.MUL,
    '/': IROpcode.DIV,
}


# =============================================================================
# IR INSTRUCTION CLASS
# =============================================================================

@dataclass
class IRInstruction:
    """
    Một instruction trong IR — dạng 3-address code.
    
    Format: dest = opcode src1, src2
    
    Ví dụ:
        %t0 = CONST 42          (dest=%t0, opcode=CONST, value=42)
        %t2 = ADD %t0, %t1      (dest=%t2, opcode=ADD, src1=%t0, src2=%t1)
        PRINT %t2               (dest=None, opcode=PRINT, src1=%t2)
    
    So sánh MLIR:
        %0 = arith.constant 42 : i32
        %2 = arith.addi %0, %1 : i32
        → Cùng format: %result = dialect.op operands : type
    """
    opcode: IROpcode
    dest: Optional[str] = None      # Destination temp (%t0, %t1, ...)
    src1: Optional[str] = None      # First source operand
    src2: Optional[str] = None      # Second source operand  
    value: Optional[float] = None   # Constant value (for CONST opcode)
    
    def __repr__(self) -> str:
        return self.format()
    
    def format(self) -> str:
        """Format instruction thành string dễ đọc."""
        if self.opcode == IROpcode.CONST:
            val = int(self.value) if self.value == int(self.value) else self.value
            return f"  {self.dest} = CONST {val}"
        
        elif self.opcode == IROpcode.PRINT:
            return f"  PRINT {self.src1}"
        
        elif self.opcode == IROpcode.NOP:
            return f"  NOP"
        
        elif self.opcode == IROpcode.NEG:
            return f"  {self.dest} = NEG {self.src1}"
        
        elif self.opcode == IROpcode.COPY:
            return f"  {self.dest} = COPY {self.src1}"
        
        else:
            # Binary operation
            return f"  {self.dest} = {self.opcode.name} {self.src1}, {self.src2}"
    
    def is_constant(self) -> bool:
        """Instruction này có phải là constant load không?"""
        return self.opcode == IROpcode.CONST
    
    def has_side_effect(self) -> bool:
        """Instruction này có side effect không? (Không thể xóa bởi DCE)"""
        return self.opcode == IROpcode.PRINT
    
    def uses(self) -> List[str]:
        """Trả về list các temps mà instruction này SỬ DỤNG (reads from)."""
        result = []
        if self.src1 is not None and self.src1.startswith('%'):
            result.append(self.src1)
        if self.src2 is not None and self.src2.startswith('%'):
            result.append(self.src2)
        return result
    
    def defines(self) -> Optional[str]:
        """Trả về temp mà instruction này ĐỊNH NGHĨA (writes to), hoặc None."""
        return self.dest


# =============================================================================
# IR PROGRAM CLASS
# =============================================================================

@dataclass
class IRProgram:
    """
    Một chương trình IR = danh sách instructions tuyến tính.
    
    Đây là representation đơn giản nhất — không có basic blocks hay control flow.
    (LLVM IR và MLIR có basic blocks; chúng ta bỏ qua vì ToyCalc không có if/while)
    
    So sánh:
        LLVM: Module → Function → BasicBlock → Instruction
        MLIR: Module → Func → Region → Block → Operation
        ToyCalc: Program → [Instruction]   (đơn giản hóa)
    """
    instructions: List[IRInstruction] = field(default_factory=list)
    
    def dump(self, title: str = "IR") -> str:
        """
        In toàn bộ IR — KỸ NĂNG #1 CỦA COMPILER ENGINEER.
        
        Thói quen: dump IR trước/sau MỖI pass.
        MLIR tương đương: --mlir-print-ir-after-all
        """
        lines = [f"--- {title} ({len(self.instructions)} instructions) ---"]
        for i, inst in enumerate(self.instructions):
            lines.append(f"  [{i:3d}] {inst.format()}")
        lines.append(f"--- end {title} ---")
        return "\n".join(lines)
    
    def copy(self) -> 'IRProgram':
        """Deep copy — để pass tạo bản mới mà không sửa bản cũ."""
        import copy
        return IRProgram(instructions=[copy.copy(inst) for inst in self.instructions])
    
    def __len__(self) -> int:
        return len(self.instructions)


# =============================================================================
# AST → IR LOWERING
# =============================================================================

class ASTToIR(ASTVisitor):
    """
    Lowering AST → IR (3-address code, SSA).
    
    Đây là phase "instruction selection" đơn giản nhất:
    mỗi AST node → 1 hoặc vài IR instructions.
    
    PROGRESSIVE LOWERING trong MLIR:
        Toy dialect → Standard dialects → LLVM dialect
        Mỗi bước = 1 ConversionPattern class
        
    Ở đây:
        AST BinaryOp(+) → IR ADD instruction
        AST NumberLiteral(42) → IR CONST instruction
    
    SSA NAMING CONVENTION:
        Mỗi temp %t0, %t1, %t2, ... được gán ĐÚNG 1 LẦN.
        Counter tăng monotonically → guaranteed unique.
        
        MLIR dùng convention tương tự: %0, %1, %2, ...
        LLVM IR: %1, %2, %3, ... (hoặc named: %result, %sum, ...)
    """
    
    def __init__(self):
        self.instructions: List[IRInstruction] = []
        self._temp_counter = 0
        self._var_to_temp: Dict[str, str] = {}  # Biến → temp hiện tại
    
    def _new_temp(self) -> str:
        """
        Sinh tên temp mới: %t0, %t1, %t2, ...
        
        SSA guarantee: mỗi _new_temp() trả về tên CHƯA TỪNG xuất hiện.
        Giống mlir::OpBuilder::create() tự sinh %result mới.
        """
        name = f"%t{self._temp_counter}"
        self._temp_counter += 1
        return name
    
    def lower(self, program: Program) -> IRProgram:
        """
        Entry point: lower toàn bộ AST Program → IRProgram.
        
        Tương đương MLIR:
            mlir-opt --convert-toy-to-standard (Toy tutorial ch.5)
        """
        self.instructions = []
        self._temp_counter = 0
        self._var_to_temp = {}
        
        for stmt in program.statements:
            stmt.accept(self)
        
        return IRProgram(instructions=self.instructions)
    
    def _emit(self, inst: IRInstruction):
        """Thêm instruction vào chương trình."""
        self.instructions.append(inst)
    
    # -------------------------------------------------------------------------
    # Visitor methods — mỗi AST node type → IR instructions
    # -------------------------------------------------------------------------
    
    def visit_NumberLiteral(self, node: NumberLiteral) -> str:
        """
        NumberLiteral(42) → %t0 = CONST 42
        
        MLIR tương đương: %0 = arith.constant 42 : i32
        """
        dest = self._new_temp()
        self._emit(IRInstruction(
            opcode=IROpcode.CONST,
            dest=dest,
            value=node.value
        ))
        return dest
    
    def visit_Identifier(self, node: Identifier) -> str:
        """
        Identifier("x") → lookup temp đang giữ giá trị x.
        
        SSA: biến "x" trong source → 1 hoặc nhiều SSA temps.
        let x = 3; → %t0 = CONST 3, var_map["x"] = %t0
        Dùng x    → trả về %t0 (đã biết từ map)
        """
        if node.name not in self._var_to_temp:
            raise ValueError(f"Undefined variable: {node.name}")
        
        # Trả về temp đang giữ giá trị biến này
        return self._var_to_temp[node.name]
    
    def visit_BinaryOp(self, node: BinaryOp) -> str:
        """
        BinaryOp('+', left, right) → lower left, lower right, emit ADD
        
        Đệ quy: lower children trước, rồi dùng kết quả.
        Post-order traversal trên AST → instructions tuyến tính.
        
        MLIR tương đương:
            %lhs = ...
            %rhs = ...
            %result = arith.addf %lhs, %rhs : f32
        """
        # Lower children trước (post-order)
        left_temp = node.left.accept(self)
        right_temp = node.right.accept(self)
        
        # Emit instruction cho operation
        dest = self._new_temp()
        opcode = OP_TO_OPCODE.get(node.op)
        if opcode is None:
            raise ValueError(f"Unknown binary operator: {node.op}")
        
        self._emit(IRInstruction(
            opcode=opcode,
            dest=dest,
            src1=left_temp,
            src2=right_temp
        ))
        return dest
    
    def visit_UnaryOp(self, node: UnaryOp) -> str:
        """UnaryOp('-', operand) → NEG instruction."""
        operand_temp = node.operand.accept(self)
        dest = self._new_temp()
        self._emit(IRInstruction(
            opcode=IROpcode.NEG,
            dest=dest,
            src1=operand_temp
        ))
        return dest
    
    def visit_LetStatement(self, node: LetStatement) -> str:
        """
        LetStatement("x", expr) → lower expr, map "x" → result temp
        
        Không emit extra COPY instruction — SSA value đã unique.
        Chỉ cập nhật mapping: var_name → temp.
        """
        result_temp = node.value.accept(self)
        self._var_to_temp[node.name] = result_temp
        return result_temp
    
    def visit_PrintStatement(self, node: PrintStatement) -> str:
        """
        PrintStatement(expr) → lower expr, emit PRINT
        
        PRINT là instruction có SIDE EFFECT → DCE không được xóa.
        MLIR tương đương: "llvm.call @printf"
        """
        result_temp = node.expr.accept(self)
        self._emit(IRInstruction(
            opcode=IROpcode.PRINT,
            src1=result_temp
        ))
        return result_temp
    
    def visit_Program(self, node: Program) -> str:
        """Không nên gọi trực tiếp — dùng lower() method."""
        for stmt in node.statements:
            stmt.accept(self)
        return ""


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from toycalc.parser import parse_source
    
    print("=" * 60)
    print(" AST → IR LOWERING — Progressive Lowering Step 1")
    print("=" * 60)
    
    source = "let x = 3 + 4 * 2; let y = x * x; print(y - 1)"
    print(f"\nSource: {source}\n")
    
    # Parse → AST
    ast = parse_source(source)
    
    # Lower → IR
    lowerer = ASTToIR()
    ir = lowerer.lower(ast)
    
    # Dump IR — thói quen #1!
    print(ir.dump("Unoptimized IR"))
    
    print()
    print("💡 Chú ý tính chất SSA:")
    print("   - Mỗi %tN chỉ xuất hiện bên TRÁI dấu = ĐÚNG 1 LẦN")
    print("   - Def-use chains hiển nhiên: nhìn vào src1/src2 biết ngay từ đâu ra")
    print("   - Đây chính là cách MLIR/LLVM hoạt động!")
    
    print()
    print("=" * 60)
    source2 = "let a = (10 + 20) * 3; print(a)"
    print(f"\nSource: {source2}\n")
    ast2 = parse_source(source2)
    ir2 = ASTToIR().lower(ast2)
    print(ir2.dump("IR cho biểu thức có ngoặc"))

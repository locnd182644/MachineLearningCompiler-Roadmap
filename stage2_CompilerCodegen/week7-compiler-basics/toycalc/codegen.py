"""
Code Generation (Codegen) — Phase 5 của Compiler Pipeline
=========================================================

CODEGEN LÀ GÌ?
    Codegen là quá trình chuyển đổi Intermediate Representation (IR)
    thành mã máy thực tế (machine code) hoặc bytecode cho Virtual Machine.
    
    Trong pipeline của chúng ta:
        IR (3-address code, SSA) → Stack machine instructions
    
    Trong compiler thực tế:
        LLVM IR → SelectionDAG → MachineInstr → MCInst → assembly/object code

STACK MACHINE VS REGISTER MACHINE:
    1. Stack Machine (JVM, CPython bytecode, WebAssembly, PostScript):
       - Toán hạng được lấy từ stack, kết quả đẩy lại stack
       - Mã ngắn gọn, dễ sinh code, dễ viết interpreter
       - KHÔNG CẦN register allocation (bài toán khó nhất trong codegen!)
       - Ví dụ: PUSH 3, PUSH 4, ADD → stack chứa [7]
    
    2. Register Machine (x86, ARM, RISC-V):
       - CPU thực có số register CỐ ĐỊNH (x86: 16, ARM: 31)
       - Cần thuật toán register allocation phức tạp (graph coloring NP-hard)
       - Chạy nhanh hơn stack machine vì truy cập register nhanh hơn memory

    Chúng ta chọn Stack Machine vì:
    - Đơn giản, focus vào học compiler pipeline
    - JVM (Java) và CPython đều dùng stack machine → pattern rất phổ biến
    - WebAssembly cũng là stack machine → công nghệ mới nhất cũng dùng

LIÊN HỆ VỚI AI COMPILER:
    - LLVM codegen: SelectionDAG → MachineInstr → MCInst → assembly
    - TVM: schedule → lower → generate CUDA/OpenCL kernel code
    - Triton: AST → Triton IR → LLVM IR → PTX
    - XLA: HLO → LLVM IR (CPU) hoặc → PTX (GPU)
    
    AI compilers thường target GPU (PTX, AMDGPU) thay vì CPU.
    Quá trình tương tự: IR → target-specific code.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Any

from .ir import IRProgram, IROpcode, IRInstruction


# =============================================================================
# STACK INSTRUCTION SET
# =============================================================================
# Instruction set cho stack machine tự định nghĩa.
# So sánh:
#   JVM bytecode: ~200 opcodes (iadd, iload, istore, invokevirtual, ...)
#   CPython bytecode: ~120 opcodes (LOAD_FAST, BINARY_ADD, POP_TOP, ...)
#   WebAssembly: ~380 opcodes (i32.add, i32.load, call, ...)

class StackOp(Enum):
    """Opcodes cho Stack Machine."""
    
    # Data movement
    PUSH = auto()       # PUSH value    → đẩy hằng số lên đỉnh stack
    POP = auto()        # POP           → lấy đỉnh stack bỏ đi
    LOAD = auto()       # LOAD name     → load giá trị biến lên stack
    STORE = auto()      # STORE name    → pop đỉnh stack, lưu vào biến
    
    # Arithmetic — pop 2 operands, push 1 result
    ADD = auto()        # ADD           → pop b, pop a, push a+b
    SUB = auto()        # SUB           → pop b, pop a, push a-b
    MUL = auto()        # MUL           → pop b, pop a, push a*b
    DIV = auto()        # DIV           → pop b, pop a, push a/b
    
    # Unary — pop 1 operand, push 1 result
    NEG = auto()        # NEG           → pop a, push -a
    
    # I/O — side effects
    PRINT = auto()      # PRINT         → pop a, in ra, ghi vào output


# =============================================================================
# STACK INSTRUCTION CLASS
# =============================================================================

@dataclass
class StackInstruction:
    """
    Một instruction của Stack Machine.
    
    Khác với 3-address code (dest = op src1, src2),
    hầu hết stack instructions KHÔNG CẦN explicit operands
    vì chúng mặc định thao tác với TOP OF STACK.
    
    Chỉ PUSH, LOAD, STORE cần operand (giá trị hoặc tên biến).
    
    So sánh JVM bytecode:
        iadd        → ADD (no operands, pops 2, pushes 1)
        bipush 42   → PUSH 42 (1 operand)
        iload_0     → LOAD "x" (1 operand — variable index)
    """
    opcode: StackOp
    operand: Any = None  # float (cho PUSH) hoặc str (cho LOAD/STORE)
    
    def format(self) -> str:
        """Format instruction thành string dễ đọc."""
        if self.operand is not None:
            if isinstance(self.operand, float):
                val = int(self.operand) if self.operand == int(self.operand) else self.operand
                return f"  {self.opcode.name} {val}"
            return f"  {self.opcode.name} {self.operand}"
        return f"  {self.opcode.name}"
    
    def __repr__(self) -> str:
        return self.format().strip()


# =============================================================================
# STACK PROGRAM CLASS
# =============================================================================

@dataclass
class StackProgram:
    """
    Chương trình Stack Machine = danh sách StackInstructions.
    
    Tương đương:
        JVM: .class file chứa bytecode
        CPython: .pyc file chứa marshalled bytecode
        WebAssembly: .wasm binary
    """
    instructions: List[StackInstruction] = field(default_factory=list)
    
    def dump(self, title: str = "Stack Code") -> str:
        """In toàn bộ stack program — debug tool."""
        lines = [f"--- {title} ({len(self.instructions)} instructions) ---"]
        for i, inst in enumerate(self.instructions):
            lines.append(f"  [{i:3d}] {inst.format()}")
        lines.append(f"--- end {title} ---")
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.instructions)


# =============================================================================
# IR → STACK MACHINE CODEGEN
# =============================================================================

class IRToStack:
    """
    Code Generator: IR (3-address code) → Stack Machine instructions.
    
    Quá trình chuyển đổi:
        Mỗi IR instruction (3-address) → 1 hoặc vài stack instructions.
    
    Ví dụ:
        IR:    %t2 = ADD %t0, %t1
        Stack: LOAD %t0      ← push giá trị %t0 lên stack
               LOAD %t1      ← push giá trị %t1 lên stack  
               ADD            ← pop 2, cộng, push kết quả
               STORE %t2     ← pop kết quả, lưu vào %t2
    
    Chiến lược:
        Dùng LOAD/STORE để map SSA temporaries (%t0, %t1, ...) thành
        named variables trong VM. Stack chỉ dùng cho tính toán tạm thời.
        
        Đây là cách đơn giản nhất — tối ưu hơn sẽ giữ values trên stack
        lâu hơn (register → stack slot mapping), nhưng phức tạp hơn nhiều.
    
    So sánh LLVM:
        LLVM SelectionDAG: chuyển LLVM IR → target-specific DAG
        → Instruction selection (pattern matching)
        → Register allocation (graph coloring)
        → Instruction scheduling
        → Emit assembly
    """
    
    def __init__(self):
        self.instructions: List[StackInstruction] = []
    
    def _emit(self, opcode: StackOp, operand: Any = None):
        """Thêm stack instruction vào chương trình."""
        self.instructions.append(StackInstruction(opcode, operand))
    
    def lower(self, ir_program: IRProgram) -> StackProgram:
        """
        Chuyển đổi IRProgram → StackProgram.
        
        Entry point chính. "Lower" vì chúng ta đang đi từ abstraction
        cao hơn (3-address IR) xuống thấp hơn (stack machine).
        
        Args:
            ir_program: IRProgram cần codegen.
            
        Returns:
            StackProgram sẵn sàng cho VM chạy.
        """
        self.instructions = []
        
        for inst in ir_program.instructions:
            self._lower_instruction(inst)
        
        return StackProgram(instructions=self.instructions)
    
    def _lower_instruction(self, inst: IRInstruction):
        """
        Chuyển 1 IR instruction → stack instructions tương đương.
        
        Đây là "instruction selection" đơn giản nhất —
        mỗi IR opcode → fixed sequence of stack ops.
        
        LLVM instruction selection phức tạp hơn nhiều:
        sử dụng pattern matching trên DAG (TableGen patterns).
        """
        if inst.opcode == IROpcode.CONST:
            # %t0 = CONST 42  →  PUSH 42; STORE %t0
            self._emit(StackOp.PUSH, inst.value)
            if inst.dest is not None:
                self._emit(StackOp.STORE, inst.dest)
        
        elif inst.opcode in (IROpcode.ADD, IROpcode.SUB, 
                             IROpcode.MUL, IROpcode.DIV):
            # %t2 = ADD %t0, %t1  →  LOAD %t0; LOAD %t1; ADD; STORE %t2
            self._emit(StackOp.LOAD, inst.src1)
            self._emit(StackOp.LOAD, inst.src2)
            
            # Map IR opcode → stack opcode
            op_map = {
                IROpcode.ADD: StackOp.ADD,
                IROpcode.SUB: StackOp.SUB,
                IROpcode.MUL: StackOp.MUL,
                IROpcode.DIV: StackOp.DIV,
            }
            self._emit(op_map[inst.opcode])
            
            if inst.dest is not None:
                self._emit(StackOp.STORE, inst.dest)
        
        elif inst.opcode == IROpcode.NEG:
            # %t1 = NEG %t0  →  LOAD %t0; NEG; STORE %t1
            self._emit(StackOp.LOAD, inst.src1)
            self._emit(StackOp.NEG)
            if inst.dest is not None:
                self._emit(StackOp.STORE, inst.dest)
        
        elif inst.opcode == IROpcode.COPY:
            # %t1 = COPY %t0  →  LOAD %t0; STORE %t1
            self._emit(StackOp.LOAD, inst.src1)
            if inst.dest is not None:
                self._emit(StackOp.STORE, inst.dest)
        
        elif inst.opcode == IROpcode.PRINT:
            # PRINT %t0  →  LOAD %t0; PRINT
            self._emit(StackOp.LOAD, inst.src1)
            self._emit(StackOp.PRINT)
        
        elif inst.opcode == IROpcode.NOP:
            # NOP → không sinh code (đã bị optimize away)
            pass


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from toycalc.parser import parse_source
    from toycalc.ir import ASTToIR
    
    print("=" * 60)
    print(" IR → STACK MACHINE CODEGEN")
    print("=" * 60)
    
    source = "let x = 3 + 4 * 2; print(x)"
    print(f"\nSource: {source}\n")
    
    # Parse → AST → IR
    ast = parse_source(source)
    ir = ASTToIR().lower(ast)
    print(ir.dump("IR Program"))
    
    # Codegen → Stack Machine
    print()
    codegen = IRToStack()
    stack_prog = codegen.lower(ir)
    print(stack_prog.dump("Stack Machine Code"))
    
    print()
    print("💡 Mỗi IR instruction 3-address → 2-4 stack instructions:")
    print("   '%t2 = ADD %t0, %t1' → 'LOAD %t0; LOAD %t1; ADD; STORE %t2'")
    print("   Stack machine đơn giản hơn nhưng cần nhiều instructions hơn.")
    print()
    print("   So sánh:")
    print("   - JVM: javac sinh bytecode tương tự cho Java expressions")
    print("   - CPython: compile() sinh bytecode LOAD_FAST, BINARY_ADD, STORE_FAST")

"""
Virtual Machine (VM) / Interpreter — Phase cuối của Compiler Pipeline
=====================================================================

VM LÀ GÌ?
    Virtual Machine (trong ngữ cảnh compiler) là phần mềm mô phỏng một CPU
    để thực thi bytecode/instructions mà compiler đã sinh ra.
    
    KHÔNG PHẢI VM trong nghĩa VMware/Docker — đó là system-level virtualization.
    Ở đây là process-level VM (language VM), giống:
    - JVM (Java Virtual Machine) — chạy Java bytecode
    - CPython VM — chạy Python bytecode (.pyc)  
    - V8 engine — chạy JavaScript
    - CLR (.NET) — chạy IL (Intermediate Language)

CẤU TRÚC VM CỦA CHÚNG TA:
    1. Instruction Memory: Chứa StackProgram (danh sách instructions)
    2. Program Counter (PC): Con trỏ tới instruction đang thực thi
    3. Stack: Cấu trúc LIFO cho tính toán trung gian
    4. Variables: Dictionary lưu giá trị các biến (named memory)
    5. Output: Danh sách giá trị đã print (cho testing)
    
    Fetch-Decode-Execute cycle:
        while PC < len(instructions):
            inst = instructions[PC]     # FETCH
            match inst.opcode:          # DECODE
                case ADD: ...           # EXECUTE
            PC += 1

STACK-BASED VS REGISTER-BASED VM:
    Stack-based (chúng ta, JVM, CPython):
        PUSH 3        stack: [3]
        PUSH 4        stack: [3, 4]
        ADD            stack: [7]        ← pop 2, push result
    
    Register-based (Lua VM, Dalvik/ART):
        LOAD R1, 3    R1 = 3
        LOAD R2, 4    R2 = 4
        ADD R3, R1, R2  R3 = 7
    
    Stack VM: dễ implement, mã nhỏ gọn (ít operands)
    Register VM: nhanh hơn (ít memory access), nhưng instructions lớn hơn

LIÊN HỆ VỚI AI COMPILER:
    Thực tế AI compilers KHÔNG dùng VM — chúng sinh native code:
    - XLA: sinh LLVM IR → compile thành native binary → chạy trực tiếp
    - TVM: sinh C/CUDA code → compile thành shared lib → chạy trên device
    - Triton: sinh PTX → chạy trực tiếp trên GPU
    
    Tuy nhiên:
    - PyTorch eager mode = interpreter cho computation graph
    - TorchScript có VM riêng (interpreter-based execution)
    - Concept VM giúp hiểu execution model của mọi runtime
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from .codegen import StackProgram, StackInstruction, StackOp


# =============================================================================
# VM RESULT
# =============================================================================

@dataclass
class VMResult:
    """
    Kết quả thực thi VM.
    
    Attributes:
        outputs:    Danh sách giá trị đã PRINT (theo thứ tự)
        variables:  Trạng thái cuối cùng của tất cả biến
        stack:      Trạng thái cuối cùng của stack (thường rỗng)
    """
    outputs: List[float] = field(default_factory=list)
    variables: Dict[str, float] = field(default_factory=dict)
    stack: List[float] = field(default_factory=list)


# =============================================================================
# VIRTUAL MACHINE CLASS  
# =============================================================================

class VMError(Exception):
    """Lỗi runtime trong VM."""
    def __init__(self, message: str, pc: int):
        self.pc = pc
        super().__init__(f"VM error at PC={pc}: {message}")


class VM:
    """
    Stack-based Virtual Machine cho ToyCalc.
    
    Thực thi StackProgram bằng fetch-decode-execute cycle.
    
    Cách dùng:
        vm = VM()
        result = vm.run(stack_program)
        print(result.outputs)   # Giá trị đã print
    
    State:
        stack:     Stack tính toán (list, top = cuối list)
        variables: Dictionary biến → giá trị
        outputs:   Accumulate giá trị print
        pc:        Program counter
    """
    
    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: Nếu True, in trace mỗi instruction (debug mode).
        """
        self.verbose = verbose
        self.stack: List[float] = []
        self.variables: Dict[str, float] = {}
        self.outputs: List[float] = []
        self.pc: int = 0
    
    def run(self, program: StackProgram) -> VMResult:
        """
        Thực thi StackProgram, trả về VMResult.
        
        Đây là main execution loop — tương đương:
        - JVM: interpreter loop trong hotspot/share/interpreter
        - CPython: ceval.c → _PyEval_EvalFrameDefault()
        
        Args:
            program: StackProgram cần chạy.
            
        Returns:
            VMResult chứa outputs, final variables, final stack.
        """
        # Reset state
        self.stack = []
        self.variables = {}
        self.outputs = []
        self.pc = 0
        
        instructions = program.instructions
        
        # Fetch-Decode-Execute loop
        while self.pc < len(instructions):
            inst = instructions[self.pc]
            
            if self.verbose:
                self._trace(inst)
            
            self._execute(inst)
            self.pc += 1
        
        return VMResult(
            outputs=list(self.outputs),
            variables=dict(self.variables),
            stack=list(self.stack)
        )
    
    def _execute(self, inst: StackInstruction):
        """
        Execute một instruction.
        
        Đây là "decode + execute" step trong fetch-decode-execute cycle.
        """
        op = inst.opcode
        
        if op == StackOp.PUSH:
            # PUSH value → đẩy lên stack
            self.stack.append(float(inst.operand))
        
        elif op == StackOp.POP:
            # POP → bỏ đỉnh stack
            self._check_stack(1)
            self.stack.pop()
        
        elif op == StackOp.LOAD:
            # LOAD name → load biến lên stack
            name = inst.operand
            if name not in self.variables:
                raise VMError(f"Undefined variable: {name}", self.pc)
            self.stack.append(self.variables[name])
        
        elif op == StackOp.STORE:
            # STORE name → pop, lưu vào biến
            self._check_stack(1)
            self.variables[inst.operand] = self.stack.pop()
        
        elif op == StackOp.ADD:
            self._binary_op(lambda a, b: a + b, "ADD")
        
        elif op == StackOp.SUB:
            self._binary_op(lambda a, b: a - b, "SUB")
        
        elif op == StackOp.MUL:
            self._binary_op(lambda a, b: a * b, "MUL")
        
        elif op == StackOp.DIV:
            self._check_stack(2)
            b = self.stack.pop()
            a = self.stack.pop()
            if b == 0:
                raise VMError("Division by zero", self.pc)
            self.stack.append(a / b)
        
        elif op == StackOp.NEG:
            self._check_stack(1)
            self.stack.append(-self.stack.pop())
        
        elif op == StackOp.PRINT:
            # PRINT → pop, ghi vào output
            self._check_stack(1)
            val = self.stack.pop()
            self.outputs.append(val)
        
        else:
            raise VMError(f"Unknown opcode: {op}", self.pc)
    
    def _binary_op(self, func, name: str):
        """Helper cho binary operations: pop 2, compute, push 1."""
        self._check_stack(2)
        b = self.stack.pop()
        a = self.stack.pop()
        self.stack.append(func(a, b))
    
    def _check_stack(self, n: int):
        """Kiểm tra stack có đủ elements không."""
        if len(self.stack) < n:
            raise VMError(
                f"Stack underflow: need {n} elements, have {len(self.stack)}",
                self.pc
            )
    
    def _trace(self, inst: StackInstruction):
        """In trace cho debug — mỗi instruction + trạng thái stack."""
        stack_str = str([int(x) if x == int(x) else x for x in self.stack])
        print(f"  PC={self.pc:3d} | {inst.format().strip():20s} | stack={stack_str}")


# =============================================================================
# DEMO — Full pipeline end-to-end
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from toycalc.parser import parse_source
    from toycalc.ir import ASTToIR
    from toycalc.codegen import IRToStack
    from toycalc.passes.const_fold import ConstantFoldingPass
    from toycalc.passes.dce import DeadCodeEliminationPass
    
    print("=" * 70)
    print(" TOYCALC VM — Full Pipeline: Source → Lexer → Parser → IR → Opt → VM")
    print("=" * 70)
    
    programs = [
        ("let x = 3 + 4 * 2; print(x)", [11]),
        ("let a = (10 + 20) * 3; print(a)", [90]),
        ("let x = 5; let y = x * x; print(y - 1)", [24]),
        ("let a = -5; let b = a * 2; print(b)", [-10]),
        ("let pi = 3.14; print(pi * 2)", [6.28]),
    ]
    
    for source, expected in programs:
        print(f"\n{'─' * 60}")
        print(f"Source: {source}")
        
        # Full pipeline
        ast = parse_source(source)
        ir = ASTToIR().lower(ast)
        ir = ConstantFoldingPass().run(ir)
        ir = DeadCodeEliminationPass().run(ir)
        stack_prog = IRToStack().lower(ir)
        
        vm = VM()
        result = vm.run(stack_prog)
        
        # Verify
        output_display = [int(x) if x == int(x) else x for x in result.outputs]
        expected_display = [int(x) if x == int(x) else x for x in expected]
        status = "✅" if result.outputs == expected else "❌"
        print(f"Output: {output_display} (expected: {expected_display}) {status}")
    
    print(f"\n{'─' * 60}")
    print("\n💡 Đây là full compiler pipeline hoàn chỉnh!")
    print("   Source text → Tokens → AST → IR → Optimized IR → Stack Code → Execution")
    print("   Tương đương: .c file → Clang → LLVM IR → opt → codegen → chạy binary")
    
    # Demo with verbose trace
    print("\n" + "=" * 70)
    print(" VERBOSE TRACE — Xem VM thực thi từng instruction")
    print("=" * 70)
    
    source = "let a = 10; let b = 20; print(a + b)"
    print(f"\nSource: {source}\n")
    
    ast = parse_source(source)
    ir = ASTToIR().lower(ast)
    ir = ConstantFoldingPass().run(ir)
    ir = DeadCodeEliminationPass().run(ir)
    stack_prog = IRToStack().lower(ir)
    
    print(stack_prog.dump("Stack Code"))
    print("\n--- Execution Trace ---")
    vm = VM(verbose=True)
    result = vm.run(stack_prog)
    print(f"\nFinal output: {result.outputs}")

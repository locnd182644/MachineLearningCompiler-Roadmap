"""
Constant Folding Pass — Optimization Pass #1
=============================================

CONSTANT FOLDING LÀ GÌ?
    Constant Folding là kỹ thuật tối ưu hóa đơn giản nhất và hiệu quả nhất:
    nếu cả hai toán hạng của một phép toán đều là HẰNG SỐ đã biết,
    thì tính kết quả ngay tại COMPILE TIME thay vì để runtime tính.

    Trước:                          Sau:
        %t0 = CONST 3                  %t4 = CONST 11
        %t1 = CONST 4
        %t2 = CONST 2
        %t3 = MUL %t1, %t2            (bị xóa bởi DCE sau)
        %t4 = ADD %t0, %t3            (bị xóa bởi DCE sau)
    
    "3 + 4 * 2" → "11" tại compile time!

TẠI SAO QUAN TRỌNG?
    1. Giảm số instruction phải chạy ở runtime
    2. Mở ra cơ hội cho các pass khác (DCE, strength reduction)
    3. Đặc biệt quan trọng trong AI compilers:
       - Shape inference thường dựa trên constant folding
       - Tensor shapes PHẢI biết tại compile time để allocate memory
       - XLA: kích thước tensor được fold ngay trong HLO graph

TRONG CÁC COMPILER THỰC TẾ:
    - LLVM: ConstantFolding.cpp — fold instructions có constant operands
    - MLIR: OpFoldResult + ConstantFolding pass 
      → Mỗi Op có thể implement fold() method
      → Trả về Attribute (constant) nếu fold thành công
    - GCC: fold-const.c — một trong những file lớn nhất (~15000 LOC)

FIXPOINT ITERATION:
    Constant folding có thể tạo ra cơ hội fold MỚI:
        %t0 = CONST 3      →  fold %t1 = CONST 8
        %t1 = CONST 8      →  fold %t2 = CONST 11
        %t2 = ADD %t0, %t1    (bây giờ cả hai operands đều là constant!)
    
    → Chạy lặp cho đến khi KHÔNG CÒN thay đổi nào (fixpoint).
    MLIR cũng dùng pattern này: GreedyPatternRewriteDriver.
"""

from __future__ import annotations
from typing import Dict, Optional
from ..ir import IRProgram, IRInstruction, IROpcode


class ConstantFoldingPass:
    """
    Constant Folding — tính toán biểu thức hằng tại compile time.
    
    Thuật toán:
    1. Xây dựng bảng constant_map: temp → giá trị (nếu temp là constant)
    2. Duyệt qua mỗi instruction:
       - Nếu CONST → ghi vào constant_map
       - Nếu binary op (ADD, SUB, MUL, DIV) mà CẢ HAI operands đều có
         trong constant_map → tính kết quả, thay instruction bằng CONST mới
       - Nếu NEG mà operand là constant → fold
       - Nếu COPY mà src là constant → fold
    3. Lặp lại cho đến khi không còn thay đổi (fixpoint)
    
    So sánh MLIR:
        - Mỗi Op có thể define fold() method
        - ConstantFolding pass gọi fold() cho mỗi op
        - Nếu fold trả về Attribute → op được thay bằng arith.constant
    """
    
    def __init__(self):
        self.constants: Dict[str, float] = {}
        self._changes_made = False
    
    def run(self, ir: IRProgram) -> IRProgram:
        """
        Chạy constant folding trên IRProgram.
        
        Lặp cho đến fixpoint — không còn fold được nữa.
        
        Args:
            ir: IRProgram cần optimize.
            
        Returns:
            IRProgram mới đã được fold.
        """
        result = ir.copy()
        
        # Fixpoint iteration
        iteration = 0
        while True:
            iteration += 1
            self._changes_made = False
            self.constants = {}
            
            result = self._fold_pass(result)
            
            if not self._changes_made:
                break
            
            # Safety: giới hạn số iteration (tránh infinite loop nếu bug)
            if iteration > 100:
                break
        
        return result
    
    def _fold_pass(self, ir: IRProgram) -> IRProgram:
        """Chạy một lần fold qua toàn bộ IR."""
        new_instructions = []
        
        for inst in ir.instructions:
            folded = self._try_fold(inst)
            new_instructions.append(folded)
        
        return IRProgram(instructions=new_instructions)
    
    def _try_fold(self, inst: IRInstruction) -> IRInstruction:
        """
        Thử fold một instruction. Trả về instruction mới nếu fold được,
        hoặc instruction gốc nếu không.
        """
        # CONST → ghi vào bảng constant
        if inst.opcode == IROpcode.CONST:
            if inst.dest is not None:
                self.constants[inst.dest] = inst.value
            return inst
        
        # COPY %src → nếu src là constant, fold thành CONST
        if inst.opcode == IROpcode.COPY:
            src_val = self.constants.get(inst.src1)
            if src_val is not None:
                self._changes_made = True
                folded = IRInstruction(
                    opcode=IROpcode.CONST,
                    dest=inst.dest,
                    value=src_val
                )
                if inst.dest is not None:
                    self.constants[inst.dest] = src_val
                return folded
            return inst
        
        # NEG %src → nếu src là constant, fold
        if inst.opcode == IROpcode.NEG:
            src_val = self.constants.get(inst.src1)
            if src_val is not None:
                result = -src_val
                self._changes_made = True
                folded = IRInstruction(
                    opcode=IROpcode.CONST,
                    dest=inst.dest,
                    value=result
                )
                if inst.dest is not None:
                    self.constants[inst.dest] = result
                return folded
            return inst
        
        # Binary ops: ADD, SUB, MUL, DIV
        if inst.opcode in (IROpcode.ADD, IROpcode.SUB, IROpcode.MUL, IROpcode.DIV):
            src1_val = self.constants.get(inst.src1)
            src2_val = self.constants.get(inst.src2)
            
            if src1_val is not None and src2_val is not None:
                # Cả hai operands đều là constant → fold!
                result = self._compute(inst.opcode, src1_val, src2_val)
                if result is not None:
                    self._changes_made = True
                    folded = IRInstruction(
                        opcode=IROpcode.CONST,
                        dest=inst.dest,
                        value=result
                    )
                    if inst.dest is not None:
                        self.constants[inst.dest] = result
                    return folded
            
            return inst
        
        # Các instruction khác (PRINT, NOP) → giữ nguyên
        return inst
    
    def _compute(self, opcode: IROpcode, a: float, b: float) -> Optional[float]:
        """
        Tính kết quả của phép toán binary trên 2 hằng số.
        
        Trả về None nếu không tính được (ví dụ: chia cho 0).
        """
        if opcode == IROpcode.ADD:
            return a + b
        elif opcode == IROpcode.SUB:
            return a - b
        elif opcode == IROpcode.MUL:
            return a * b
        elif opcode == IROpcode.DIV:
            if b == 0:
                return None  # Không fold chia cho 0 — để runtime error
            return a / b
        return None


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
    from toycalc.parser import parse_source
    from toycalc.ir import ASTToIR
    
    print("=" * 60)
    print(" CONSTANT FOLDING PASS — Tính toán tại compile time")
    print("=" * 60)
    
    source = "let x = 3 + 4 * 2; let y = x + 10; print(y)"
    print(f"\nSource: {source}\n")
    
    # Parse → AST → IR
    ast = parse_source(source)
    ir = ASTToIR().lower(ast)
    
    print(ir.dump("TRƯỚC Constant Folding"))
    
    # Run constant folding
    folder = ConstantFoldingPass()
    folded_ir = folder.run(ir)
    
    print()
    print(folded_ir.dump("SAU Constant Folding"))
    
    # Thống kê
    before_count = len(ir.instructions)
    after_consts = sum(1 for i in folded_ir.instructions if i.opcode == IROpcode.CONST)
    print(f"\n📊 Thống kê:")
    print(f"   Instructions trước: {before_count}")
    print(f"   Instructions sau:   {len(folded_ir.instructions)}")
    print(f"   Constants sau fold: {after_consts}")
    
    print()
    print("💡 Chú ý: 3 + 4 * 2 = 11 đã được tính tại COMPILE TIME!")
    print("   Tiếp theo: DCE pass sẽ xóa các CONST thừa không ai dùng.")
    
    # Demo 2: Cascading fold
    print("\n" + "=" * 60)
    source2 = "let a = 2 * 3; let b = a + 4; let c = b * 2; print(c)"
    print(f"\nSource: {source2}")
    print("(Cascading: 2*3=6, 6+4=10, 10*2=20 — tất cả fold tại compile time)\n")
    
    ast2 = parse_source(source2)
    ir2 = ASTToIR().lower(ast2)
    print(ir2.dump("TRƯỚC"))
    
    folded2 = ConstantFoldingPass().run(ir2)
    print()
    print(folded2.dump("SAU"))

"""
Dead Code Elimination (DCE) Pass — Optimization Pass #2
========================================================

DCE LÀ GÌ?
    Dead Code Elimination xóa bỏ các instructions mà KẾT QUẢ CỦA CHÚNG
    KHÔNG BAO GIỜ ĐƯỢC SỬ DỤNG. Code "chết" = code không ảnh hưởng output.

    Trước (sau constant folding):     Sau DCE:
        %t0 = CONST 3                    %t4 = CONST 11
        %t1 = CONST 4                    PRINT %t4
        %t2 = CONST 2
        %t3 = CONST 8   ← dead!
        %t4 = CONST 11
        PRINT %t4
    
    %t0, %t1, %t2, %t3 không ai dùng → XÓA!

TẠI SAO SSA LÀM DCE TRIVIAL?
    Trong SSA form:
    - Mỗi temp chỉ có ĐÚNG 1 definition (nơi nó được gán)
    - Use-def chains HIỂN NHIÊN: scan tất cả src1/src2 → biết ai dùng temp nào
    - Nếu một temp không xuất hiện trong BẤT KỲ use nào → definition của nó là dead
    
    Không có SSA thì sao?
    - Biến có thể bị gán lại nhiều lần
    - Phải dùng reaching definitions analysis (phức tạp hơn nhiều)
    - Đây là lý do SSA được dùng trong MỌI compiler hiện đại

TRONG CÁC COMPILER THỰC TẾ:
    - LLVM: DCE pass (lib/Transforms/Scalar/DCE.cpp)
      → Dùng isInstructionTriviallyDead() — kiểm tra:
        1. Instruction không có side effects (no memory write, no I/O)
        2. Kết quả không được dùng bởi instruction nào khác
    - MLIR: --mlir-opt --canonicalize thường bao gồm DCE
      → Mỗi pass có thể mark operations để DCE xóa
    - Aggressive DCE (ADCE): phức tạp hơn, xóa cả dead control flow

IMPORTANT:
    Instructions có SIDE EFFECT (như PRINT) KHÔNG BAO GIỜ bị xóa,
    dù kết quả không ai dùng. Side effects ảnh hưởng thế giới bên ngoài.
"""

from __future__ import annotations
from typing import Set
from ..ir import IRProgram, IRInstruction, IROpcode


class DeadCodeEliminationPass:
    """
    Dead Code Elimination — xóa instructions mà kết quả không ai dùng.
    
    Thuật toán (iterative worklist):
    1. Scan toàn bộ IR, xây dựng tập USED: tất cả temps xuất hiện trong
       src1 hoặc src2 của BẤT KỲ instruction nào
    2. Một instruction là DEAD nếu:
       a. Nó define một temp (dest != None)
       b. Temp đó KHÔNG nằm trong tập USED
       c. Instruction KHÔNG có side effects
    3. Xóa dead instructions
    4. Lặp lại (xóa 1 instruction có thể làm instruction khác trở thành dead)
    
    Ví dụ cascading:
        %t0 = CONST 3
        %t1 = ADD %t0, %t0   ← chỉ %t1 dùng %t0
        %t2 = MUL %t1, %t1   ← chỉ %t2 dùng %t1
        PRINT %t2
        
        Nếu %t2 không ai dùng (giả sử bỏ PRINT):
        → Xóa %t2 → %t1 không ai dùng → xóa %t1 → %t0 không ai dùng → xóa %t0
        → Cascading DCE!
    """
    
    def run(self, ir: IRProgram) -> IRProgram:
        """
        Chạy DCE trên IRProgram.
        
        Lặp cho đến khi không xóa thêm được instruction nào (fixpoint).
        
        Args:
            ir: IRProgram cần optimize.
            
        Returns:
            IRProgram mới đã loại bỏ dead code.
        """
        result = ir.copy()
        
        # Fixpoint iteration
        iteration = 0
        while True:
            iteration += 1
            
            # Bước 1: Xây dựng tập USED — tất cả temps được sử dụng
            used_temps = self._build_use_set(result)
            
            # Bước 2: Lọc bỏ dead instructions
            alive_instructions = []
            removed_count = 0
            
            for inst in result.instructions:
                if self._is_dead(inst, used_temps):
                    removed_count += 1
                else:
                    alive_instructions.append(inst)
            
            result = IRProgram(instructions=alive_instructions)
            
            # Nếu không xóa được gì → đã đạt fixpoint
            if removed_count == 0:
                break
            
            # Safety limit
            if iteration > 100:
                break
        
        return result
    
    def _build_use_set(self, ir: IRProgram) -> Set[str]:
        """
        Xây dựng tập hợp tất cả temps được SỬ DỤNG (read from).
        
        Scan qua src1 và src2 của mọi instruction.
        Temp nào xuất hiện ở đây → đang được dùng → KHÔNG phải dead code.
        
        Đây chính là bước xây dựng use-def chains trong SSA analysis.
        Vì SSA, mỗi use chỉ đến đúng 1 def → analysis rất đơn giản.
        """
        used: Set[str] = set()
        
        for inst in ir.instructions:
            # Collect tất cả temps được đọc
            for temp in inst.uses():
                used.add(temp)
        
        return used
    
    def _is_dead(self, inst: IRInstruction, used_temps: Set[str]) -> bool:
        """
        Kiểm tra instruction có phải dead code không.
        
        Dead = define một temp mà KHÔNG AI DÙNG + KHÔNG CÓ side effects.
        
        Lưu ý: NOP luôn bị coi là dead (trừ khi ai đó override has_side_effect).
        """
        # Instructions có side effects KHÔNG BAO GIỜ dead
        if inst.has_side_effect():
            return False
        
        # NOP luôn dead
        if inst.opcode == IROpcode.NOP:
            return True
        
        # Instruction không define temp nào → không dead (e.g., bare PRINT)
        dest = inst.defines()
        if dest is None:
            return False
        
        # Nếu temp mà instruction define KHÔNG có trong tập used → dead!
        return dest not in used_temps


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
    from toycalc.parser import parse_source
    from toycalc.ir import ASTToIR
    from toycalc.passes.const_fold import ConstantFoldingPass
    
    print("=" * 60)
    print(" DEAD CODE ELIMINATION — Xóa code không ai dùng")
    print("=" * 60)
    
    # Demo 1: Dead variable sau constant folding
    source = "let x = 3 + 4 * 2; let unused = 99; print(x)"
    print(f"\nSource: {source}")
    print("('unused' không được dùng → dead code)\n")
    
    ast = parse_source(source)
    ir = ASTToIR().lower(ast)
    print(ir.dump("IR gốc"))
    
    # Chạy constant folding trước
    folded = ConstantFoldingPass().run(ir)
    print()
    print(folded.dump("Sau Constant Folding"))
    
    # Chạy DCE
    dce = DeadCodeEliminationPass()
    cleaned = dce.run(folded)
    print()
    print(cleaned.dump("Sau DCE"))
    
    print(f"\n📊 Thống kê:")
    print(f"   Instructions gốc:      {len(ir.instructions)}")
    print(f"   Sau constant folding:   {len(folded.instructions)}")
    print(f"   Sau DCE:                {len(cleaned.instructions)}")
    print(f"   Đã loại bỏ:            {len(ir.instructions) - len(cleaned.instructions)}")
    
    # Demo 2: Cascading DCE
    print("\n" + "=" * 60)
    source2 = "let a = 1 + 2; let b = a * 3; let c = 100; print(c)"
    print(f"\nSource: {source2}")
    print("(a, b đều dead vì chỉ c được in)\n")
    
    ast2 = parse_source(source2)
    ir2 = ASTToIR().lower(ast2)
    print(ir2.dump("IR gốc"))
    
    folded2 = ConstantFoldingPass().run(ir2)
    cleaned2 = dce.run(folded2)
    print()
    print(cleaned2.dump("Sau Fold + DCE"))
    print(f"\n💡 Cascading: a → b đều bị xóa vì không ai dùng!")

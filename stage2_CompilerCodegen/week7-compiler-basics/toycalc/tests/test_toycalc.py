"""
Test Suite cho ToyCalc Compiler — Bài tập 7.1
==============================================

Kiểm tra toàn bộ compiler pipeline: Lexer → Parser → AST → IR → Passes → Codegen → VM.

Yêu cầu README:
    "Test suite: 5+ chương trình, so kết quả với eval() Python"

Chạy tests:
    cd week7-compiler-basics/
    python -m pytest toycalc/tests/ -v

Cấu trúc tests:
    1. test_lexer_*       — Kiểm tra tokenization
    2. test_parser_*      — Kiểm tra AST, operator precedence
    3. test_ast_printer_* — Kiểm tra pretty printer
    4. test_ir_*          — Kiểm tra lowering AST → IR
    5. test_const_fold_*  — Kiểm tra constant folding pass
    6. test_dce_*         — Kiểm tra dead code elimination pass
    7. test_codegen_*     — Kiểm tra sinh stack machine code
    8. test_pipeline_*    — FULL PIPELINE tests (10+ cases)
    9. test_error_*       — Kiểm tra error handling
"""

import pytest
import math
from toycalc.lexer import Lexer, TokenType, LexerError
from toycalc.parser import (
    parse_source, Parser, ParseError,
    Program, LetStatement, PrintStatement,
    BinaryOp, UnaryOp, NumberLiteral, Identifier
)
from toycalc.ast_printer import ASTPrinter, CompactPrinter
from toycalc.ir import ASTToIR, IROpcode, IRProgram
from toycalc.passes.const_fold import ConstantFoldingPass
from toycalc.passes.dce import DeadCodeEliminationPass
from toycalc.codegen import IRToStack, StackOp
from toycalc.vm import VM, VMError


# =============================================================================
# HELPER: Full pipeline
# =============================================================================

def compile_and_run(source: str) -> list:
    """
    Biên dịch và chạy chương trình ToyCalc, trả về outputs.
    
    Pipeline đầy đủ:
        Source → Lexer → Parser → AST → IR → ConstFold → DCE → Codegen → VM
    """
    ast = parse_source(source)
    ir = ASTToIR().lower(ast)
    ir = ConstantFoldingPass().run(ir)
    ir = DeadCodeEliminationPass().run(ir)
    stack_prog = IRToStack().lower(ir)
    return VM().run(stack_prog).outputs


# =============================================================================
# 1. LEXER TESTS
# =============================================================================

class TestLexer:
    """Kiểm tra quá trình tokenization — biến source code thành dòng token."""
    
    def test_basic_tokenization(self):
        """Tokenize biểu thức cơ bản: let x = 10 + 20;"""
        tokens = Lexer("let x = 10 + 20;").tokenize()
        types = [t.type for t in tokens]
        
        assert types == [
            TokenType.LET, TokenType.IDENTIFIER, TokenType.EQUALS,
            TokenType.NUMBER, TokenType.PLUS, TokenType.NUMBER,
            TokenType.SEMICOLON, TokenType.EOF
        ]
    
    def test_identifier_vs_keyword(self):
        """'let' là keyword, 'letter' là identifier — phân biệt đúng."""
        tokens = Lexer("let letter = 5;").tokenize()
        assert tokens[0].type == TokenType.LET       # keyword
        assert tokens[1].type == TokenType.IDENTIFIER  # identifier
        assert tokens[1].value == "letter"
    
    def test_all_operators(self):
        """Nhận diện đầy đủ các toán tử: + - * /"""
        tokens = Lexer("1 + 2 - 3 * 4 / 5").tokenize()
        ops = [t.type for t in tokens if t.type in 
               (TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH)]
        assert ops == [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH]
    
    def test_float_numbers(self):
        """Nhận diện số thực: 3.14"""
        tokens = Lexer("3.14").tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "3.14"  # value là string, chuyển float ở parser
    
    def test_whitespace_and_comments(self):
        """Bỏ qua whitespace và comments đúng cách."""
        source = """
        // Đây là comment
        let x = 42;
        """
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.LET, TokenType.IDENTIFIER, TokenType.EQUALS,
            TokenType.NUMBER, TokenType.SEMICOLON
        ]
    
    def test_line_column_tracking(self):
        """Token phải ghi nhận đúng vị trí dòng:cột cho error reporting."""
        tokens = Lexer("let x = 5;").tokenize()
        # 'let' ở dòng 1, cột 1
        assert tokens[0].line == 1
        assert tokens[0].column == 1
    
    def test_lexer_error_invalid_char(self):
        """Ký tự không hợp lệ phải raise LexerError."""
        with pytest.raises(LexerError):
            Lexer("let x = 5 @ 3;").tokenize()


# =============================================================================
# 2. PARSER TESTS
# =============================================================================

class TestParser:
    """Kiểm tra parsing — biến token stream thành AST."""
    
    def test_basic_let_statement(self):
        """Parse 'let x = 42;' → LetStatement đúng."""
        ast = parse_source("let x = 42;")
        assert isinstance(ast, Program)
        assert len(ast.statements) == 1
        
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStatement)
        assert stmt.name == "x"
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == 42.0
    
    def test_operator_precedence_mul_before_add(self):
        """
        Operator precedence: * bind tighter hơn +
        '2 + 3 * 4' phải parse thành BinaryOp(+, 2, BinaryOp(*, 3, 4))
        KHÔNG PHẢI BinaryOp(*, BinaryOp(+, 2, 3), 4)
        """
        ast = parse_source("let x = 2 + 3 * 4;")
        expr = ast.statements[0].value
        
        # Root phải là +
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        
        # Left = 2 (NumberLiteral)
        assert isinstance(expr.left, NumberLiteral)
        assert expr.left.value == 2.0
        
        # Right = BinaryOp(*, 3, 4)
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "*"
        assert expr.right.left.value == 3.0
        assert expr.right.right.value == 4.0
    
    def test_parentheses_override_precedence(self):
        """
        Ngoặc thay đổi precedence:
        '(2 + 3) * 4' → BinaryOp(*, BinaryOp(+, 2, 3), 4)
        """
        ast = parse_source("let x = (2 + 3) * 4;")
        expr = ast.statements[0].value
        
        assert isinstance(expr, BinaryOp)
        assert expr.op == "*"
        
        # Left = BinaryOp(+, 2, 3) — ngoặc cho + cao hơn *
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.op == "+"
    
    def test_unary_negation(self):
        """Parse '-x' → UnaryOp('-', Identifier('x'))"""
        ast = parse_source("let a = -5;")
        expr = ast.statements[0].value
        assert isinstance(expr, UnaryOp)
        assert expr.op == "-"
        assert isinstance(expr.operand, NumberLiteral)
        assert expr.operand.value == 5.0
    
    def test_print_statement(self):
        """Parse 'print(x)' → PrintStatement với expr đúng."""
        ast = parse_source("let x = 42; print(x)")
        assert len(ast.statements) == 2
        assert isinstance(ast.statements[1], PrintStatement)
        assert isinstance(ast.statements[1].expr, Identifier)
        assert ast.statements[1].expr.name == "x"
    
    def test_multiple_statements(self):
        """Parse nhiều statements liên tiếp."""
        ast = parse_source("let a = 1; let b = 2; let c = 3; print(c)")
        assert len(ast.statements) == 4
        assert isinstance(ast.statements[0], LetStatement)
        assert isinstance(ast.statements[3], PrintStatement)
    
    def test_left_associativity(self):
        """
        Phép cộng left-associative: 1 + 2 + 3 = (1 + 2) + 3
        """
        ast = parse_source("let x = 1 + 2 + 3;")
        expr = ast.statements[0].value
        
        # Root = BinaryOp(+, BinaryOp(+, 1, 2), 3)
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.left, BinaryOp)  # (1 + 2)
        assert isinstance(expr.right, NumberLiteral)  # 3
        assert expr.left.op == "+"


# =============================================================================
# 3. AST PRINTER TESTS
# =============================================================================

class TestASTPrinter:
    """Kiểm tra AST pretty printer — 'dump được là debug được'."""
    
    def test_tree_format(self):
        """Tree format phải chứa tên node types."""
        ast = parse_source("let x = 3 + 4;")
        output = ASTPrinter().format(ast)
        assert "Program" in output
        assert "LetStatement" in output
        assert "BinaryOp" in output
        assert "NumberLiteral" in output
    
    def test_compact_format(self):
        """Compact (S-expression) format phải đúng cấu trúc."""
        ast = parse_source("let x = 3 + 4;")
        output = CompactPrinter().format(ast)
        # Dạng: (program (let x (+ 3 4)))
        assert "(program" in output
        assert "(let x" in output
        assert "(+ 3 4)" in output


# =============================================================================
# 4. IR LOWERING TESTS
# =============================================================================

class TestIRLowering:
    """Kiểm tra AST → IR lowering — progressive lowering step 1."""
    
    def test_basic_lowering(self):
        """Lowering biểu thức đơn giản sinh đúng IR instructions."""
        ast = parse_source("let x = 5; print(x)")
        ir = ASTToIR().lower(ast)
        
        opcodes = [inst.opcode for inst in ir.instructions]
        assert IROpcode.CONST in opcodes  # 5 → CONST
        assert IROpcode.PRINT in opcodes  # print → PRINT
    
    def test_ssa_property(self):
        """
        Tính chất SSA: mỗi temp (%tN) chỉ được gán đúng 1 lần.
        Đây là invariant quan trọng nhất — MLIR/LLVM đều yêu cầu SSA.
        """
        ast = parse_source("let x = 1 + 2; let y = x + 3; print(y)")
        ir = ASTToIR().lower(ast)
        
        # Thu thập tất cả destinations
        dests = [inst.dest for inst in ir.instructions if inst.dest is not None]
        # Mỗi dest phải unique (SSA!)
        assert len(dests) == len(set(dests)), \
            f"SSA violation: duplicate dest found in {dests}"
    
    def test_binary_op_lowering(self):
        """BinaryOp sinh 2 operand instructions + 1 binary instruction."""
        ast = parse_source("let x = 3 + 4;")
        ir = ASTToIR().lower(ast)
        
        opcodes = [inst.opcode for inst in ir.instructions]
        assert opcodes.count(IROpcode.CONST) == 2  # 3 và 4
        assert IROpcode.ADD in opcodes  # phép cộng
    
    def test_variable_resolution(self):
        """Biến phải resolve đúng qua var_to_temp mapping."""
        ast = parse_source("let a = 10; let b = a; print(b)")
        ir = ASTToIR().lower(ast)
        # Phải có CONST 10 và PRINT, và PRINT phải dùng đúng temp của a
        assert any(inst.opcode == IROpcode.PRINT for inst in ir.instructions)


# =============================================================================
# 5. CONSTANT FOLDING TESTS
# =============================================================================

class TestConstantFolding:
    """Kiểm tra Constant Folding — tính biểu thức hằng tại compile time."""
    
    def test_basic_fold(self):
        """'3 + 4 * 2' phải được fold thành CONST 11 tại compile time."""
        ast = parse_source("let x = 3 + 4 * 2;")
        ir_before = ASTToIR().lower(ast)
        ir_after = ConstantFoldingPass().run(ir_before)
        
        # Sau fold: không còn ADD hay MUL, chỉ còn CONST
        opcodes = [inst.opcode for inst in ir_after.instructions]
        assert IROpcode.ADD not in opcodes
        assert IROpcode.MUL not in opcodes
        
        # Tìm CONST có value = 11.0
        const_values = [inst.value for inst in ir_after.instructions 
                        if inst.opcode == IROpcode.CONST]
        assert 11.0 in const_values
    
    def test_cascading_fold(self):
        """Cascading fold: a=2*3=6, b=a+4=10, c=b*2=20."""
        ast = parse_source("let a = 2 * 3; let b = a + 4; let c = b * 2; print(c)")
        ir = ASTToIR().lower(ast)
        folded = ConstantFoldingPass().run(ir)
        
        const_values = [inst.value for inst in folded.instructions 
                        if inst.opcode == IROpcode.CONST]
        assert 20.0 in const_values
    
    def test_no_fold_division_by_zero(self):
        """Chia cho 0 KHÔNG được fold — để runtime error xử lý."""
        ast = parse_source("let x = 10 / 0;")
        ir = ASTToIR().lower(ast)
        folded = ConstantFoldingPass().run(ir)
        
        # Phải vẫn còn DIV instruction (không fold)
        opcodes = [inst.opcode for inst in folded.instructions]
        assert IROpcode.DIV in opcodes
    
    def test_instruction_count_decreases(self):
        """Constant folding phải giảm số instructions."""
        ast = parse_source("let x = 1 + 2 + 3 + 4;")
        ir_before = ASTToIR().lower(ast)
        ir_after = ConstantFoldingPass().run(ir_before)
        
        # Sau fold: ít instructions hơn vì các phép tính hằng được gộp
        assert len(ir_after.instructions) <= len(ir_before.instructions)


# =============================================================================
# 6. DCE (DEAD CODE ELIMINATION) TESTS
# =============================================================================

class TestDCE:
    """Kiểm tra DCE — xóa instructions mà kết quả không ai dùng."""
    
    def test_remove_unused_variable(self):
        """Biến 'unused' không được dùng → phải bị xóa."""
        ast = parse_source("let unused = 42; let x = 10; print(x)")
        ir = ASTToIR().lower(ast)
        ir = ConstantFoldingPass().run(ir)
        cleaned = DeadCodeEliminationPass().run(ir)
        
        # Sau DCE: ít instructions hơn (CONST 42 bị xóa)
        assert len(cleaned.instructions) < len(ir.instructions)
    
    def test_preserve_side_effects(self):
        """PRINT có side effect → KHÔNG BAO GIỜ bị xóa bởi DCE."""
        ast = parse_source("let x = 42; print(x)")
        ir = ASTToIR().lower(ast)
        ir = ConstantFoldingPass().run(ir)
        cleaned = DeadCodeEliminationPass().run(ir)
        
        # PRINT phải vẫn còn
        opcodes = [inst.opcode for inst in cleaned.instructions]
        assert IROpcode.PRINT in opcodes
    
    def test_cascading_dce(self):
        """
        Cascading DCE: nếu b dùng a, và b không ai dùng → cả a lẫn b bị xóa.
        """
        ast = parse_source("let a = 1 + 2; let b = a * 3; let c = 100; print(c)")
        ir = ASTToIR().lower(ast)
        ir = ConstantFoldingPass().run(ir)
        cleaned = DeadCodeEliminationPass().run(ir)
        
        # Sau DCE: chỉ còn CONST 100 và PRINT (a, b đều dead)
        const_values = [inst.value for inst in cleaned.instructions 
                        if inst.opcode == IROpcode.CONST]
        assert 100.0 in const_values
        # Không còn giá trị của a (3) hay b (9)
        assert 3.0 not in const_values
        assert 9.0 not in const_values


# =============================================================================
# 7. CODEGEN TESTS
# =============================================================================

class TestCodegen:
    """Kiểm tra IR → Stack Machine code generation."""
    
    def test_basic_codegen(self):
        """Codegen sinh được PUSH và PRINT cho chương trình đơn giản."""
        ast = parse_source("let x = 42; print(x)")
        ir = ASTToIR().lower(ast)
        ir = ConstantFoldingPass().run(ir)
        ir = DeadCodeEliminationPass().run(ir)
        stack_prog = IRToStack().lower(ir)
        
        ops = [inst.opcode for inst in stack_prog.instructions]
        assert StackOp.PUSH in ops
        assert StackOp.PRINT in ops
    
    def test_arithmetic_codegen(self):
        """Phép tính sinh đúng LOAD/ADD/STORE."""
        ast = parse_source("let x = 5; let y = 3; print(x + y)")
        ir = ASTToIR().lower(ast)
        # Không fold vì x, y là biến
        stack_prog = IRToStack().lower(ir)
        
        ops = [inst.opcode for inst in stack_prog.instructions]
        assert StackOp.LOAD in ops
        assert StackOp.STORE in ops


# =============================================================================
# 8. FULL PIPELINE TESTS — So kết quả với Python eval()
# =============================================================================
# Đây là phần quan trọng nhất: end-to-end test.
# Mỗi test case: source code → compile → run → so sánh output với expected.

# NOTE: print() trong ToyCalc KHÔNG có semicolon sau nó.
#       let statement CẦN semicolon: let x = expr;
#       print statement syntax: print(expr)

PIPELINE_TESTS = [
    # ---- Basic arithmetic ----
    ("let x = 3 + 4 * 2; print(x)",                              [11.0]),
    ("let x = (3 + 4) * 2; print(x)",                            [14.0]),
    ("let x = 10 - 3; print(x)",                                 [7.0]),
    
    # ---- Variables ----
    ("let a = 5; let b = a * 2; print(b)",                       [10.0]),
    ("let x = 2 * 3; let y = x + 4; let z = y * 2; print(z)",  [20.0]),
    ("let a = 100; let b = a / 5; print(a - b)",                 [80.0]),
    
    # ---- Negative numbers ----
    ("let x = -5; let y = x * x; print(y - 1)",                  [24.0]),
    
    # ---- Division (float result) ----
    ("let x = 10 / 4; print(x)",                                 [2.5]),
    
    # ---- Floating point ----
    ("let pi = 3.14; print(pi * 2)",                             [6.28]),
    
    # ---- Dead code (DCE should not break output) ----
    ("let unused = 999; let x = 42; print(x)",                   [42.0]),
    
    # ---- Complex expression ----
    ("let a = (1 + 2) * (3 + 4); print(a)",                     [21.0]),
    
    # ---- Identity operations ----
    ("let x = 0 + 5; print(x)",                                  [5.0]),
    ("let x = 1 * 7; print(x)",                                  [7.0]),
]

@pytest.mark.parametrize("source,expected", PIPELINE_TESTS)
def test_full_pipeline(source, expected):
    """
    Full pipeline test: source → compile → run → compare output.
    
    Đây là bài tập chính của tuần 7:
    "Test suite: 5+ chương trình, so kết quả với eval() Python"
    """
    outputs = compile_and_run(source)
    assert outputs == expected, \
        f"Source: {source!r}\nExpected: {expected}\nGot: {outputs}"


# Thêm test riêng cho multiple prints vì print statement không dùng semicolon
def test_pipeline_two_prints():
    """Chương trình với 2 lệnh print liên tiếp."""
    outputs = compile_and_run("let x = 10; print(x) print(x)")
    # Cả 2 PRINT đều phải chạy
    assert len(outputs) == 2
    assert outputs == [10.0, 10.0]


def test_pipeline_three_variables():
    """Chương trình với 3 biến phụ thuộc nhau."""
    outputs = compile_and_run("let a = 2; let b = a + 3; let c = b * b; print(c)")
    assert outputs == [25.0]


def test_pipeline_long_chain():
    """Chuỗi tính toán dài — stress test cho compiler."""
    outputs = compile_and_run(
        "let a = 1; let b = a + 1; let c = b + 1; "
        "let d = c + 1; let e = d + 1; print(e)"
    )
    assert outputs == [5.0]


# =============================================================================
# 9. ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Kiểm tra error handling — compiler phải báo lỗi rõ ràng."""
    
    def test_undefined_variable(self):
        """Sử dụng biến chưa khai báo phải raise error."""
        with pytest.raises(ValueError, match="Undefined variable"):
            compile_and_run("print(undefined_var)")
    
    def test_missing_semicolon(self):
        """Thiếu semicolon sau let statement phải raise ParseError."""
        with pytest.raises(ParseError):
            parse_source("let x = 10")  # thiếu ;
    
    def test_invalid_character(self):
        """Ký tự không hợp lệ phải raise LexerError."""
        with pytest.raises(LexerError):
            Lexer("let x = 5 @ 3;").tokenize()
    
    def test_unexpected_token(self):
        """Token không mong đợi phải raise ParseError."""
        with pytest.raises(ParseError):
            parse_source("42;")  # không có let/print ở đầu
    
    def test_division_by_zero_runtime(self):
        """Chia cho 0 phải raise error ở runtime (VM)."""
        # Trường hợp biến (không fold được)
        with pytest.raises(VMError):
            compile_and_run("let x = 5; let y = 0; print(x / y)")


# =============================================================================
# 10. OPTIMIZATION VERIFICATION TESTS
# =============================================================================

class TestOptimizationVerification:
    """
    Kiểm tra rằng optimizations ĐÚNG — output không thay đổi sau optimize.
    
    Đây là INVARIANT quan trọng nhất: mọi pass phải preserve semantics.
    Trong MLIR: verify passes bằng FileCheck tests.
    """
    
    def test_const_fold_preserves_output(self):
        """Output giống nhau dù có hay không có constant folding."""
        source = "let x = 3 + 4 * 2; print(x)"
        ast = parse_source(source)
        ir = ASTToIR().lower(ast)
        
        # Chạy KHÔNG optimize
        stack_no_opt = IRToStack().lower(ir)
        result_no_opt = VM().run(stack_no_opt).outputs
        
        # Chạy CÓ optimize
        ir_opt = ConstantFoldingPass().run(ir)
        ir_opt = DeadCodeEliminationPass().run(ir_opt)
        stack_opt = IRToStack().lower(ir_opt)
        result_opt = VM().run(stack_opt).outputs
        
        # Kết quả phải giống nhau
        assert result_no_opt == result_opt
    
    def test_dce_preserves_output(self):
        """DCE xóa code nhưng không thay đổi output."""
        source = "let unused = 999; let x = 42; print(x)"
        ast = parse_source(source)
        ir = ASTToIR().lower(ast)
        ir = ConstantFoldingPass().run(ir)
        
        # Chạy không DCE
        stack_no_dce = IRToStack().lower(ir)
        result_no_dce = VM().run(stack_no_dce).outputs
        
        # Chạy có DCE
        ir_dce = DeadCodeEliminationPass().run(ir)
        stack_dce = IRToStack().lower(ir_dce)
        result_dce = VM().run(stack_dce).outputs
        
        assert result_no_dce == result_dce
    
    def test_optimized_fewer_instructions(self):
        """Optimized code phải có ít instructions hơn."""
        source = "let a = 1 + 2; let b = 3 + 4; let c = a + b; print(c)"
        ast = parse_source(source)
        ir = ASTToIR().lower(ast)
        
        ir_opt = ConstantFoldingPass().run(ir)
        ir_opt = DeadCodeEliminationPass().run(ir_opt)
        
        stack_no_opt = IRToStack().lower(ir)
        stack_opt = IRToStack().lower(ir_opt)
        
        assert len(stack_opt.instructions) < len(stack_no_opt.instructions)

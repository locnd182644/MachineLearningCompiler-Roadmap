"""
Parser + AST — Phase 2 của Compiler Pipeline
==============================================

PARSER LÀ GÌ?
    Parser nhận dòng token từ Lexer và xây dựng Abstract Syntax Tree (AST).
    AST biểu diễn CẤU TRÚC của chương trình, bỏ đi chi tiết cú pháp
    (dấu ngoặc, dấu chấm phẩy, ...).

    Token stream: [LET, ID(x), EQ, NUM(3), PLUS, NUM(4), STAR, NUM(2), SEMI]
    
    AST:
        LetStatement
        ├── name: "x"
        └── value: BinaryOp(+)
                   ├── left:  NumberLiteral(3)
                   └── right: BinaryOp(*)
                              ├── left:  NumberLiteral(4)
                              └── right: NumberLiteral(2)

KỸ THUẬT PARSING:
    Chúng ta dùng Recursive Descent Parser — kỹ thuật phổ biến nhất
    cho hand-written parsers (GCC, Clang, Go compiler đều dùng).
    
    Mỗi non-terminal trong grammar → 1 method trong parser.
    Đệ quy tương hỗ giữa các method tạo ra cây AST.

GRAMMAR CỦA TOYCALC (BNF):
    program    → statement* EOF
    statement  → let_stmt | print_stmt
    let_stmt   → 'let' IDENTIFIER '=' expression ';'
    print_stmt → 'print' '(' expression ')'
    expression → term (('+' | '-') term)*
    term       → factor (('*' | '/') factor)*
    factor     → ('-')? primary
    primary    → NUMBER | IDENTIFIER | '(' expression ')'

    Operator precedence (thấp → cao):
        +, -  (additive)
        *, /  (multiplicative)
        -     (unary)
        ()    (grouping)

TẠI SAO AI COMPILER KHÔNG CẦN PARSER TRUYỀN THỐNG?
    - PyTorch FX: dùng torch.fx.Tracer — chạy Python code, ghi lại ops
    - ract values, recorJAX: dùng tracing — feed abstd transformations  
    - Triton: phân tích Python AST trực tiếp (ast module)
    → Tất cả đều "capture" computation graph thay vì parse text
    → Nhưng MLIR assembly format VẪN CẦN parser (mlir/lib/AsmParser/)
    → Hiểu parsing giúp hiểu cách đọc/viết .mlir files
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any
from .lexer import Token, TokenType, Lexer


# =============================================================================
# AST NODE CLASSES
# =============================================================================
# AST = Abstract Syntax Tree = biểu diễn cây của chương trình.
#
# "Abstract" vì bỏ đi chi tiết cú pháp (dấu ngoặc, semicolons, keywords).
# Chỉ giữ lại NGỮ NGHĨA: "gán x bằng 3 cộng 4 nhân 2".
#
# Mỗi node class tương ứng một construct trong ngôn ngữ.
# Visitor pattern cho phép traverse AST mà không sửa node classes.
#
# So sánh với AI compilers:
#   - PyTorch FX: Node class (op, target, args, kwargs)
#   - MLIR: Operation (dialect.opname, operands, results, attributes, regions)
#   - XLA HLO: HloInstruction (opcode, operands, shape)
# =============================================================================

@dataclass
class ASTNode:
    """Base class cho mọi AST node. Hỗ trợ Visitor pattern."""
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        """Dispatch visitor method dựa trên type của node."""
        method_name = f'visit_{type(self).__name__}'
        visitor_method = getattr(visitor, method_name, None)
        if visitor_method is None:
            raise NotImplementedError(
                f"Visitor {type(visitor).__name__} missing method {method_name}"
            )
        return visitor_method(self)


@dataclass
class NumberLiteral(ASTNode):
    """Hằng số: 42, 3.14"""
    value: float = 0.0


@dataclass
class Identifier(ASTNode):
    """Tên biến: x, result"""
    name: str = ""


@dataclass 
class BinaryOp(ASTNode):
    """
    Phép toán hai ngôi: left OP right
    
    Ví dụ: 3 + 4 → BinaryOp('+', NumberLiteral(3), NumberLiteral(4))
    
    So sánh:
        MLIR: %result = arith.addf %lhs, %rhs : f32
        XLA:  HloInstruction::CreateBinary(HloOpcode::kAdd, lhs, rhs)
    """
    op: str = ""
    left: ASTNode = None
    right: ASTNode = None


@dataclass
class UnaryOp(ASTNode):
    """Phép toán một ngôi: -x"""
    op: str = ""
    operand: ASTNode = None


@dataclass
class LetStatement(ASTNode):
    """
    Gán biến: let x = expression;
    
    Trong SSA (tuần sau): mỗi let tạo một definition mới.
    MLIR tương đương: %x = "some.op"(...) : type
    """
    name: str = ""
    value: ASTNode = None


@dataclass
class PrintStatement(ASTNode):
    """In giá trị: print(expression)"""
    expr: ASTNode = None


@dataclass
class Program(ASTNode):
    """
    Chương trình = danh sách statements.
    
    Đây là root node của AST. Tương đương:
        MLIR: builtin.module { ... }
        XLA:  HloModule
    """
    statements: List[ASTNode] = field(default_factory=list)


# =============================================================================
# AST VISITOR BASE CLASS
# =============================================================================

class ASTVisitor:
    """
    Base class cho Visitor pattern.
    
    Visitor pattern là DESIGN PATTERN QUAN TRỌNG NHẤT trong compiler:
    - Tách THUẬT TOÁN (visitor) khỏi CẤU TRÚC DỮ LIỆU (AST nodes)
    - Thêm pass mới = thêm visitor mới, KHÔNG SỬA AST classes
    - MLIR dùng tương tự: OpVisitor, RewritePattern
    
    Mỗi concrete visitor override visit_XYZ methods.
    """
    
    def visit_NumberLiteral(self, node: NumberLiteral) -> Any:
        raise NotImplementedError
    
    def visit_Identifier(self, node: Identifier) -> Any:
        raise NotImplementedError
    
    def visit_BinaryOp(self, node: BinaryOp) -> Any:
        raise NotImplementedError
    
    def visit_UnaryOp(self, node: UnaryOp) -> Any:
        raise NotImplementedError
    
    def visit_LetStatement(self, node: LetStatement) -> Any:
        raise NotImplementedError
    
    def visit_PrintStatement(self, node: PrintStatement) -> Any:
        raise NotImplementedError
    
    def visit_Program(self, node: Program) -> Any:
        raise NotImplementedError


# =============================================================================
# PARSER CLASS — Recursive Descent
# =============================================================================

class ParseError(Exception):
    """Lỗi cú pháp trong quá trình parsing."""
    def __init__(self, message: str, token: Token):
        self.token = token
        super().__init__(
            f"Parse error at L{token.line}:{token.column}: {message} "
            f"(got {token.type.name} '{token.value}')"
        )


class Parser:
    """
    Recursive Descent Parser cho ToyCalc.
    
    Kỹ thuật: LL(1) — đọc trái-sang-phải, dẫn xuất trái, lookahead 1 token.
    
    Mỗi method parse_XXX tương ứng một production rule trong grammar:
        parse_program()    → program
        parse_statement()  → statement  
        parse_expression() → expression (xử lý +, -)
        parse_term()       → term (xử lý *, /)
        parse_factor()     → factor (xử lý unary -)
        parse_primary()    → primary (number, identifier, parenthesized expr)
    
    Operator precedence được encode vào CẤU TRÚC ĐỆ QUY:
        expression gọi term gọi factor gọi primary
        → * được parse ở mức sâu hơn + → bind tighter
    
    So sánh:
        - Clang dùng kỹ thuật tương tự cho C++ (phức tạp hơn nhiều)
        - MLIR assembly parser (mlir/lib/AsmParser/Parser.cpp) cũng recursive descent
        - Python parser mới dùng PEG (Parsing Expression Grammar)
    """
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    # -------------------------------------------------------------------------
    # Token navigation helpers
    # -------------------------------------------------------------------------
    
    def _current(self) -> Token:
        """Token hiện tại."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF
    
    def _peek(self) -> Token:
        """Nhìn token tiếp theo (lookahead = 1)."""
        next_pos = self.pos + 1
        if next_pos < len(self.tokens):
            return self.tokens[next_pos]
        return self.tokens[-1]
    
    def _advance(self) -> Token:
        """Đọc token hiện tại và tiến sang token tiếp theo."""
        token = self._current()
        self.pos += 1
        return token
    
    def _expect(self, token_type: TokenType) -> Token:
        """
        Đọc token hiện tại, kiểm tra đúng type. Nếu sai → ParseError.
        
        Đây là pattern phổ biến trong recursive descent:
        "Tôi MONG ĐỢI token này ở đây, nếu không có thì lỗi cú pháp."
        """
        token = self._current()
        if token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {token.type.name}",
                token
            )
        return self._advance()
    
    def _match(self, *token_types: TokenType) -> Optional[Token]:
        """Nếu token hiện tại match bất kỳ type nào → advance và trả về. Nếu không → None."""
        if self._current().type in token_types:
            return self._advance()
        return None
    
    # -------------------------------------------------------------------------
    # Grammar rules — mỗi method = 1 production rule
    # -------------------------------------------------------------------------
    
    def parse(self) -> Program:
        """
        Entry point. Parse toàn bộ chương trình.
        
        Grammar: program → statement* EOF
        """
        return self.parse_program()
    
    def parse_program(self) -> Program:
        """program → statement* EOF"""
        statements = []
        while self._current().type != TokenType.EOF:
            stmt = self.parse_statement()
            statements.append(stmt)
        return Program(statements=statements)
    
    def parse_statement(self) -> ASTNode:
        """
        statement → let_stmt | print_stmt
        
        Nhìn token đầu tiên để quyết định parse rule nào (predictive parsing).
        """
        current = self._current()
        
        if current.type == TokenType.LET:
            return self.parse_let_statement()
        elif current.type == TokenType.PRINT:
            return self.parse_print_statement()
        else:
            raise ParseError(
                "Expected 'let' or 'print' at start of statement",
                current
            )
    
    def parse_let_statement(self) -> LetStatement:
        """
        let_stmt → 'let' IDENTIFIER '=' expression ';'
        
        Ví dụ: let x = 3 + 4 * 2;
        """
        self._expect(TokenType.LET)
        name_token = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.EQUALS)
        value = self.parse_expression()
        self._expect(TokenType.SEMICOLON)
        return LetStatement(name=name_token.value, value=value)
    
    def parse_print_statement(self) -> PrintStatement:
        """
        print_stmt → 'print' '(' expression ')'
        
        Note: không cần semicolon sau print (design choice đơn giản).
        """
        self._expect(TokenType.PRINT)
        self._expect(TokenType.LPAREN)
        expr = self.parse_expression()
        self._expect(TokenType.RPAREN)
        return PrintStatement(expr=expr)
    
    def parse_expression(self) -> ASTNode:
        """
        expression → term (('+' | '-') term)*
        
        Đây là mức THẤP NHẤT về precedence: + và - bind yếu nhất.
        
        Left-associative: 1 + 2 + 3 = (1 + 2) + 3
        Achieved bằng loop (thay vì left-recursion → infinite loop).
        """
        left = self.parse_term()
        
        while self._current().type in (TokenType.PLUS, TokenType.MINUS):
            op_token = self._advance()
            right = self.parse_term()
            left = BinaryOp(op=op_token.value, left=left, right=right)
        
        return left
    
    def parse_term(self) -> ASTNode:
        """
        term → factor (('*' | '/') factor)*
        
        * và / bind TIGHTER hơn + và - vì parse_term được gọi
        từ parse_expression → nó parse ở mức sâu hơn trong đệ quy.
        
        Đây là cách elegant nhất để encode operator precedence
        trong recursive descent — KHÔNG CẦN precedence table riêng.
        """
        left = self.parse_factor()
        
        while self._current().type in (TokenType.STAR, TokenType.SLASH):
            op_token = self._advance()
            right = self.parse_factor()
            left = BinaryOp(op=op_token.value, left=left, right=right)
        
        return left
    
    def parse_factor(self) -> ASTNode:
        """
        factor → ('-')? primary
        
        Xử lý unary minus: -x, -42, -(3 + 4)
        """
        if self._current().type == TokenType.MINUS:
            op_token = self._advance()
            operand = self.parse_primary()
            return UnaryOp(op=op_token.value, operand=operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        """
        primary → NUMBER | IDENTIFIER | '(' expression ')'
        
        Đây là "leaf level" — base cases của đệ quy.
        Ngoặc tạo đệ quy ngược lên parse_expression → cho phép grouping.
        """
        current = self._current()
        
        # Number literal
        if current.type == TokenType.NUMBER:
            self._advance()
            return NumberLiteral(value=float(current.value))
        
        # Identifier (variable reference)
        if current.type == TokenType.IDENTIFIER:
            self._advance()
            return Identifier(name=current.value)
        
        # Parenthesized expression — đệ quy!
        if current.type == TokenType.LPAREN:
            self._advance()
            expr = self.parse_expression()
            self._expect(TokenType.RPAREN)
            return expr
        
        raise ParseError(
            "Expected number, identifier, or '('",
            current
        )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def parse_source(source: str) -> Program:
    """Tokenize + parse source code, trả về AST."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    from .ast_printer import ASTPrinter
    
    source = "let x = 3 + 4 * 2; let y = x * x; print(y - 1)"
    print(f"Source: {source}")
    print()
    
    program = parse_source(source)
    printer = ASTPrinter()
    print(printer.format(program))

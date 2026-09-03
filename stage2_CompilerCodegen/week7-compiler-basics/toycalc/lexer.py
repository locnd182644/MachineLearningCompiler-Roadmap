"""
Lexer (Tokenizer) — Phase 1 của Compiler Pipeline
===================================================

LEXER LÀ GÌ?
    Lexer (hay tokenizer/scanner) là phase đầu tiên của mọi compiler.
    Nhiệm vụ: chuyển chuỗi ký tự (source code) thành dòng token có nghĩa.
    
    Ví dụ: "let x = 3 + 4 * 2" 
    → [LET, ID("x"), EQUALS, NUM(3), PLUS, NUM(4), STAR, NUM(2)]

TẠI SAO TÁCH LEXER KHỎI PARSER?
    1. Separation of concerns: lexer lo ký tự, parser lo cấu trúc
    2. Lexer hoạt động như finite automaton (đơn giản, nhanh)
    3. Parser hoạt động như pushdown automaton (phức tạp hơn, cần stack)
    4. Tách ra → dễ test, dễ thay đổi syntax mà không ảnh hưởng parser

LIÊN HỆ VỚI AI COMPILER:
    - MLIR có lexer riêng cho .mlir files (mlir/lib/AsmParser/Lexer.cpp)
    - PyTorch FX/JAX KHÔNG CẦN lexer truyền thống vì input là Python AST đã parsed
    - Triton cũng dùng Python AST → bỏ qua lexer/parser hoàn toàn
    → AI compilers thường "capture" computation graph thay vì parse text
    → Nhưng hiểu lexer/parser giúp hiểu MLIR assembly format

LÝ THUYẾT ĐẰNG SAU:
    - Mỗi token type tương ứng một regular expression (regex)
    - Lexer = Deterministic Finite Automaton (DFA) 
    - Lex/Flex tools sinh DFA từ regex specs
    - Chúng ta viết tay (hand-written lexer) vì dễ hiểu + control error messages
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


# =============================================================================
# TOKEN TYPES
# =============================================================================
# Mỗi enum value đại diện cho một loại "từ" trong ngôn ngữ.
# Tương tự: MLIR có các token types như `tok_identifier`, `tok_integer`, etc.
# (xem mlir/lib/AsmParser/Token.h)

class TokenType(Enum):
    """Các loại token trong ngôn ngữ ToyCalc."""
    
    # Literals — giá trị cố định
    NUMBER = auto()         # 42, 3.14
    
    # Identifiers — tên biến
    IDENTIFIER = auto()     # x, result, my_var
    
    # Keywords — từ khóa (giống reserved words trong C)
    LET = auto()            # let
    PRINT = auto()          # print
    
    # Operators — toán tử
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # *
    SLASH = auto()          # /
    
    # Delimiters — dấu phân cách  
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    EQUALS = auto()         # =
    SEMICOLON = auto()      # ;
    
    # Special
    EOF = auto()            # End of file/input


# Bảng keywords — dùng để phân biệt keyword vs identifier
# Ví dụ: "let" là keyword, "letter" là identifier
KEYWORDS = {
    'let': TokenType.LET,
    'print': TokenType.PRINT,
}

# Bảng single-character tokens
SINGLE_CHAR_TOKENS = {
    '+': TokenType.PLUS,
    '-': TokenType.MINUS,
    '*': TokenType.STAR,
    '/': TokenType.SLASH,
    '(': TokenType.LPAREN,
    ')': TokenType.RPAREN,
    '=': TokenType.EQUALS,
    ';': TokenType.SEMICOLON,
}


# =============================================================================
# TOKEN CLASS
# =============================================================================

@dataclass
class Token:
    """
    Một token = đơn vị từ vựng nhỏ nhất có nghĩa.
    
    Attributes:
        type:   Loại token (NUMBER, PLUS, LET, ...)
        value:  Giá trị gốc (string) — "42", "+", "let"
        line:   Dòng trong source (1-indexed) — cho error messages
        column: Cột trong source (1-indexed) — cho error messages
        
    Trong MLIR, token cũng mang location info (mlir::Location) để trace
    lỗi ngược về source. Đây là best practice quan trọng.
    """
    type: TokenType
    value: str
    line: int
    column: int
    
    def __repr__(self) -> str:
        if self.type in (TokenType.NUMBER, TokenType.IDENTIFIER):
            return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.column})"
        return f"Token({self.type.name}, L{self.line}:{self.column})"


# =============================================================================
# LEXER CLASS
# =============================================================================

class LexerError(Exception):
    """Lỗi trong quá trình tokenization."""
    def __init__(self, message: str, line: int, column: int):
        self.line = line
        self.column = column
        super().__init__(f"Lexer error at L{line}:{column}: {message}")


class Lexer:
    """
    Lexer cho ngôn ngữ ToyCalc.
    
    Hoạt động như một Deterministic Finite Automaton (DFA):
    - State = vị trí hiện tại trong source + ký tự đang xét
    - Transitions = luật chuyển state dựa trên ký tự tiếp theo
    - Accept states = khi hoàn thành một token
    
    Cách dùng:
        lexer = Lexer("let x = 42;")
        tokens = lexer.tokenize()
        # hoặc
        for token in lexer:
            print(token)
    """
    
    def __init__(self, source: str):
        """
        Args:
            source: Source code string cần tokenize.
        """
        self.source = source
        self.pos = 0            # Vị trí hiện tại trong source
        self.line = 1           # Dòng hiện tại (1-indexed)
        self.column = 1         # Cột hiện tại (1-indexed)
        self.tokens: List[Token] = []
    
    # -------------------------------------------------------------------------
    # Helper methods — đọc và di chuyển trong source
    # -------------------------------------------------------------------------
    
    def _current_char(self) -> Optional[str]:
        """Trả về ký tự hiện tại, hoặc None nếu hết source."""
        if self.pos < len(self.source):
            return self.source[self.pos]
        return None
    
    def _peek(self) -> Optional[str]:
        """Nhìn trước 1 ký tự mà không di chuyển (lookahead = 1)."""
        next_pos = self.pos + 1
        if next_pos < len(self.source):
            return self.source[next_pos]
        return None
    
    def _advance(self) -> str:
        """
        Đọc ký tự hiện tại và di chuyển sang ký tự tiếp theo.
        Cập nhật line/column cho error reporting.
        """
        char = self.source[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def _skip_whitespace(self):
        """Bỏ qua whitespace (space, tab, newline)."""
        while self.pos < len(self.source) and self.source[self.pos].isspace():
            self._advance()
    
    def _skip_comment(self):
        """Bỏ qua comment dạng // ... (single-line)."""
        if (self.pos < len(self.source) - 1 and 
                self.source[self.pos] == '/' and self.source[self.pos + 1] == '/'):
            while self.pos < len(self.source) and self.source[self.pos] != '\n':
                self._advance()
    
    # -------------------------------------------------------------------------
    # Token reading methods — mỗi method đọc một loại token
    # -------------------------------------------------------------------------
    # Mỗi method tương ứng với một "state" trong DFA:
    #   _read_number():     state khi đang đọc digits
    #   _read_identifier(): state khi đang đọc chữ cái
    # -------------------------------------------------------------------------
    
    def _read_number(self) -> Token:
        """
        Đọc một số (integer hoặc float).
        
        Regex tương đương: [0-9]+(\.[0-9]+)?
        DFA: START →(digit)→ INTEGER →(.)→ DECIMAL →(digit)→ FLOAT
        """
        start_line = self.line
        start_col = self.column
        result = ''
        
        # Phần integer
        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            result += self._advance()
        
        # Phần decimal (optional)
        if (self.pos < len(self.source) and self.source[self.pos] == '.' 
                and self._peek() is not None and self._peek().isdigit()):
            result += self._advance()  # dấu '.'
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                result += self._advance()
        
        return Token(TokenType.NUMBER, result, start_line, start_col)
    
    def _read_identifier(self) -> Token:
        """
        Đọc identifier hoặc keyword.
        
        Regex: [a-zA-Z_][a-zA-Z0-9_]*
        Sau khi đọc xong, tra bảng KEYWORDS để xem có phải keyword không.
        
        Đây là cách hầu hết compiler phân biệt keyword vs identifier:
        đọc toàn bộ "word" trước, tra bảng sau. Không hardcode vào DFA.
        """
        start_line = self.line
        start_col = self.column
        result = ''
        
        while (self.pos < len(self.source) and 
               (self.source[self.pos].isalnum() or self.source[self.pos] == '_')):
            result += self._advance()
        
        # Tra bảng keyword
        token_type = KEYWORDS.get(result, TokenType.IDENTIFIER)
        return Token(token_type, result, start_line, start_col)
    
    # -------------------------------------------------------------------------
    # Main tokenization
    # -------------------------------------------------------------------------
    
    def _next_token(self) -> Optional[Token]:
        """
        Đọc token tiếp theo từ source.
        
        Đây là "transition function" chính của DFA:
        nhìn ký tự đầu tiên → quyết định chuyển sang state nào.
        """
        self._skip_whitespace()
        
        # Kiểm tra comments
        if (self.pos < len(self.source) - 1 and 
                self.source[self.pos] == '/' and self.source[self.pos + 1] == '/'):
            self._skip_comment()
            self._skip_whitespace()
        
        if self.pos >= len(self.source):
            return Token(TokenType.EOF, '', self.line, self.column)
        
        char = self.source[self.pos]
        
        # Số → chuyển sang state đọc number
        if char.isdigit():
            return self._read_number()
        
        # Chữ cái hoặc _ → chuyển sang state đọc identifier/keyword
        if char.isalpha() or char == '_':
            return self._read_identifier()
        
        # Single-character token → tra bảng
        if char in SINGLE_CHAR_TOKENS:
            token = Token(
                SINGLE_CHAR_TOKENS[char],
                char,
                self.line,
                self.column
            )
            self._advance()
            return token
        
        # Ký tự không hợp lệ → error
        raise LexerError(
            f"Unexpected character: {char!r}",
            self.line,
            self.column
        )
    
    def tokenize(self) -> List[Token]:
        """
        Tokenize toàn bộ source code, trả về list of tokens.
        
        Đây là interface chính của Lexer. Token cuối cùng luôn là EOF.
        
        Returns:
            List[Token]: Danh sách tokens kết thúc bằng EOF.
        """
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        while True:
            token = self._next_token()
            self.tokens.append(token)
            if token.type == TokenType.EOF:
                break
        
        return self.tokens
    
    def __iter__(self):
        """Cho phép iterate qua tokens: for tok in Lexer(source)."""
        self.tokenize()
        return iter(self.tokens)


# =============================================================================
# DEMO
# =============================================================================

def _print_tokens(source: str):
    """Helper: tokenize và in ra tokens đẹp."""
    print(f"\n{'='*60}")
    print(f"Source: {source!r}")
    print(f"{'='*60}")
    
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    for i, tok in enumerate(tokens):
        print(f"  [{i:2d}] {tok}")
    
    print(f"  Total: {len(tokens)} tokens (including EOF)")


if __name__ == '__main__':
    print("=" * 60)
    print(" TOYCALC LEXER — Phase 1: Source → Tokens")
    print("=" * 60)
    print()
    print("Lexer chuyển source code (chuỗi ký tự) thành dòng token.")
    print("Mỗi token là đơn vị từ vựng nhỏ nhất có nghĩa.")
    print()
    
    # Demo 1: Biểu thức đơn giản
    _print_tokens("let x = 3 + 4 * 2; let y = x * x; print(y - 1)")
    
    # Demo 2: Có ngoặc
    _print_tokens("let result = (10 + 20) * 3; print(result)")
    
    # Demo 3: Biến và phép tính
    _print_tokens("let a = 100; let b = a / 5; print(a - b)")
    
    # Demo 4: Số thực
    _print_tokens("let pi = 3.14; print(pi * 2)")
    
    print()
    print("=" * 60)
    print("💡 Mỗi token type tương ứng một regex pattern:")
    print("   NUMBER     : [0-9]+(\\.[0-9]+)?")
    print("   IDENTIFIER : [a-zA-Z_][a-zA-Z0-9_]*")
    print("   KEYWORD    : 'let' | 'print' (subset của identifier)")
    print("   OPERATOR   : '+' | '-' | '*' | '/'")
    print("=" * 60)

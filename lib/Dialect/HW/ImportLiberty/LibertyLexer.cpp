//===- LibertyLexer.cpp - .liberty file lexer implementation
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .liberty file lexer.
//
//===----------------------------------------------------------------------===//

#include "LibertyLexer.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace liberty;
using llvm::SMLoc;
using llvm::SMRange;
using llvm::SourceMgr;

#define isdigit(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS
#define isalpha(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS

//===----------------------------------------------------------------------===//
// LibertyToken
//===----------------------------------------------------------------------===//

SMLoc LibertyToken::getLoc() const {
  return SMLoc::getFromPointer(spelling.data());
}

SMLoc LibertyToken::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange LibertyToken::getLocRange() const {
  return SMRange(getLoc(), getEndLoc());
}

StringRef LibertyToken::getKindString() const {
  switch (kind) {
#define TOK_MARKER(NAME)                                                       \
  case LibertyToken::Kind::NAME:                                               \
    return #NAME;
#define TOK_IDENTIFIER(NAME)                                                   \
  case LibertyToken::Kind::NAME:                                               \
    return #NAME;
#define TOK_LITERAL(NAME)                                                      \
  case LibertyToken::Kind::NAME:                                               \
    return #NAME;
#define TOK_PUNCTUATION(NAME, SPELLING)                                        \
  case LibertyToken::Kind::NAME:                                               \
    return #NAME;
#define TOK_KEYWORD(SPELLING)                                                  \
  case LibertyToken::Kind::kw_##SPELLING:                                      \
    return #SPELLING;

#include "LibertyTokenKinds.def"

    // default:
    //   assert(false && "Unknown token type");
  }
  return "unknown";
}

//===----------------------------------------------------------------------===//
// LibertyLexer
//===----------------------------------------------------------------------===//

static StringAttr getMainBufferNameIdentifier(const llvm::SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

LibertyLexer::LibertyLexer(const llvm::SourceMgr &sourceMgr,
                           MLIRContext *context)
    : sourceMgr(sourceMgr),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
      curBuffer(
          sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
      curPtr(curBuffer.begin()),
      // Prime the Libertyst token.
      curToken(lexTokenImpl()) {}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location LibertyLexer::translateLocation(llvm::SMLoc loc) {
  assert(loc.isValid());
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
}

/// Emit an error message and return a LibertyToken::error token.
LibertyToken LibertyLexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(translateLocation(SMLoc::getFromPointer(loc)), message);
  return formToken(LibertyToken::error, loc);
}

//===----------------------------------------------------------------------===//
// Lexer Implementation Methods
//===----------------------------------------------------------------------===//

LibertyToken LibertyLexer::lexTokenImpl() {
  while (true) {
    const char *tokStart = curPtr;
    switch (*curPtr++) {
    default:
      // Handle identifiers.
      if (llvm::isAlpha(curPtr[-1]))
        return lexIdentifierOrKeyword(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that llvm::MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(LibertyToken::eof, tokStart);

      [[fallthrough]]; // Treat as whitespace.

    case '\\':
      // Handle line continuations.
      if (*curPtr == '\r') {
        ++curPtr;
      }
      if (*curPtr == '\n') {
        ++curPtr;
        continue;
      }

      [[fallthrough]]; // Treat as whitespace.

    // case '.':
    case ',':
      return formToken(LibertyToken::comma, tokStart);
    case ':':
      return lexColonValue(tokStart);
    case '(':
      return lexParenValue(tokStart);
    case '{':
      return formToken(LibertyToken::l_brace, tokStart);
    case '}':
      return formToken(LibertyToken::r_brace, tokStart);
    case '"':
      return lexString(tokStart);

    case '\n':
    case ';':
    case ' ':
    case '\t':
    case '\r':
      // Handle whitespace.
      continue;

    case '#':
      skipComment();
      continue;

    case '/':
      if (*curPtr == '/') {
        ++curPtr;
        skipComment();
      } else if (*curPtr == '*') {
        ++curPtr;
        skipBlockComment();
      } else {
        return emitError(tokStart, "unexpected character after '/'");
      }
      continue;

    case '-':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexNumber(tokStart);
    }
  }
}

/// Lex an identifier.
///
///   LegalStartChar ::= [a-zA-Z]
///   LegalIdChar    ::= LegalStartChar | [0-9] | '$' | '_'
///
LibertyToken LibertyLexer::lexIdentifierOrKeyword(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z$_]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_')
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);
  LibertyToken::Kind kind = llvm::StringSwitch<LibertyToken::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, LibertyToken::kw_##SPELLING)
#include "LibertyTokenKinds.def"
                                .Default(LibertyToken::identifier);

  return formToken(kind, tokStart);
}

/// Skip a comment line, starting with a '#' and going to end of line.
void LibertyLexer::skipComment() {
  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

void LibertyLexer::skipBlockComment() {
  while (true) {
    switch (*curPtr++) {
    case '*':
      if (*curPtr == '/') {
        ++curPtr;
        return;
      }
    }
  }
}

/// Lex a number literal.
///
///    ( '+' | '-' )? Digit+ '.' Digit+
///
LibertyToken LibertyLexer::lexNumber(const char *tokStart) {
  assert(llvm::isDigit(curPtr[-1]) || curPtr[-1] == '+' || curPtr[-1] == '-');

  // There needs to be at least one digit.
  if (!llvm::isDigit(*curPtr) && !llvm::isDigit(curPtr[-1]))
    return emitError(tokStart, "unexpected character after sign");

  while (*curPtr == '.' || llvm::isDigit(*curPtr))
    ++curPtr;

  return formToken(LibertyToken::number, tokStart);
}

/// Lex a string literal.
///
LibertyToken LibertyLexer::lexString(const char *tokStart) {
  while (true) {
    switch (*curPtr++) {
    case '"': // This is the end of the string literal.
      return formToken(LibertyToken::string, tokStart);

    case 0:
      // This could be the end of file in the middle of the string.  If so
      // emit an error.
      if (curPtr - 1 != curBuffer.end())
        break;
      [[fallthrough]];
    case '\n': // Vertical whitespace isn't allowed in a string.
    case '\r':
    case '\v':
    case '\f':
      return emitError(tokStart, "unterminated string");
    default:
      if (curPtr[-1] & ~0x7F)
        return emitError(tokStart, "string characters must be 7-bit ASCII");
      // Skip over other characters.
      break;
    }
  }
}

LibertyToken LibertyLexer::lexParenValue(const char *tokStart) {
  assert(*tokStart == '(');
  ++tokStart;

  while (*curPtr != ')') {
    ++curPtr;
  }

  // might be multiple string values inside one paren
  auto tok = formToken(LibertyToken::paren_value, tokStart);
  ++curPtr; // skip ')'
  return tok;
}

LibertyToken LibertyLexer::lexColonValue(const char *tokStart) {
  assert(*tokStart == ':');
  tokStart = skipWhitespace(curPtr);

  curPtr = tokStart;
  while (*curPtr != ';') {
    ++curPtr;
  }

  auto tok = formTokenIgnoreQuotes(LibertyToken::colon_value, tokStart);
  ++curPtr; // skip ';'
  return tok;
}

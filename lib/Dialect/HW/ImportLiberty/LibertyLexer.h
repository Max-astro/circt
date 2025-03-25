//===- LibertyLexer.h - .liberty lexer and token definitions ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the a Lexer and Token interface for .liberty files.
//
//===----------------------------------------------------------------------===//

#ifndef LIBERTY_LEXER_H
#define LIBERTY_LEXER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
class Location;
} // namespace mlir

namespace circt {
namespace liberty {

/// This represents a specific token for .lib files.
class LibertyToken {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "LibertyTokenKinds.def"
  };

  LibertyToken(Kind kind, StringRef spelling)
      : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  // Debug utils
  StringRef getKindString() const;

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  // Location processing.
  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

class LibertyLexerCursor;

/// This implements a lexer for .lib files.
class LibertyLexer {
public:
  LibertyLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context);

  const llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }

  /// Move to the next valid token.
  void lexToken() { curToken = lexTokenImpl(); }

  const LibertyToken &getToken() const { return curToken; }

  mlir::Location translateLocation(llvm::SMLoc loc);

  /// Get an opaque pointer into the lexer state that can be restored later.
  LibertyLexerCursor getCursor() const;

private:
  LibertyToken lexTokenImpl();

  // Helpers.
  LibertyToken formToken(LibertyToken::Kind kind, const char *tokStart) {
    return LibertyToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  // skip leading and ending quotes if any
  LibertyToken formTokenIgnoreQuotes(LibertyToken::Kind kind,
                                     const char *tokStart) {
    if (*tokStart == '"') {
      tokStart++;
      return LibertyToken(kind, StringRef(tokStart, curPtr - tokStart - 1));
    }
    return formToken(kind, tokStart);
  }

  const char *skipWhitespace(const char *tokStart) {
    while (*tokStart == ' ') {
      ++tokStart;
    }
    return tokStart;
  }

  LibertyToken emitError(const char *loc, const Twine &message);

  // Lexer implementation methods.
  LibertyToken lexFileInfo(const char *tokStart);
  LibertyToken lexIdentifierOrKeyword(const char *tokStart);
  LibertyToken lexNumber(const char *tokStart);
  LibertyToken lexString(const char *tokStart);
  void skipComment();
  void skipBlockComment();
  LibertyToken lexParenValue(const char *tokStart);
  LibertyToken lexKeyString(const char *tokStart);
  LibertyToken lexColonValue(const char *tokStart);
  LibertyToken lexCommand(const char *tokStart);

  const llvm::SourceMgr &sourceMgr;
  const mlir::StringAttr bufferNameIdentifier;

  StringRef curBuffer;
  const char *curPtr;

  /// This is the next token that hasn't been consumed yet.
  LibertyToken curToken;

  LibertyLexer(const LibertyLexer &) = delete;
  void operator=(const LibertyLexer &) = delete;
  friend class LibertyLexerCursor;
};

/// This is the state captured for a lexer cursor.
class LibertyLexerCursor {
public:
  LibertyLexerCursor(const LibertyLexer &lexer)
      : state(lexer.curPtr), curToken(lexer.getToken()) {}

  void restore(LibertyLexer &lexer) {
    lexer.curPtr = state;
    lexer.curToken = curToken;
  }

private:
  const char *state;
  LibertyToken curToken;
};

inline LibertyLexerCursor LibertyLexer::getCursor() const {
  return LibertyLexerCursor(*this);
}

} // namespace liberty
} // namespace circt

#endif // LIBERTY_LEXER_H

#include "LibertyAST.h"
#include "LibertyLexer.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/StandardCellLiberty.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <memory>
#include <variant>

using namespace circt;
using namespace liberty;

using llvm::SMLoc;
using llvm::SourceMgr;
using mlir::LocationAttr;

//===----------------------------------------------------------------------===//
// BLIFFileParser
//===----------------------------------------------------------------------===//

namespace {
struct LibertyParser {
  LibertyParser(MLIRContext *context, LibertyLexer &lexer)
      : context(context), lexer(lexer) {}

  MLIRContext *getContext() const { return context; }

  LibertyLexer &getLexer() { return lexer; }

  /// Return the current token the parser is inspecting.
  const LibertyToken &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  Location translateLocation(llvm::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {}) {
    auto diag = mlir::emitError(translateLocation(loc), message);

    // If we hit a parse error in response to a lexer error, then the lexer
    // already reported the error.
    if (getToken().is(LibertyToken::error))
      diag.abandon();
    return diag;
  }

  /// Emit a warning.
  InFlightDiagnostic emitWarning(const Twine &message = {}) {
    return emitWarning(getToken().getLoc(), message);
  }

  InFlightDiagnostic emitWarning(SMLoc loc, const Twine &message = {}) {
    return mlir::emitWarning(translateLocation(loc), message);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.

  /// If not, return false.
  bool consumeIf(LibertyToken::Kind kind) {
    if (getToken().isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  ///
  /// This returns the consumed token.
  LibertyToken consumeToken() {
    LibertyToken consumedToken = getToken();
    assert(consumedToken.isNot(LibertyToken::eof, LibertyToken::error) &&
           "shouldn't advance past EOF or errors");
    lexer.lexToken();
    return consumedToken;
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  ///
  /// This returns the consumed token.
  LibertyToken consumeToken(LibertyToken::Kind kind) {
    LibertyToken consumedToken = getToken();
    if (!consumedToken.is(kind)) {
      emitError(consumedToken.getLoc(),
                "expected token: " + consumedToken.getKindString());
      exit(1);
    }
    // assert(consumedToken.is(kind) && "consumed an unexpected token");
    consumeToken();
    return consumedToken;
  }

  //===--------------------------------------------------------------------===//
  // Common Parser Rules
  //===--------------------------------------------------------------------===//

  llvm::ParseResult parseLibertyFile();
  llvm::ParseResult parseLibertyValue(LibraryAST &lib);
  llvm::ParseResult parseStatement(CommonGroupAST &group);
  llvm::ParseResult parseDefineStatement(CommonGroupAST &parentGroup);
  llvm::ParseResult parseAttributeStatement(CommonGroupAST &parentGroup);
  llvm::ParseResult parseGroupStatement(CommonGroupAST &parentGroup,
                                        StringRef groupName,
                                        StringRef parenValue);
  llvm::ParseResult parseCellGroup(LibraryAST &lib);
  llvm::ParseResult parsePinGroup(CommonGroupAST &parentGroup);
  llvm::ParseResult parseLuTemplate(LibraryAST &lib);

  const LibraryAST &getParseResult() const { return libAST; }

private:
  LibertyParser(const LibertyParser &) = delete;
  void operator=(const LibertyParser &) = delete;

  /// The context in which we are parsing.
  MLIRContext *context;

  /// BLIFParser is subclassed and reinstantiated.  Do not add additional
  /// non-trivial state here, add it to SharedParserConstants.
  LibertyLexer &lexer;

  /// Parse result.
  LibraryAST libAST;
};

//===--------------------------------------------------------------------===//
// Common Parser Rules
//===--------------------------------------------------------------------===//

llvm::ParseResult LibertyParser::parseLibertyFile() {
  assert("library" == consumeToken(LibertyToken::kw_library).getSpelling());
  libAST.libName = consumeToken(LibertyToken::paren_value).getSpelling().trim();
  // return parseGroupValue(libInfo.libData);
  return parseLibertyValue(libAST);
}

llvm::ParseResult LibertyParser::parseLibertyValue(LibraryAST &libAST) {
  consumeToken(LibertyToken::l_brace);
  while (getToken().getKind() != LibertyToken::r_brace) {
    switch (getToken().getKind()) {
    case LibertyToken::eof:
    case LibertyToken::error:
      emitError(getToken().getLoc(), "Unexpected token, expected statement");
      return failure();

    case LibertyToken::kw_define:
    case LibertyToken::identifier:
      if (parseStatement(libAST.group)) {
        return failure();
      }
      break;
    case LibertyToken::kw_cell:
      if (parseCellGroup(libAST)) {
        return failure();
      }
      break;
    case LibertyToken::kw_lu_table_template:
      if (parseLuTemplate(libAST)) {
        return failure();
      }
      break;

    default:
      llvm::outs() << "Failed to parse liberty statement, current token: "
                   << getToken().getKindString() << " ("
                   << getToken().getSpelling() << ")\n";
      return failure();
    }
  }
  consumeToken(LibertyToken::r_brace);
  return success();
}

llvm::ParseResult LibertyParser::parseStatement(CommonGroupAST &group) {
  if (getToken().getKind() == LibertyToken::kw_define) {
    return parseDefineStatement(group);
  }
  if (getToken().getKind() == LibertyToken::identifier) {
    return parseAttributeStatement(group);
  }
  if (getToken().getKind() == LibertyToken::kw_pin) {
    return parsePinGroup(group);
  }

  emitError(getToken().getLoc(),
            "Unexpected token `" + getToken().getSpelling() + " [" +
                getToken().getKindString() +
                "]`, expected define, simple attribute or group");
  return failure();
}

llvm::ParseResult
LibertyParser::parseDefineStatement(CommonGroupAST &parentGroup) {
  consumeToken(LibertyToken::kw_define);
  auto defineValue = consumeToken(LibertyToken::paren_value);
  auto [attributeName, res1] = defineValue.getSpelling().trim().split(",");
  auto [groupName, res2] = defineValue.getSpelling().trim().split(",");
  auto [attributeType, _] = defineValue.getSpelling().trim().split(",");

  parentGroup.defines.try_emplace(
      attributeName, DefineType(attributeName, groupName, attributeType));

  return success();
}

llvm::ParseResult
LibertyParser::parseAttributeStatement(CommonGroupAST &parentGroup) {
  auto attrName = consumeToken(LibertyToken::identifier).getSpelling().trim();
  switch (getToken().getKind()) {
  case LibertyToken::colon_value: {
    /// simple attribute:
    /// `attribute_name : attribute_value`
    auto attrValue =
        consumeToken(LibertyToken::colon_value).getSpelling().trim();
    parentGroup.emplaceAttribute(attrName, attrValue);
    return success();
  }

  case LibertyToken::paren_value: {
    auto parenValue =
        consumeToken(LibertyToken::paren_value).getSpelling().trim();
    if (getToken().getKind() == LibertyToken::l_brace) {
      /// group statements:
      /// `group_name (paren_value) { ... }`
      return parseGroupStatement(parentGroup, attrName, parenValue);
    }

    /// complex attribute:
    /// `attribute_name (attribute_values...)`
    parentGroup.emplaceAttribute(attrName, parenValue);
    return success();
  }

  default:
    emitError(getToken().getLoc(),
              "Unexpected token, expected attribute value");
    return failure();
  }
}

/// group_name (paren_value) {
///   ... statements ...
/// }
llvm::ParseResult
LibertyParser::parseGroupStatement(CommonGroupAST &parentGroup,
                                   StringRef groupName, StringRef parenValue) {

  consumeToken(LibertyToken::l_brace);

  /// Need to create a subgroup for sub statements
  auto subGrp = std::make_unique<CommonGroupAST>(groupName, parenValue);
  while (getToken().getKind() != LibertyToken::r_brace) {
    if (getToken().getKind() == LibertyToken::eof ||
        getToken().getKind() == LibertyToken::error) {
      emitError(getToken().getLoc(), "Unexpected token, expected statement");
      return failure();
    }

    if (parseStatement(*subGrp)) {
      return failure();
    }
  }

  consumeToken(LibertyToken::r_brace);
  parentGroup.emplaceSubGroup(groupName, std::move(subGrp));

  return success();
}

llvm::ParseResult LibertyParser::parseCellGroup(LibraryAST &lib) {
  consumeToken(LibertyToken::kw_cell);

  auto cellName = consumeToken(LibertyToken::paren_value).getSpelling().trim();
  auto [it, first] = lib.cells.try_emplace(
      cellName, std::make_unique<CommonGroupAST>(cellName));
  if (!first) {
    emitWarning(getToken().getLoc(), "Redefined cell `" + cellName + "`");
  }

  auto &cell = it->second;

  consumeToken(LibertyToken::l_brace);
  while (getToken().getKind() != LibertyToken::r_brace) {
    switch (getToken().getKind()) {
    case LibertyToken::eof:
    case LibertyToken::error: {
      emitError(getToken().getLoc(),
                "Unexpected token in cell group, expected statement");
      return failure();
    }

    case LibertyToken::kw_pin: {
      if (parsePinGroup(*cell))
        return failure();
      break;
    }

    default: {
      if (parseStatement(*cell))
        return failure();
      break;
    }
    }
  }

  consumeToken(LibertyToken::r_brace);
  return success();
}

llvm::ParseResult LibertyParser::parsePinGroup(CommonGroupAST &parentGroup) {
  consumeToken(LibertyToken::kw_pin);
  auto pinName = consumeToken(LibertyToken::paren_value).getSpelling().trim();
  SmallVector<CommonGroupValue> v;
  v.emplace_back(std::make_unique<CommonGroupAST>("pin", pinName));
  auto [it, first] = parentGroup.subGroups.try_emplace(pinName, std::move(v));
  if (!first) {
    emitWarning(getToken().getLoc(), "Redefined pin `" + pinName +
                                         "` in group `" + parentGroup.groupKey +
                                         "`");
  }

  auto &pinGroup = it->second[0];

  consumeToken(LibertyToken::l_brace);
  while (getToken().getKind() != LibertyToken::r_brace) {
    if (parseStatement(*pinGroup))
      return failure();
  }

  consumeToken(LibertyToken::r_brace);
  return success();
}

llvm::ParseResult LibertyParser::parseLuTemplate(LibraryAST &lib) {
  consumeToken(LibertyToken::kw_lu_table_template);
  auto luTemplateName =
      consumeToken(LibertyToken::paren_value).getSpelling().trim();
  auto [it, first] = lib.luTableTemplates.try_emplace(
      luTemplateName,
      std::make_unique<CommonGroupAST>("lu_table_template", luTemplateName));
  if (!first) {
    emitWarning(getToken().getLoc(),
                "Redefined lu_table_template `" + luTemplateName + "`");
  }

  auto &luTemplateGroup = it->second;

  consumeToken(LibertyToken::l_brace);
  while (getToken().getKind() != LibertyToken::r_brace) {
    if (parseStatement(*luTemplateGroup))
      return failure();
  }

  consumeToken(LibertyToken::r_brace);
  return success();
}

//===--------------------------------------------------------------------===//
// AST printer
//===--------------------------------------------------------------------===//

void printDefineType(llvm::raw_ostream &os, const DefineType &define,
                     int indent) {
  os << llvm::indent(indent) << "Define(" << define.attributeName << ", "
     << define.groupName << ", " << define.attributeType << ")\n";
}

void printCommonGroupAST(llvm::raw_ostream &os, const CommonGroupAST &group,
                         int indent) {
  os << llvm::indent(indent) << "Group: " << group.groupName << "("
     << group.groupKey << ")\n";
  if (!group.defines.empty()) {
    os << llvm::indent(indent + 2) << "defines:\n";
    for (const auto &[attrName, define] : group.defines) {
      printDefineType(os, define, indent + 4);
    }
    os << "\n";
  }

  if (!group.attributes.empty()) {
    os << llvm::indent(indent + 2) << "attributes:\n";
    for (const auto &[attrName, attrValue] : group.attributes) {
      os << llvm::indent(indent + 4) << attrName << " : ";
      for (const auto &value : attrValue) {
        os << value << ", ";
      }
      os << "\n";
    }
    os << "\n";
  }

  if (!group.subGroups.empty()) {
    os << llvm::indent(indent + 2) << "subGroups:\n";
    for (const auto &[subGroupName, subGroup] : group.subGroups) {
      for (const auto &grp : subGroup) {
        printCommonGroupAST(os, *grp, indent + 4);
      }
    }
    os << "\n";
  }
}

void printLibaryAST(llvm::raw_ostream &os, const LibraryAST &lib) {
  os << "Library: " << lib.libName << "\n";
  os << "luTemplates:\n";
  for (const auto &[attrName, grp] : lib.luTableTemplates) {
    printCommonGroupAST(os, *grp, 2);
  }
  os << "\n";

  os << "cells:\n";
  for (const auto &[cellName, cell] : lib.cells) {
    printCommonGroupAST(os, *cell, 2);
  }
  os << "\n";

  printCommonGroupAST(os, lib.group, 2);
}

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Parse the specified .lib file into the specified MLIR context.
mlir::OwningOpRef<mlir::ModuleOp>
circt::liberty::importLibertyFile(SourceMgr &sourceMgr, MLIRContext *context,
                                  mlir::TimingScope &ts,
                                  LibertyParserOptions options) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  context->loadDialect<hw::HWDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<mlir::ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(),
                          /*line=*/0,
                          /*column=*/0)));

  {
    LibertyLexer lexer(sourceMgr, context);
    auto lexLogFile = mlir::openOutputFile("lex.log");
    if (!lexLogFile) {
      llvm::errs() << "Failed to open lex.log\n";
    }

    auto &lexLog = lexLogFile->os();

    int idx = 0;
    auto tok = lexer.getToken();
    while (tok.isNot(LibertyToken::eof, LibertyToken::error)) {
      // Format idx as fixed width (4 characters) and token type as fixed
      // width (15 characters)
      lexLog << "Token " << llvm::formatv("{0,4}", idx++)
             << " : type = " << llvm::formatv("{0,-12}", tok.getKindString())
             << " |  " << tok.getSpelling() << "\n";
      lexer.lexToken();
      tok = lexer.getToken();
    }
    lexLogFile->keep();
  }

  LibertyLexer lexer(sourceMgr, context);
  LibertyParser parser(context, lexer);
  if (parser.parseLibertyFile()) {
    return nullptr;
  }

  auto &&libInfo = parser.getParseResult();
  printLibaryAST(llvm::outs(), libInfo);

  llvm::outs() << "\n";
  return module;
}

void circt::liberty::registerFromLibertyFileTranslation() {
  static mlir::TranslateToMLIRRegistration fromLib(
      "import-liberty", "import .lib",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        return importLibertyFile(sourceMgr, context, ts);
      });
}
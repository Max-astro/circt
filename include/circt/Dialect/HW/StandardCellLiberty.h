//===- StandardCellLiberty.h - .lib to HW dialect parser --------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the .lib file parser.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_STANDARDCELLLIBERTY_H
#define CIRCT_DIALECT_HW_STANDARDCELLLIBERTY_H

#include "circt/Support/LLVM.h"
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class LocationAttr;
class TimingScope;
} // namespace mlir

namespace circt {
namespace liberty {

struct LibertyParserOptions {
//   /// Specify how @info locators should be handled.
//   enum class InfoLocHandling {
//     /// If this is set to true, the @info locators are ignored, and the
//     /// locations are set to the location in the .BLIF file.
//     IgnoreInfo,
//     /// Prefer @info locators, fallback to .BLIF locations.
//     PreferInfo,
//     /// Attach both @info locators (when present) and .BLIF locations.
//     FusedInfo
//   };

//   InfoLocHandling infoLocatorHandling = InfoLocHandling::PreferInfo;

//   /// parse strict blif instead of extended blif
//   bool strictBLIF = false;
};

mlir::OwningOpRef<mlir::ModuleOp>
importLibertyFile(llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context,
               mlir::TimingScope &ts, LibertyParserOptions options = {});


void registerFromLibertyFileTranslation();

} // namespace liberty
} // namespace circt

#endif // CIRCT_DIALECT_HW_STANDARDCELLLIBERTY_H

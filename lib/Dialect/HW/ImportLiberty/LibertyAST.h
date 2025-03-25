// LibertyValue.h
#ifndef CIRCT_DIALECT_HW_IMPORTLIBERTY_LIBERTYVALUE_H
#define CIRCT_DIALECT_HW_IMPORTLIBERTY_LIBERTYVALUE_H

#include <memory>

#include "circt/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"

namespace circt {
namespace liberty {

// struct GroupInfo;
struct CommonGroupAST;
struct CellAST;

template <typename ValueTy>
using StringMapVector =
    llvm::MapVector<StringRef, ValueTy, llvm::StringMap<int>>;

template <typename ValueTy>
using StringMultiMapVector =
    llvm::MapVector<StringRef, SmallVector<ValueTy>, llvm::StringMap<int>>;

using CommonGroupValue = std::unique_ptr<CommonGroupAST>;
using GroupMap = StringMapVector<CommonGroupValue>;
using GroupMultiMap = StringMultiMapVector<CommonGroupValue>;

using AttributeValue = StringRef;
using AttributeMultiMap = StringMultiMapVector<AttributeValue>;

using CellMap = GroupMap;

// ----------------------------------------------------------------------------

/// define (attribute_name, group_name, attribute_type)
struct DefineType {
  DefineType(StringRef attributeName, StringRef groupName,
             StringRef attributeType)
      : attributeName(attributeName), groupName(groupName),
        attributeType(attributeType) {}

  StringRef attributeName;
  StringRef groupName;
  StringRef attributeType;
};

/// TimingGroup example:
/// timing() {
///    related_pin : "A1";
///    cell_rise ("delay_outputslew_template_7X8") {
///      ... LookUpTable ...
///    }
///    cell_fall ("delay_outputslew_template_7X8") {
///      ... LookUpTable ...
///    }
/// }
///

/// group_name (group_key) {
///   ... defines ...
///   ... simple attributes ...
///   ... group statements ...
/// }
struct CommonGroupAST {
  StringRef groupName; // like cell / pin / timing ...
  StringRef groupKey;  // like AND / Z / DFF ... , nullable

  StringMapVector<DefineType> defines;
  AttributeMultiMap attributes;
  GroupMultiMap subGroups;

  CommonGroupAST(StringRef groupName = "", StringRef groupKey = "")
      : groupName(groupName), groupKey(groupKey) {}

  void emplaceAttribute(StringRef attrName, StringRef attrValue) {
    if (auto *it = attributes.find(attrName); it != attributes.end()) {
      it->second.emplace_back(attrValue);
    } else {
      attributes.try_emplace(attrName, SmallVector<StringRef>{attrValue});
    }
  }

  void emplaceSubGroup(StringRef groupName, CommonGroupValue group) {
    if (auto *it = subGroups.find(groupName); it != subGroups.end()) {
      it->second.emplace_back(std::move(group));
    } else {
      SmallVector<CommonGroupValue> v{};
      v.emplace_back(std::move(group));
      subGroups.try_emplace(groupName, std::move(v));
    }
  }
};

/// library (library_name) {
///   ... statements ...
/// }
struct LibraryAST {
  StringRef libName;

  GroupMap luTableTemplates;
  GroupMap cells;

  CommonGroupAST group;

  const CommonGroupAST &getGroupInfo() const { return group; }
  const GroupMap &getLuTableTemplates() const { return luTableTemplates; }
  const CellMap &getCells() const { return cells; }
};

} // namespace liberty
} // namespace circt

#endif // CIRCT_DIALECT_HW_IMPORTLIBERTY_LIBERTYVALUE_H
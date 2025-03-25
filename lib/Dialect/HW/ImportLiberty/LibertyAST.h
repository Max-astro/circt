// LibertyValue.h
#ifndef CIRCT_DIALECT_HW_IMPORTLIBERTY_LIBERTYVALUE_H
#define CIRCT_DIALECT_HW_IMPORTLIBERTY_LIBERTYVALUE_H

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "circt/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

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

/*
struct LuTableTemplateInfo {
  SmallVector<StringRef, 3> variables;
  SmallVector<StringRef, 3> indices;
};

struct LibertyTableInfo {
  StringRef templateName;
  SmallVector<StringRef, 3> indices;
  StringRef tableData;
};

struct TimingArcInfo {
  StringRef relatedPin;
  StringRef sense;
  StringRef type;
  LibertyTableInfo table;
};

struct LibPinInfo {};

struct LibCellInfo {};
*/

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

/// GroupInfo example:
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
/// GroupInfo grp;
/// grp.groupName = "timing";
/// grp.groupKey = "";
/// grp.attributes["related_pin"] = "A1";
/// auto subGroup1 = GroupInfo("cell_rise", "delay_outputslew_template_7X8");
/// auto subGroup2 = GroupInfo("cell_fall", "delay_outputslew_template_7X8");
/// grp.subGroups.emplace_back(std::move(subGroup1));
/// grp.subGroups.emplace_back(std::move(subGroup2));
///
// struct GroupInfo {
//   StringRef groupName; // like cell / pin / timing ...
//   StringRef groupKey;  // like AND / Z / DFF ... , nullable

//   std::vector<StringRef> defines;
//   LibertyValueMap attributes;
//   std::vector<GroupValue> subGroups;

//   static LibertyValueType newLibertyValue(StringRef groupName,
//                                           StringRef groupKey) {
//     return std::make_unique<GroupInfo>(groupName, groupKey);
//   }

//   GroupInfo(StringRef groupName = "", StringRef groupKey = "")
//       : groupName(groupName), groupKey(groupKey) {}
// };

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

// struct CellAST {
//   CellAST(StringRef cellName) : cellName(cellName) {}

//   StringRef cellName;
//   GroupMap pins;
//   CommonGroupAST group;

//   const GroupMap &getPins() const { return pins; }
//   const CommonGroupAST &getGroupInfo() const { return group; }
// };

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
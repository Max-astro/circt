add_circt_dialect_library(CIRCTLibertyImport
  LibertyLexer.cpp
  LibertyParser.cpp
  LibertyImport.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSupport
#   MLIRIR
#   MLIRPass
#   MLIRTransformUtils
)

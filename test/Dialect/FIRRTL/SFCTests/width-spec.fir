; RUN: firtool --split-input-file %s --ir-fir | FileCheck %s
; Tests extracted from:
; - test/scala/firrtlTests/WidthSpec.scala

; Literal width checks
FIRRTL version 3.0.0
circuit Foo :
  ; CHECK-LABEL: firrtl.module @Foo(
  ; CHECK-SAME: out %si0: !firrtl.sint<3>
  ; CHECK-SAME: out %si1: !firrtl.sint<3>
  ; CHECK-SAME: out %si2: !firrtl.sint<2>
  ; CHECK-SAME: out %si3: !firrtl.sint<1>
  ; CHECK-SAME: out %si4: !firrtl.sint<1>
  ; CHECK-SAME: out %si5: !firrtl.sint<2>
  ; CHECK-SAME: out %si6: !firrtl.sint<3>
  ; CHECK-SAME: out %si7: !firrtl.sint<3>
  ; CHECK-SAME: out %si8: !firrtl.sint<4>
  ; CHECK-SAME: out %ui0: !firrtl.uint<1>
  ; CHECK-SAME: out %ui1: !firrtl.uint<1>
  ; CHECK-SAME: out %ui2: !firrtl.uint<2>
  ; CHECK-SAME: out %ui3: !firrtl.uint<2>
  ; CHECK-SAME: out %ui4: !firrtl.uint<3>
  module Foo :
    output si0 : SInt
    output si1 : SInt
    output si2 : SInt
    output si3 : SInt
    output si4 : SInt
    output si5 : SInt
    output si6 : SInt
    output si7 : SInt
    output si8 : SInt

    output ui0 : UInt
    output ui1 : UInt
    output ui2 : UInt
    output ui3 : UInt
    output ui4 : UInt

    connect si0, SInt(-4)
    connect si1, SInt(-3)
    connect si2, SInt(-2)
    connect si3, SInt(-1)
    connect si4, SInt(0)
    connect si5, SInt(1)
    connect si6, SInt(2)
    connect si7, SInt(3)
    connect si8, SInt(4)

    connect ui0, UInt(0)
    connect ui1, UInt(1)
    connect ui2, UInt(2)
    connect ui3, UInt(3)
    connect ui4, UInt(4)

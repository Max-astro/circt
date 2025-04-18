; RUN: firtool %s --format=fir --ir-fir              | circt-opt | FileCheck %s --check-prefix=OPT
; RUN: firtool %s --format=fir --ir-fir -disable-opt | circt-opt | FileCheck %s --check-prefix=NOOPT

FIRRTL version 4.0.0
circuit test_cse :
  public module test_cse :
    input a1: UInt<4>
    input a2: UInt<4>
    output b: UInt<5>
    output c: UInt<5>
    output d: UInt<5>
    output e: UInt<5>
    connect b, add(a1, a2)
    connect c, add(a1, a2)
    connect d, and(a1, UInt<4>(0))
    connect e, and(UInt<4>(3), UInt<4>(1))

; OPT-DAG: %c0_ui5 = firrtl.constant 0 : !firrtl.uint<5>
; OPT-DAG: %c1_ui5 = firrtl.constant 1 : !firrtl.uint<5>

; Only one add.
; OPT: %0 = firrtl.add %a1, %a2
; OPT: firrtl.matchingconnect %b, %0
; OPT: firrtl.matchingconnect %c, %0

; Connect with zero and one directly.
; OPT: firrtl.matchingconnect %d, %c0_ui5
; OPT: firrtl.matchingconnect %e, %c1_ui5

; Both adds persist.
; NOOPT: %0 = firrtl.add %a1, %a2
; NOOPT: firrtl.matchingconnect %b, %0
; NOOPT: %1 = firrtl.add %a1, %a2
; NOOPT: firrtl.matchingconnect %c, %1

; Ands persist.
; NOOPT: %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
; NOOPT: firrtl.and %a1, %c0_ui4
; NOOPT: firrtl.matchingconnect %d,

; NOOPT-DAG: %c3_ui4 = firrtl.constant 3 : !firrtl.uint<4>
; NOOPT-DAG: %c1_ui4 = firrtl.constant 1 : !firrtl.uint<4>
; NOOPT: firrtl.and %c3_ui4, %c1_ui4
; NOOPT: firrtl.matchingconnect %e,

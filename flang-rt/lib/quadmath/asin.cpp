//===-- lib/quadmath/asin.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math-entries.h"

namespace Fortran::runtime {
extern "C" {

#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(AsinF128)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Asin<true>::invoke(x);
}
#endif

} // extern "C"
} // namespace Fortran::runtime

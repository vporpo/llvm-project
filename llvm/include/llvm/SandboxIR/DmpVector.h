//===- DmpVector.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A DmpVector is a vector that you can dump().
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SANDBOXIR_DMPVECTOR_H
#define LLVM_TRANSFORMS_SANDBOXIR_DMPVECTOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace sandboxir {
class Value;
class Instruction;
class BasicBlock;
} // namespace sandboxir

/// Just like a small vector but with dump() and operator<<.
template <typename T> class DmpVectorBase : public SmallVector<T> {
public:
  DmpVectorBase() : SmallVector<T>() {}
  DmpVectorBase(std::initializer_list<T> Vals) : SmallVector<T>(Vals) {}
  DmpVectorBase(ArrayRef<T> Vals) : SmallVector<T>(Vals) {}
  DmpVectorBase(SmallVector<T> &&Vals) : SmallVector<T>(std::move(Vals)) {}
  DmpVectorBase(DmpVectorBase &&Other) : SmallVector<T>(std::move(Other)) {}
  DmpVectorBase(const DmpVectorBase &Other) : SmallVector<T>(Other) {}
  explicit DmpVectorBase(size_t Sz) { this->grow(Sz); }
  template <typename ItT>
  DmpVectorBase(ItT Begin, ItT End) : SmallVector<T>(Begin, End) {}
  DmpVectorBase &operator=(const DmpVectorBase &Other) {
    SmallVector<T>::operator=(Other);
    return *this;
  }
  void reserve(size_t Sz) { this->grow(Sz); }
  hash_code hash() const {
    return hash_combine_range(this->begin(), this->end());
  }
  friend hash_code hash_value(const DmpVectorBase &B) { return B.hash(); }
  raw_ostream &dump(raw_ostream &OS) const {
    for (const T V : *this) {
      if (V == nullptr)
        OS << "NULL";
      else
        OS << *V;
      OS << "\n";
    }
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
  friend raw_ostream &operator<<(raw_ostream &OS, const DmpVectorBase<T> &Vec) {
    return Vec.dump(OS);
  }
};

template <typename T> class DmpVector : public DmpVectorBase<T> {
public:
  DmpVector() : DmpVectorBase<T>() {}
  DmpVector(std::initializer_list<T> Vals) : DmpVectorBase<T>(Vals) {}
  DmpVector(ArrayRef<T> Vals) : DmpVectorBase<T>(Vals) {}
  DmpVector(SmallVector<T> &&Vals) : DmpVectorBase<T>(std::move(Vals)) {}
  DmpVector(DmpVector &&Other) : DmpVectorBase<T>(std::move(Other)) {}
  DmpVector(const DmpVector &Other) : DmpVectorBase<T>(std::move(Other)) {}
  explicit DmpVector(size_t Sz) : DmpVectorBase<T>(Sz) {}
  template <typename ItT>
  DmpVector(ItT Begin, ItT End) : DmpVectorBase<T>(Begin, End) {}
  DmpVector &operator=(const DmpVector &Other) {
    DmpVectorBase<T>::operator=(Other);
    return *this;
  }
};

/// An immutable view of a DmpVector with dump().
/// This inherits from ArrayRef, so it has a similar API.
template <typename T> class DmpVectorView : public ArrayRef<T> {
public:
  /// DmpVector constructor.
  DmpVectorView(const DmpVector<T> &Vec) : ArrayRef<T>(Vec) {}
  /// Range constructor.
  DmpVectorView(T *Begin, T *End) : ArrayRef<T>(Begin, End) {}
  /// ArrayRef constructor.
  DmpVectorView(const ArrayRef<T> &Array) : ArrayRef<T>(Array) {}
  /// Default constructor.
  DmpVectorView() : ArrayRef<T>() {}

  hash_code hash() const {
    return hash_combine_range(this->begin(), this->end());
  }
  friend hash_code hash_value(const DmpVectorView &View) { return View.hash(); }
  raw_ostream &dump(raw_ostream &OS) const {
    for (const T V : *this) {
      if (V == nullptr)
        OS << "NULL";
      else
        OS << *V;
      OS << "\n";
    }
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
  friend raw_ostream &operator<<(raw_ostream &OS, const DmpVectorView<T> &Vec) {
    return Vec.dump(OS);
  }
};

/// Spcialization for Value*
template <> class DmpVector<Value *> : public DmpVectorBase<Value *> {
public:
  DmpVector<Value *>() : DmpVectorBase() {}
  DmpVector<Value *>(std::initializer_list<Value *> Vals)
      : DmpVectorBase(Vals) {}
  DmpVector<Value *>(ArrayRef<Value *> Vals) : DmpVectorBase(Vals) {}
  DmpVector<Value *>(SmallVector<Value *> &&Vals)
      : DmpVectorBase(std::move(Vals)) {}
  DmpVector<Value *>(DmpVector<Value *> &&Other)
      : DmpVectorBase(std::move(Other)) {}
  DmpVector<Value *>(const DmpVector<Value *> &Other) : DmpVectorBase(Other) {}
  explicit DmpVector<Value *>(size_t Sz) : DmpVectorBase(Sz) {}
  template <typename ItT>
  DmpVector<Value *>(ItT Begin, ItT End) : DmpVectorBase(Begin, End) {}
  DmpVector<Value *> &operator=(const DmpVector<Value *> &Other) {
    DmpVectorBase<Value *>::operator=(Other);
    return *this;
  }
  static DmpVector<Value *>
  create(const DmpVector<sandboxir::Value *> &SBVec);

  using instr_iterator = mapped_iterator<iterator, Instruction *(*)(Value *)>;
  using const_instr_iterator =
      mapped_iterator<const_iterator, Instruction *(*)(Value *)>;

  static Instruction *valueToInstr(Value *V) { return cast<Instruction>(V); }
  instr_iterator ibegin() { return map_iterator(begin(), valueToInstr); }
  instr_iterator iend() { return map_iterator(end(), valueToInstr); }
  const_instr_iterator ibegin() const {
    return map_iterator<const_iterator, Instruction *(*)(Value *)>(
        begin(), valueToInstr);
  }
  const_instr_iterator iend() const {
    return map_iterator<const_iterator, Instruction *(*)(Value *)>(
        end(), valueToInstr);
  }
  using instr_range_t = iterator_range<instr_iterator>;
  instr_range_t instrRange() { return make_range(ibegin(), iend()); }
  using const_instr_range_t = iterator_range<const_instr_iterator>;
  const_instr_range_t instrRange() const {
    return make_range(ibegin(), iend());
  }
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<DmpVector<Value *>> {
  static inline DmpVector<Value *> getEmptyKey() {
    return DmpVector<Value *>((Value *)-1);
  }
  static inline DmpVector<Value *> getTombstoneKey() {
    return DmpVector<Value *>((Value *)-2);
  }
  static unsigned getHashValue(const DmpVector<Value *> &B) { return B.hash(); }
  static bool isEqual(const DmpVector<Value *> &B1,
                      const DmpVector<Value *> &B2) {
    return B1 == B2;
  }
};

/// Spcialization for sandboxir::Value*
template <>
class DmpVector<sandboxir::Value *>
    : public DmpVectorBase<sandboxir::Value *> {
  void init(const DmpVector<Value *> &Vec, const sandboxir::BasicBlock &SBBB);

public:
  DmpVector<sandboxir::Value *>() : DmpVectorBase<sandboxir::Value *>() {}
  DmpVector<sandboxir::Value *>(
      std::initializer_list<sandboxir::Value *> Instrs)
      : DmpVectorBase(Instrs) {}
  DmpVector<sandboxir::Value *>(ArrayRef<sandboxir::Value *> Instrs)
      : DmpVectorBase(Instrs) {}
  DmpVector<sandboxir::Value *>(SmallVector<sandboxir::Value *> &&Instrs)
      : DmpVectorBase(std::move(Instrs)) {}
  DmpVector<sandboxir::Value *>(DmpVector<sandboxir::Value *> &&Other)
      : DmpVectorBase<sandboxir::Value *>(std::move(Other)) {}
  DmpVector<sandboxir::Value *>(const DmpVector<sandboxir::Value *> &Other)
      : DmpVectorBase<sandboxir::Value *>(Other) {}
  explicit DmpVector<sandboxir::Value *>(size_t Sz) : DmpVectorBase(Sz) {}
  template <typename ItT>
  DmpVector<sandboxir::Value *>(ItT Begin, ItT End)
      : DmpVectorBase(Begin, End) {}
  DmpVector<sandboxir::Value *>(const DmpVector<Value *> &Vec,
                                  const sandboxir::BasicBlock &SBBB) {
    init(Vec, SBBB);
  }
  DmpVector<sandboxir::Value *> &
  operator=(const DmpVector<sandboxir::Value *> &Other) {
    DmpVectorBase<sandboxir::Value *>::operator=(Other);
    return *this;
  }
  DmpVector<Value *> getLLVMValueVector() const;

  using instr_iterator =
      mapped_iterator<iterator,
                      sandboxir::Instruction *(*)(sandboxir::Value *)>;
  using const_instr_iterator =
      mapped_iterator<const_iterator,
                      sandboxir::Instruction *(*)(sandboxir::Value *)>;

  static sandboxir::Instruction *valueToInstr(sandboxir::Value *V) {
    return cast<sandboxir::Instruction>(V);
  }
  instr_iterator ibegin() { return map_iterator(begin(), valueToInstr); }
  instr_iterator iend() { return map_iterator(end(), valueToInstr); }
  const_instr_iterator ibegin() const {
    return map_iterator<const_iterator,
                        sandboxir::Instruction *(*)(sandboxir::Value *)>(
        begin(), valueToInstr);
  }
  const_instr_iterator iend() const {
    return map_iterator<const_iterator,
                        sandboxir::Instruction *(*)(sandboxir::Value *)>(
        end(), valueToInstr);
  }
  using instr_range_t = iterator_range<instr_iterator>;
  instr_range_t instrRange() { return make_range(ibegin(), iend()); }
  using const_instr_range_t = iterator_range<const_instr_iterator>;
  const_instr_range_t instrRange() const {
    return make_range(ibegin(), iend());
  }
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<DmpVector<sandboxir::Value *>> {
  static inline DmpVector<sandboxir::Value *> getEmptyKey() {
    return DmpVector<sandboxir::Value *>((sandboxir::Value *)-1);
  }
  static inline DmpVector<sandboxir::Value *> getTombstoneKey() {
    return DmpVector<sandboxir::Value *>((sandboxir::Value *)-2);
  }
  static unsigned getHashValue(const DmpVector<sandboxir::Value *> &B) {
    return B.hash();
  }
  static bool isEqual(const DmpVector<sandboxir::Value *> &B1,
                      const DmpVector<sandboxir::Value *> &B2) {
    return B1 == B2;
  }
};

/// Spcialization for sandboxir::Instruction*
template <>
class DmpVector<sandboxir::Instruction *>
    : public DmpVectorBase<sandboxir::Instruction *> {
  void init(const DmpVector<Value *> &Vec, const sandboxir::BasicBlock &SBBB);

public:
  DmpVector<sandboxir::Instruction *>()
      : DmpVectorBase<sandboxir::Instruction *>() {}
  DmpVector<sandboxir::Instruction *>(
      std::initializer_list<sandboxir::Instruction *> Instrs)
      : DmpVectorBase(Instrs) {}
  DmpVector<sandboxir::Instruction *>(
      ArrayRef<sandboxir::Instruction *> Instrs)
      : DmpVectorBase(Instrs) {}
  DmpVector<sandboxir::Instruction *>(
      SmallVector<sandboxir::Instruction *> &&Instrs)
      : DmpVectorBase(std::move(Instrs)) {}
  DmpVector<sandboxir::Instruction *>(
      DmpVector<sandboxir::Instruction *> &&Other)
      : DmpVectorBase<sandboxir::Instruction *>(std::move(Other)) {}
  DmpVector<sandboxir::Instruction *>(
      const DmpVector<sandboxir::Instruction *> &Other)
      : DmpVectorBase<sandboxir::Instruction *>(Other) {}
  explicit DmpVector<sandboxir::Instruction *>(size_t Sz)
      : DmpVectorBase(Sz) {}
  template <typename ItT>
  DmpVector<sandboxir::Instruction *>(ItT Begin, ItT End)
      : DmpVectorBase(Begin, End) {}
  DmpVector<sandboxir::Instruction *>(const DmpVector<Value *> &Vec,
                                        const sandboxir::BasicBlock &SBBB) {
    init(Vec, SBBB);
  }
  DmpVector<sandboxir::Instruction *> &
  operator=(const DmpVector<sandboxir::Instruction *> &Other) {
    DmpVectorBase<sandboxir::Instruction *>::operator=(Other);
    return *this;
  }
  DmpVector<sandboxir::Value *> getOperandBundle(unsigned OpIdx) const;
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<DmpVector<sandboxir::Instruction *>> {
  static inline DmpVector<sandboxir::Instruction *> getEmptyKey() {
    return DmpVector<sandboxir::Instruction *>(
        (sandboxir::Instruction *)-1);
  }
  static inline DmpVector<sandboxir::Instruction *> getTombstoneKey() {
    return DmpVector<sandboxir::Instruction *>(
        (sandboxir::Instruction *)-2);
  }
  static unsigned
  getHashValue(const DmpVector<sandboxir::Instruction *> &Vec) {
    return Vec.hash();
  }
  static bool isEqual(const DmpVector<sandboxir::Instruction *> &Vec1,
                      const DmpVector<sandboxir::Instruction *> &Vec2) {
    return Vec1 == Vec2;
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SANDBOXIR_DMPVECTOR_H

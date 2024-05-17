//===- Bundle.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A ValueBundle is a collection of Values that we are planning to vectorize.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_BUNDLE_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_BUNDLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class SBValue;
class SBInstruction;
class SBBasicBlock;

template <typename T> class Bundle {
protected:
  SmallVector<T> Vals;

public:
  using value_type = T;
  Bundle() = default;
  Bundle(std::initializer_list<T> Vals) : Vals(Vals) {}
  Bundle(ArrayRef<T> Vals) : Vals(Vals) {}
  explicit Bundle(T Val) : Vals({Val}) {}
  Bundle(SmallVector<T> &&Vals) : Vals(std::move(Vals)) {}
  Bundle(Bundle &&Other) = default;
  Bundle(const Bundle &Other) = default;
  explicit Bundle(size_t Sz) { Vals.reserve(Sz); }
  Bundle &operator=(const Bundle &Other) = default;
  template <typename ItT> Bundle(ItT Begin, ItT End) : Vals(Begin, End) {}
  ArrayRef<T> get() const { return Vals; }
  using iterator = typename decltype(Vals)::iterator;
  using const_iterator = typename decltype(Vals)::const_iterator;

  iterator begin() { return Vals.begin(); }
  iterator end() { return Vals.end(); }
  const_iterator begin() const { return Vals.begin(); }
  const_iterator end() const { return Vals.end(); }
  T front() const { return Vals.front(); }
  T back() const { return Vals.back(); }

  iterator erase(iterator It) { return Vals.erase(It); }
  iterator erase(iterator ItB, iterator ItE) { return Vals.erase(ItB, ItE); }
  void pop_back() { Vals.pop_back(); }
  uint32_t size() const { return Vals.size(); }
  bool empty() const { return Vals.empty(); }
  void reserve(size_t Sz) { Vals.reserve(Sz); }
  void resize(size_t Sz) { Vals.resize(Sz); }
  void push_back(T V) { Vals.push_back(V); }
  iterator insert(iterator Where, T &&V) {
    return Vals.insert(Where, std::move(V));
  }
  iterator insert(iterator Where, const T &V) { return Vals.insert(Where, V); }
  hash_code hash() const { return hash_combine_range(begin(), end()); }
  friend hash_code hash_value(const Bundle &B) { return B.hash(); }
  T &operator[](unsigned Idx) { return Vals[Idx]; }
  T operator[](unsigned Idx) const { return Vals[Idx]; }
  bool operator==(const Bundle &Other) const {
    return size() == Other.size() && equal(Vals, Other.Vals);
  }
  template <typename OtherIterableT>
  bool operator==(const OtherIterableT &Other) const {
    return size() == Other.size() && equal(Vals, Other);
  }
  raw_ostream &dump(raw_ostream &OS) const {
    for (const T V : Vals) {
      if (V == nullptr)
        OS << "NULL";
      else
        OS << *V;
      OS << "\n";
    }
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
  friend raw_ostream &operator<<(raw_ostream &OS, const Bundle<T> &Bndl) {
    return Bndl.dump(OS);
  }
};

// An immutable view of a Bundle.
template <typename T> class BundleViewBase {
protected:
  const Bundle<T> &Bndl;
  typename Bundle<T>::iterator Begin;
  typename Bundle<T>::iterator End;

public:
  BundleViewBase() = default;
  explicit BundleViewBase(const Bundle<T> &Bndl) : Bndl(Bndl) {}
  BundleViewBase(const Bundle<T> &Bndl, unsigned FromIdx, unsigned ToIdx)
      : Bndl(Bndl), Begin(Bndl.begin() + FromIdx), End(Bndl.begin() + ToIdx) {
    assert(FromIdx < Bndl.size() && ToIdx < Bndl.size() &&
           "FromIdx or ToIdx out of bounds!");
  }
  BundleViewBase &operator=(const BundleViewBase &Other) = default;
  template <typename ItT>
  BundleViewBase(const Bundle<T> &Bndl, ItT Begin, ItT End)
      : Bndl(Bndl), Begin(Begin), End(End) {}
  const Bundle<T> &get() const { return Bndl; }
  using iterator = typename Bundle<T>::iterator;
  using const_iterator = typename Bundle<T>::const_iterator;

  iterator begin() { return Begin; }
  iterator end() { return End; }
  const_iterator begin() const { return Begin; }
  const_iterator end() const { return End; }
  T front() const { return *Begin; }
  T back() const { return *std::prev(End); }
  uint32_t size() const { return End - Begin; }
  bool empty() const { return size() == 0; }
  hash_code hash() const { return hash_combine_range(begin(), end()); }
  friend hash_code hash_value(const BundleViewBase &B) { return B.hash(); }
  T &operator[](unsigned Idx) { return *(begin() + Idx); }
  T operator[](unsigned Idx) const { return *(begin() + Idx); }
  iterator_range<iterator> range() const { return make_range(begin(), end()); }
  bool operator==(const BundleViewBase &Other) const {
    return size() == Other.size() && equal(range(), Other.range());
  }
  raw_ostream &dump(raw_ostream &OS) const {
    for (const T V : range()) {
      if (V == nullptr)
        OS << "NULL";
      else
        OS << *V;
      OS << "\n";
    }
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const { dump(dbgs()); }
  friend raw_ostream &operator<<(raw_ostream &OS, const Bundle<T> &Bndl) {
    return Bndl.dump(OS);
  }
};

static Instruction *valueToInstr(Value *V) { return cast<Instruction>(V); }

class SBValBundle;

class ValueBundle : public Bundle<Value *> {
public:
  ValueBundle() = default;
  ValueBundle(std::initializer_list<Value *> Vals) : Bundle(Vals) {}
  ValueBundle(ArrayRef<Value *> Vals) : Bundle(Vals) {}
  ValueBundle(SmallVector<Value *> &&Vals) : Bundle(std::move(Vals)) {}
  ValueBundle(Value *V) : Bundle(V) {}
  ValueBundle(ValueBundle &&Other) = default;
  ValueBundle(const ValueBundle &Other) = default;
  ValueBundle(size_t Sz) : Bundle(Sz) {}
  ValueBundle &operator=(const ValueBundle &Other) = default;
  template <typename ItT>
  ValueBundle(ItT Begin, ItT End) : Bundle(Begin, End) {}
  static ValueBundle create(const SBValBundle &SBBndl);

  using instr_iterator = mapped_iterator<iterator, Instruction *(*)(Value *)>;
  using const_instr_iterator =
      mapped_iterator<const_iterator, Instruction *(*)(Value *)>;

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
template <> struct DenseMapInfo<ValueBundle> {
  static inline ValueBundle getEmptyKey() { return ValueBundle((Value *)-1); }
  static inline ValueBundle getTombstoneKey() {
    return ValueBundle((Value *)-2);
  }
  static unsigned getHashValue(const ValueBundle &B) { return B.hash(); }
  static bool isEqual(const ValueBundle &B1, const ValueBundle &B2) {
    return B1 == B2;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const SBValue &SBV);

class SBValBundle : public Bundle<SBValue *> {
public:
  SBValBundle() = default;
  SBValBundle(std::initializer_list<SBValue *> Nodes) : Bundle(Nodes) {}
  SBValBundle(ArrayRef<SBValue *> Nodes) : Bundle(Nodes) {}
  SBValBundle(SmallVector<SBValue *> &&Nodes) : Bundle(std::move(Nodes)) {}
  SBValBundle(SBValBundle &&Other) = default;
  SBValBundle(const SBValBundle &Other) = default;
  SBValBundle(size_t Sz) : Bundle(Sz) {}
  SBValBundle &operator=(const SBValBundle &Other) = default;
  template <typename ItT>
  SBValBundle(ItT Begin, ItT End) : Bundle(Begin, End) {}
  SBValBundle(const ValueBundle &Bndl, const SBBasicBlock &SBBB);

  SBValBundle getOperandBundle(unsigned OpIdx) const;
  SmallVector<SBValBundle, 2> getOperandBundles() const;
  ValueBundle getValueBundle() const;
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<SBValBundle> {
  static inline SBValBundle getEmptyKey() {
    return SBValBundle((SBValue *)-1);
  }
  static inline SBValBundle getTombstoneKey() {
    return SBValBundle((SBValue *)-2);
  }
  static unsigned getHashValue(const SBValBundle &B) { return B.hash(); }
  static bool isEqual(const SBValBundle &B1, const SBValBundle &B2) {
    return B1 == B2;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const SBInstruction &SBI);

class SBInstrBundle : public Bundle<SBInstruction *> {
public:
  SBInstrBundle() = default;
  SBInstrBundle(std::initializer_list<SBInstruction *> Nodes)
      : Bundle(Nodes) {}
  SBInstrBundle(ArrayRef<SBInstruction *> Nodes) : Bundle(Nodes) {}
  SBInstrBundle(SmallVector<SBInstruction *> &&Nodes)
      : Bundle(std::move(Nodes)) {}
  SBInstrBundle(SBInstrBundle &&Other) = default;
  SBInstrBundle(const SBInstrBundle &Other) = default;
  SBInstrBundle(size_t Sz) : Bundle(Sz) {}
  SBInstrBundle &operator=(const SBInstrBundle &Other) = default;
  template <typename ItT>
  SBInstrBundle(ItT Begin, ItT End) : Bundle(Begin, End) {}
  SBInstrBundle(const ValueBundle &Bndl, const SBBasicBlock &SBBB);

  SBValBundle getOperandBundle(unsigned OpIdx) const;
};

/// Traits for DenseMap.
template <> struct DenseMapInfo<SBInstrBundle> {
  static inline SBInstrBundle getEmptyKey() {
    return SBInstrBundle((SBInstruction *)-1);
  }
  static inline SBInstrBundle getTombstoneKey() {
    return SBInstrBundle((SBInstruction *)-2);
  }
  static unsigned getHashValue(const SBInstrBundle &B) { return B.hash(); }
  static bool isEqual(const SBInstrBundle &B1, const SBInstrBundle &B2) {
    return B1 == B2;
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_BUNDLE_H

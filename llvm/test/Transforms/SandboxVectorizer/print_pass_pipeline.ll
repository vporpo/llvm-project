; RUN: opt -passes=sandbox-vectorizer -sbvec-passes=boup-vectorize,dump-region,dump-region -disable-output -sbvec-print-pass-pipeline %s 2>&1 | FileCheck %s

; CHECK: bb-pass-manager(boup-vectorize(dump-region,dump-region,accept-or-revert))
define void @foo() {
  ret void
}

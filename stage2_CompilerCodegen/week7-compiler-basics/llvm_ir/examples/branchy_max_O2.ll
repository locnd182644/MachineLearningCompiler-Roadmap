; ModuleID = 'branchy_max.c'
source_filename = "branchy_max.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local i32 @max(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define dso_local i32 @clamp(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = icmp slt i32 %0, %1
  %5 = icmp sgt i32 %0, %2
  %6 = select i1 %5, i32 %2, i32 %0
  %7 = select i1 %4, i32 %1, i32 %6
  ret i32 %7
}

define dso_local i32 @abs_val(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i32 @llvm.abs.i32(i32 %0, i1 true)
  ret i32 %2
}

declare i32 @llvm.abs.i32(i32, i1 immarg) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable "frame-pointer"="none" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

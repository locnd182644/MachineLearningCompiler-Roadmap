; ModuleID = 'loop_sum.c'
source_filename = "loop_sum.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local i32 @loop_sum(ptr nocapture noundef readonly %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %15

4:                                                ; preds = %2
  %5 = zext i32 %1 to i64
  br label %6

6:                                                ; preds = %4, %6
  %7 = phi i64 [ 0, %4 ], [ %12, %6 ]
  %8 = phi i32 [ 0, %4 ], [ %11, %6 ]
  %9 = getelementptr inbounds i32, ptr %0, i64 %7
  %10 = load i32, ptr %9, align 4
  %11 = add nsw i32 %8, %10
  %12 = add nuw nsw i64 %7, 1
  %13 = icmp eq i64 %12, %5
  br i1 %13, label %14, label %6

14:                                               ; preds = %6
  br label %15

15:                                               ; preds = %14, %2
  %16 = phi i32 [ 0, %2 ], [ %11, %14 ]
  ret i32 %16
}

define dso_local i32 @dot_product(ptr nocapture noundef readonly %0, ptr nocapture noundef readonly %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %20

5:                                                ; preds = %3
  %6 = zext i32 %2 to i64
  br label %7

7:                                                ; preds = %5, %7
  %8 = phi i64 [ 0, %5 ], [ %17, %7 ]
  %9 = phi i32 [ 0, %5 ], [ %16, %7 ]
  %10 = getelementptr inbounds i32, ptr %0, i64 %8
  %11 = load i32, ptr %10, align 4
  %12 = getelementptr inbounds i32, ptr %1, i64 %8
  %13 = load i32, ptr %12, align 4
  %14 = mul nsw i32 %13, %11
  %15 = add nsw i32 %9, %14
  %16 = add nuw nsw i64 %8, 1
  %17 = icmp eq i64 %16, %6
  br i1 %17, label %18, label %7

18:                                               ; preds = %7
  br label %20

20:                                               ; preds = %18, %3
  %21 = phi i32 [ 0, %3 ], [ %15, %18 ]
  ret i32 %21
}

attributes #0 = { nofree norecurse nosync nounwind readonly uwtable "frame-pointer"="none" }

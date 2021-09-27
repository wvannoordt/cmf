#ifndef TEST_KERNEL_H
#define TEST_KERNEL_H
#include "cmf.h"
#include "box.h"

void FillBlockGpu(cmf::BlockArray<double, 1>& arLb, const box& bx);

#endif
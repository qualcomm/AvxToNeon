/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.
 * Description: avx2neon unit test main
 * Author: xuqimeng
 * Create: 2019-11-05
*/

#include <stdio.h>
#include <string.h>
#include "a2ntest.h"
#define MAX_ERR_MSG_LEN 50

static unsigned int errTsNum;;
void print_error_messages(char (*err_msg)[MAX_ERR_MSG_LEN], unsigned int num);

int main()
{
    unsigned int i;
    int passCount = 0;
    int failCount = 0;
    char err_msg[UT_MM_MAX][MAX_ERR_MSG_LEN];
    for (i = UT_MM_EXTRACT_EPI32; i < UT_MM_MAX; i++) {
        int flag = 0;
        const char *s = RunTest((InstructionTest)i, &flag);
        printf("Running Test %s\n", s);
        if (flag) {
            passCount++;
        } else {
            printf("**FAILURE** AVX2NEONTest %s\n", s);
            failCount++;
            strncpy_s(err_msg[failCount - 1], MAX_ERR_MSG_LEN, s, _TRUNCATE);
            errTsNum++;
        }
    }
    printf("AVX2NEONTest Complete: Passed %d tests : Failed %d\n", passCount, failCount);
    print_error_messages(err_msg, errTsNum);
    return 0;
}

void print_error_messages(char (*err_msg)[MAX_ERR_MSG_LEN], unsigned int num)
{
    for (unsigned int i = 0; i < num; ++i)
    {
        if (err_msg[i][0] == '\0') {
            break;
        }
        printf("AVX2NEONTest Error Test Case %d: %s\n", i, err_msg[i]);
    }
}

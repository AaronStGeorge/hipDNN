/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/LoggingUtils.hpp>

#define BINARY_ADD_PLUGIN_TESTS "binary_add_integration_test"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn_sdk::test_utilities::initializeSpdlogDefaultLogger(BINARY_ADD_PLUGIN_TESTS);

    return RUN_ALL_TESTS();
}

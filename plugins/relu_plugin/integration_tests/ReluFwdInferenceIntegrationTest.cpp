// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceImplementation.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

namespace
{

struct Relu2dTestCase
{
    int64_t n;
    int64_t m;

    friend std::ostream& operator<<(std::ostream& ss, const Relu2dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " m:" << tc.m << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, m};
    }
};

struct ReluTensorBundle
{
    ReluTensorBundle(const std::vector<int64_t>& dims,
                            unsigned int seed = 1)
        : derivedDims({1, dims[1], 1, 1})
        , xTensor(dims)
        , yTensor(dims)
    {
        xTensor.fillWithRandomValues(
            -1.0f, 1.0f, seed);
        yTensor.fillWithRandomValues(
            -100.0f, 100.0f, seed);
    }

    std::vector<int64_t> derivedDims;
    PinnedTensor<float> xTensor;
    PinnedTensor<float> yTensor;
};

} // namespace

class ReluForwardInferenceIntegrationTest
    : public ::testing::TestWithParam<Relu2dTestCase>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Uncomment if you want debug logging info.
        setenv("HIPDNN_LOG_LEVEL", "info", 1);

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        //Note: The plugin paths has to be set before we create the hipdnn handle.
        const std::array<const char*, 1> paths = {PLUGIN_DIR};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        // Create handle
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

        //todo: bring back stream support once MigratableMemory supports it
        //ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
        //ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    static std::unordered_map<int64_t, void*>
        createVariantPack(const graph::TensorAttributes& xTensorAttr,
                          const graph::TensorAttributes& yTensorAttr,
                          ReluTensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[yTensorAttr.get_uid()] = tensorBundle.yTensor.memory().deviceData();

        return variantPack;
    }

    void runReluFwd(
        ReluTensorBundle& graphTensorBundle,
        DataType_t inputDataType)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();

        graph->set_name("RELUInferenceTest");

        int64_t uid = 1;
        auto xAttr = graph::makeTensorAttributes("X", inputDataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        graph::PointwiseAttributes attributes;
        attributes.name = "PointwiseNode";
        attributes.set_mode(PointwiseMode_t::RELU_FWD);

        auto yTensorAttr = graph->pointwise(xTensorAttr, attributes);

        if(!yTensorAttr->has_uid())
        {
            HIPDNN_LOG_INFO("yTensorAttr does not have a UID, giving it a UID");
            yTensorAttr->set_uid(uid++);
        }

        yTensorAttr->set_data_type(inputDataType);

        yTensorAttr->set_output(true);

        // Validate and build graph
        auto result = graph->validate();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->create_execution_plans(_handle);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

        auto variantPack = createVariantPack(*xTensorAttr,
                                                                          *yTensorAttr,
                                                                          graphTensorBundle);

        result = graph->execute(_handle, variantPack, nullptr);
        ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    }

    static void runCpuReluFwd(ReluTensorBundle& cpuTensorBundle)
    {
        auto* input = cpuTensorBundle.xTensor.memory().hostData();
        auto* output = cpuTensorBundle.yTensor.memory().hostData();
        size_t size = cpuTensorBundle.xTensor.memory().count();

        for (size_t i = 0; i < size; i++) {
            output[i] = std::fmax(0.0f, input[i]);
        }
    }

    void runReluTest(const Relu2dTestCase& testCase,
                          float tolerance = 1e-4f)
    {
        auto inputDataType = getDataTypeEnumFromType<float>();

        unsigned int seed = std::random_device{}();
        //log the random seed in case we need to reproduce the test
        HIPDNN_LOG_INFO("Test is using {} for its random seed", seed);

        ReluTensorBundle graphTensorBundle(
            testCase.getDims(), seed);

        ReluTensorBundle cpuTensorBundle(
            testCase.getDims(), seed);

        runReluFwd(
            graphTensorBundle, inputDataType);
        graphTensorBundle.yTensor.memory().markDeviceModified();

        runCpuReluFwd(cpuTensorBundle);

        CpuFpReferenceValidation<float> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(cpuRefValidation.allClose(cpuTensorBundle.yTensor.memory(),
                                              graphTensorBundle.yTensor.memory()));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

namespace
{

std::vector<Relu2dTestCase> getReluFwdInferenceTestCases()
{
    return {
        {.n = 64, .m = 64},
        {.n = 64, .m = 128},
        {.n = 128, .m = 64},
    };
}

} // namespace

TEST_P(ReluForwardInferenceIntegrationTest, RunFloatFwdBatchnormGraphNCHW)
{
    Relu2dTestCase testCase = GetParam();
    runReluTest(testCase, 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(RunFloatFwdBatchnormGraph,
                         ReluForwardInferenceIntegrationTest,
                         testing::ValuesIn(getReluFwdInferenceTestCases()));

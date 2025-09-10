#include "hipdnn_sdk/data_objects/graph_generated.h"
#include "hipdnn_sdk/data_objects/tensor_attributes_generated.h"
#include "hipdnn_sdk/plugin/PluginApiDataTypes.h"
#include "hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp"
#include "hipdnn_sdk/plugin/PluginHelpers.hpp"
#include <flatbuffers/flatbuffers.h>
#include <fusilli/backend/backend.h>
#include <fusilli/graph/graph.h>
#include <fusilli/support/logging.h>
#include <hip/hip_runtime.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/plugin/PluginApi.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include "fusilli.h"

static const char* pluginName = "iree_plugin";
static const char* pluginVersion = "0.0.1";
static const int64_t ENGINE_ID = 1001;

#define UNWRAP_FUSILLI_ERROROR(expr)                                                               \
    ({                                                                                             \
        auto errorOr = (expr);                                                                     \
        if(isError(errorOr))                                                                       \
        {                                                                                          \
            throw hipdnn_plugin ::HipdnnPluginException(                                           \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, fusilli ::ErrorObject(errorOr).getMessage()); \
        }                                                                                          \
        std ::move(*errorOr);                                                                      \
    });

#define FUSILLI_REQUIRE(expr)                                                                \
    do                                                                                       \
    {                                                                                        \
        fusilli::ErrorObject err = (expr);                                                   \
        if(isError(err))                                                                     \
        {                                                                                    \
            throw hipdnn_plugin ::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, \
                                                        err.getMessage());                   \
        }                                                                                    \
    } while(false)

// s_lastError is thread_local static so can't be initialized in the header file
// as the header file is included in many context. Clear the string here.
// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

std::string getNodeName(const hipdnn_sdk::data_objects::Node& node)
{
    return node.name() != nullptr ? node.name()->str() : "";
}

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers)
{
    for(uint32_t i = 0; i < numDeviceBuffers; i++)
    {
        if(uid == deviceBuffers[i].uid)
        {
            return deviceBuffers[i];
        }
    }

    throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                               "Device buffer with the uid: " + std::to_string(uid)
                                                   + " not found in the provided device buffers.");
}

struct HipdnnEnginePluginHandle
{
public:
    void setStream(hipStream_t stream)
    {
        _stream = stream;
    }

    hipStream_t getStream() const
    {
        return _stream;
    }

    void storeEngineDetailsBuffer(const void* ptr,
                                  std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        _engineDetailsBuffers[ptr] = std::move(buffer);
    }

    void removeEngineDetailsBuffer(const void* ptr)
    {
        _engineDetailsBuffers.erase(ptr);
    }

private:
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engineDetailsBuffers;
};

struct HipdnnEnginePluginExecutionContext
{
	fusilli::Graph graph;

	// TODO: need info to map hipDNN varentpack to fusilli varentpack
};

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    *name = pluginName;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    *version = pluginVersion;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    *type = HIPDNN_PLUGIN_TYPE_ENGINE;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// NOLINTNEXTLINE
void hipdnnPluginGetLastErrorString(const char** error_str)
{

    LOG_API_ENTRY("errorStrPtr={:p}", static_cast<void*>(error_str));

    hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(error_str);

        *error_str = hipdnn_plugin::PluginLastErrorManager::getLastError();

        LOG_API_SUCCESS(apiName, "errorStr={:p}", static_cast<void*>(error_str));
    });
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(callback);
        hipdnn::logging::initializeCallbackLogging(pluginName, callback);
        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    LOG_API_ENTRY("engineIds={:p}, maxEngines={}, numEngines={:p}",
                  static_cast<void*>(engineIds),
                  maxEngines,
                  static_cast<void*>(numEngines));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        if(maxEngines != 0)
        {
            hipdnn_plugin::throwIfNull(engineIds);
        }
        hipdnn_plugin::throwIfNull(numEngines);

        // Set `numEngines` regardless of how many engines are actually returned.
        // The backend queries this function twice:
        // - First pass: engineIds=NULL, maxEngines=0 to get the count
        // - Second pass: engineIds allocated based on numEngines from first pass
        *numEngines = 1;

        if(maxEngines >= 1)
        {
            engineIds[0] = ENGINE_ID;
        }

        LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);

        *handle = new HipdnnEnginePluginHandle();

        LOG_API_SUCCESS(apiName, "createdHandle={:p}", static_cast<void*>(*handle));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);

        delete handle;
        handle = nullptr;

        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                 hipStream_t stream)
{
    LOG_API_ENTRY(
        "handle={:p}, stream_id={:p}", static_cast<void*>(handle), static_cast<void*>(stream));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);

        handle->setStream(stream);

        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* opGraph,
                                             int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t* numEngines)
{
    LOG_API_ENTRY("handle={:p}, opGraph={:p}, engineIds={:p}, maxEngines={}, numEngines={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(engineIds),
                  maxEngines,
                  static_cast<void*>(numEngines));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);
        hipdnn_plugin::throwIfNull(opGraph);
        if(maxEngines != 0)
        {
            hipdnn_plugin::throwIfNull(engineIds);
        }
        hipdnn_plugin::throwIfNull(numEngines);

        hipdnn_plugin::GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);

        *numEngines = 0;
        if(maxEngines < 1)
        {
            HIPDNN_LOG_INFO("Maximum number of engines reached ({}), ignoring additional "
                            "engines, numEngines count: {}",
                            maxEngines,
                            *numEngines);
            return;
        }

        bool shouldHandle = true; // TODO: check the graph
        if(shouldHandle)
        {
            engineIds[0] = ENGINE_ID;
            *numEngines = 1;
        }

        LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                        int64_t engineId,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle={:p}, engineId={}, opGraph={:p}, engineDetails={:p}",
                  static_cast<void*>(handle),
                  engineId,
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(engineDetails));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);
        hipdnn_plugin::throwIfNull(opGraph);
        hipdnn_plugin::throwIfNull(engineDetails);

        // Create a proper flatbuffer for engine details
        flatbuffers::FlatBufferBuilder builder;
        auto engineDetailsObj = hipdnn_sdk::data_objects::CreateEngineDetails(builder, engineId);
        builder.Finish(engineDetailsObj);

        auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
        engineDetails->ptr = detachedBuffer->data();
        engineDetails->size = detachedBuffer->size();

        // Store the buffer in the handle to keep it alive
        handle->storeEngineDetailsBuffer(engineDetails->ptr, std::move(detachedBuffer));

        LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                            hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle={:p}, engineDetails={}",
                  static_cast<void*>(handle),
                  static_cast<void*>(engineDetails));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);
        hipdnn_plugin::throwIfNull(engineDetails);
        hipdnn_plugin::throwIfNull(engineDetails->ptr);

        // Remove the buffer from the handle (this will deallocate it)
        handle->removeEngineDetailsBuffer(engineDetails->ptr);
        engineDetails->ptr = nullptr;
        engineDetails->size = 0;

        LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                                        const hipdnnPluginConstData_t* engineConfig,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        size_t* workspaceSize)
{
    LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, workspaceSize={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(engineConfig),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(workspaceSize));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);
        hipdnn_plugin::throwIfNull(engineConfig);
        hipdnn_plugin::throwIfNull(opGraph);
        hipdnn_plugin::throwIfNull(workspaceSize);

        *workspaceSize = 0;

        LOG_API_SUCCESS(apiName, "workspaceSize={}", *workspaceSize);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, executionContext={:p}",
                  static_cast<void*>(handle),
                  static_cast<const void*>(engineConfig),
                  static_cast<const void*>(opGraph),
                  static_cast<void*>(executionContext));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);
        hipdnn_plugin::throwIfNull(engineConfig);
        hipdnn_plugin::throwIfNull(opGraph);
        hipdnn_plugin::throwIfNull(executionContext);

        hipdnn_plugin::EngineConfigWrapper engineConfigWrapper(engineConfig->ptr,
                                                               engineConfig->size);
        if(engineConfigWrapper.engineId() != ENGINE_ID)
        {
            throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                       "unexpected engine id");
        }
        hipdnn_plugin::GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);

        // ===============================================================
        int64_t n = 16;
        int64_t c = 128;
        int64_t h = 64;
        int64_t w = 64;
        int64_t k = 256;
        int64_t r = 1;
        int64_t s = 1;

        fusilli::FusilliHandle handle
            = UNWRAP_FUSILLI_ERROROR(fusilli::FusilliHandle::create(fusilli::Backend::GFX942));
        fusilli::Graph graph = fusilli::Graph();

        graph.setName("fprop_sample");
        graph.setIODataType(fusilli::DataType::Float).setComputeDataType(fusilli::DataType::Float);

        auto xTensor = graph.tensor(fusilli::TensorAttr()
                                        .setName("image")
                                        .setDim({n, c, h, w})
                                        .setStride({c * h * w, h * w, w, 1}));

        auto wTensor = graph.tensor(fusilli::TensorAttr()
                                        .setName("filter")
                                        .setDim({k, c, r, s})
                                        .setStride({c * r * s, r * s, s, 1}));

        auto convAttr = fusilli::ConvFPropAttr()
                            .setPadding({0, 0})
                            .setStride({1, 1})
                            .setDilation({1, 1})
                            .setName("conv_fprop");

        auto yTensor = graph.convFProp(xTensor, wTensor, convAttr);

        // Specify Y's dimensions and strides
        yTensor->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
        yTensor->setOutput(true);

        FUSILLI_REQUIRE(graph.validate());

        FUSILLI_REQUIRE(graph.validate());

        FUSILLI_REQUIRE(graph.compile(handle, /*remove=*/true));
        // ===============================================================

        *executionContext = new HipdnnEnginePluginExecutionContext{.graph = std::move(graph)};

        LOG_API_SUCCESS(
            apiName, "created_execution_context={:p}", static_cast<void*>(*executionContext));
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                              hipdnnEnginePluginExecutionContext_t executionContext)
{
    LOG_API_ENTRY("handle={:p}, executionContext={:p}",
                  static_cast<void*>(handle),
                  static_cast<void*>(executionContext));

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);
        hipdnn_plugin::throwIfNull(executionContext);

        delete executionContext;

        LOG_API_SUCCESS(apiName, "destroyed executionContext");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers)
{
    LOG_API_ENTRY("handle={:p}, executionContext={:p}, workspace={:p}, deviceBuffers={:p}, "
                  "numDeviceBuffers={}",
                  static_cast<void*>(handle),
                  static_cast<void*>(executionContext),
                  workspace,
                  static_cast<const void*>(deviceBuffers),
                  numDeviceBuffers);

    return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin::throwIfNull(handle);
        hipdnn_plugin::throwIfNull(executionContext);
        hipdnn_plugin::throwIfNull(deviceBuffers);

            throw hipdnn_plugin ::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "TACO!"); 


        LOG_API_SUCCESS(apiName, "executed graph");
    });
}

} // extern "C"

#include <filesystem>
#include <fstream>
#include <fusilli/backend/backend.h>
#include <fusilli/graph/graph.h>
#include <ios>
#include <iostream>

#include "fusilli.h"

using namespace fusilli;

#define FUSILLI_PRINTERR_LABEL_RED(X) \
    std::cerr << FUSILLI_COLOR_RED << "[FUSILLI] " << X << FUSILLI_COLOR_RESET
#define FUSILLI_PRINTERR_ENDL(X) std::cerr << X << "\n"

#define FUSILLI_UNWRAP(expr)                                                       \
    ({                                                                             \
        auto error_or = (expr);                                                    \
        if(isError(error_or))                                                      \
        {                                                                          \
            FUSILLI_PRINTERR_LABEL_RED("ERROR: ");                                 \
            FUSILLI_PRINTERR_ENDL(#expr << " at " << __FILE__ << ":" << __LINE__); \
            exit(1);                                                               \
        }                                                                          \
        std::move(*error_or);                                                      \
    })

#define FUSILLI_REQUIRE(expr)                                                      \
    do                                                                             \
    {                                                                              \
        auto _errorOr = (expr);                                                    \
        if(isError(_errorOr))                                                      \
        {                                                                          \
            FUSILLI_PRINTERR_LABEL_RED("ERROR: ");                                 \
            FUSILLI_PRINTERR_ENDL(#expr << " at " << __FILE__ << ":" << __LINE__); \
            exit(1);                                                               \
        }                                                                          \
    } while(false)

int main(int argc, char* argv[])
{

    int64_t n = 16;
    int64_t c = 128;
    int64_t h = 64;
    int64_t w = 64;
    int64_t k = 256;
    int64_t r = 1;
    int64_t s = 1;

    
    FusilliHandle handle = FUSILLI_UNWRAP(FusilliHandle::create(Backend::CPU));
    auto graph = std::make_shared<Graph>();

    graph->setName("fprop_sample");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto xTensor = graph->tensor(
        TensorAttr().setName("image").setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1}));

    auto wTensor = graph->tensor(
        TensorAttr().setName("filter").setDim({k, c, r, s}).setStride({c * r * s, r * s, s, 1}));

    auto convAttr
        = ConvFPropAttr().setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1}).setName(
            "conv_fprop");

    auto yTensor = graph->convFProp(xTensor, wTensor, convAttr);

    // Specify Y's dimensions and strides
    yTensor->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
    yTensor->setOutput(true);

    FUSILLI_REQUIRE(graph->validate());


    FUSILLI_REQUIRE(graph->validate());

    FUSILLI_REQUIRE(graph->compile(handle, /*remove=*/true));

    std::cout << "proof of life: hipDNN -> fusilli -> iree connection" << "\n";

    return 0;
}

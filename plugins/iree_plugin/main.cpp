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

    auto graph = std::make_shared<Graph>();
    graph->setBackend(Backend::GFX942);

    graph->setName("fprop_sample");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto x_tensor = graph->tensor(
        TensorAttr().setName("image").setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1}));

    auto w_tensor = graph->tensor(
        TensorAttr().setName("filter").setDim({k, c, r, s}).setStride({c * r * s, r * s, s, 1}));

    auto conv_attr
        = ConvFPropAttr().setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1}).setName(
            "conv_fprop");

    auto y_tensor = graph->convFProp(x_tensor, w_tensor, conv_attr);

    // Specify Y's dimensions and strides
    y_tensor->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
    y_tensor->setOutput(true);

    FUSILLI_REQUIRE(graph->validate());

    std::filesystem::path vmfb = FUSILLI_UNWRAP(graph->readOrGenerateCompiledArtifact(FUSILLI_UNWRAP(graph->emitAsm())));

    // Dump contents of the compiled artifact file
    std::cout << "\nCompiled artifact path: " << vmfb << "\n";
    std::cout << "Contents of compiled artifact:\n";
    std::cout << "==============================\n";
    std::ifstream file(vmfb, std::ios::binary);
    if (file.is_open()) {
        std::cout << file.rdbuf();
        file.close();
    } else {
        std::cerr << "Failed to open file: " << vmfb << "\n";
    }
    std::cout << "\n==============================\n";

    return 0;
}

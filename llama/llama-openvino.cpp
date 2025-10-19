#include "ggml.h"
#include "ggml-openvino.h"

extern "C" {
    void force_link_openvino() {
        struct ggml_backend* b = ggml_backend_openvino_init(0);
        if (b) {
            ggml_backend_free(b);
        }
    }
}

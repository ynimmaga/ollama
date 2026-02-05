patch -p1 < openvino_pre_sync_patch.patch
#git add llama/llama.go
#git add .rsync

make -f Makefile-openvino.sync all

patch -p1 < sampling_patch.patch
patch -p1 < 0018-ggml-Add-batch-size-hint.patch
patch -p1 < 0020-ggml-No-alloc-mode.patch
patch -p1 < 0022-ggml-Enable-resetting-backend-devices.patch
patch -p1 < 0024-GPU-discovery-enhancements.patch
patch -p1 < add_device_patch.patch
patch -p1 < graph_compute_patch.patch

#For ml/backend/ggml/ggml/src/CMakeLists.txt
#comment out the alderlake and amx lines
#comment out mem_hip and mem_nvml

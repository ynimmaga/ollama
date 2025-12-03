# OpenVINO Backend in Ollama

OpenVINO is a high-performance AI inference toolkit to optimize performance on Intel CPUs, Intel integrated and discrete GPUs, and NPUs. This branch contains the OpenVINO backend for Ollama. OpenVINO converts the GGML compute graph to OpenVINO IR and accelerates inference on Intel AI PCs.

# Instructions to build and run OpenVINO Backend

## Prerequisites

- Linux or Windows system with Intel hardware (CPU, GPU, or NPU)
- **For Intel GPU or NPU Usage**: Install the appropriate hardware drivers for your Intel GPU or NPU. For detailed instructions, see: [Additional Configurations for Hardware Acceleration](https://docs.openvino.ai/2025/get-started/install-openvino/configurations.html).
- Git, CMake, and Ninja software tools are needed for building.

## Install OpenVINO Runtime

### 1. Install OpenVINO Runtime

- Follow the guide to install OpenVINO Runtime from an archive file: [Linux](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-linux.html) | [Windows](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-windows.html)

<details>
<summary>📦 Click to expand OpenVINO 2025.3 installation from an archive file on Ubuntu</summary>
<br>

```bash
wget https://raw.githubusercontent.com/ravi9/misc-scripts/main/openvino/ov-archive-install/install-openvino-from-archive.sh
chmod +x install-openvino-from-archive.sh
./install-openvino-from-archive.sh
```
</details>

- Verify OpenVINO is initialized properly
```bash
echo $OpenVINO_DIR
```

## Build Ollama with OpenVINO Backend on Linux

### Clone Ollama

Clone the OpenVINO-enabled Ollama fork:

```bash
git clone https://github.com/ynimmaga/ollama.git
cd ollama
git checkout dev_backend_openvino
```

### Vendor the required OpenVINO patches

```
git apply openvino_pre_sync_patch.patch
make -f Makefile-openvino.sync all
git apply openvino_post_sync_patch.patch
```

### Build GGML OpenVINO Backend and Add to the Library path

```bash
mkdir build && cd build
cmake .. -DGGML_OPENVINO=ON -DBUILD_SHARED_LIBS=ON
make -j8

```
### Build Ollama

```bash
cd $ollama_root
export LD_LIBRARY_PATH=$PWD/build/lib/ollama:$LD_LIBRARY_PATH
export CGO_CXXFLAGS="-DGGML_USE_OPENVINO -I$INTEL_OPENVINO_DIR/runtime/include"
export CGO_LDFLAGS="-L${INTEL_OPENVINO_DIR}/runtime/lib/intel64 -lopenvino -L${PWD}/build/lib/ollama -lggml-openvino -lstdc++"
go clean -cache
go mod tidy
go build .
```

### Download models for testing:

```bash
# Download model file: Llama-3.2-1B-Instruct.fp16.gguf
wget https://huggingface.co/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct.fp16.gguf \
     -O Llama-3.2-1B-Instruct.fp16.gguf
```

### Create Modelfile and add the below text:
```bash
FROM ./Llama-3.2-1B-Instruct.fp16.gguf
```
### Start Ollama server and run inference

```bash
cd $ollama_root
OLLAMA_FLASH_ATTENTION=1 ./ollama serve
```

Open another terminal, create, and run Ollama model
```bash
./ollama create llama3.2-1b-f16 -f Modelfile
OLLAMA_FLASH_ATTENTION=1 ./ollama run llama3.2-1b-f16
```

## Build Ollama with OpenVINO Backend on Windows

Once OpenVINO is downloaded on windows, follow the below instructions. 
(Note: The instructions need to be streamlined and cleaned up further)

We need to switch between msys2 and command prompt terminals for the build steps.

### Install MSYS2 terminal and set up

Download and install from [msys2.org](https://msys2.org)
Open the "MSYS2 UCRT64" terminal and install dependencies
```
pacman -S mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-make base-devel rsync git mingw-w64-ucrt-x86_64-go mingw-w64-ucrt-x86_64-cmake
```

Set GOROOT for MSYS2's native Go and verify the installation by checking the version
```
export GOROOT=/ucrt64/lib/go
export PATH=$GOROOT/bin:$PATH
go version
```

### Clone Ollama

Clone the OpenVINO-enabled Ollama fork from MSYS2 terminal:

```bash
git clone https://github.com/ynimmaga/ollama.git
cd ollama
git checkout dev_backend_openvino
```

### Vendor the required OpenVINO patches

```
git apply openvino_pre_sync_patch.patch
make -f Makefile-openvino.sync all
git apply openvino_post_sync_patch.patch
```

### Build GGML OpenVINO Backend and Add to the Library path

For building GGML OpenVINO backend, open a command prompt and first source the openvino variables using `setupvars.bat` and then do the following from the `Ollama` directory:
```bash
mkdir build && cd build
cmake .. -DGGML_OPENVINO=ON -DBUILD_SHARED_LIBS=OFF 
cmake --build . --target INSTALL --config Release
```

### Build Ollama

```bash
export INTEL_OPENVINO_DIR=<path to OpenVINO dir that has setupvars.bat>
copy build\ml\backend\ggml\ggml\src\ggml-openvino\Release\ggml-openvino.lib .
copy %INTEL_OPENVINO_DIR%\runtime\lib\intel64\Release\openvino.lib .

set CGO_CXXFLAGS="-DGGML_USE_OPENVINO -I$INTEL_OPENVINO_DIR/runtime/include"
export CGO_LDFLAGS="-L./ -lopenvino -lggml-openvino"

go clean -cache
go mod tidy
go build .
```

### Download models for testing:

```bash
# Download model file: Llama-3.2-1B-Instruct.fp16.gguf
wget https://huggingface.co/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct.fp16.gguf \
     -O Llama-3.2-1B-Instruct.fp16.gguf
```

### Create Modelfile and add the below text:
```bash
FROM ./Llama-3.2-1B-Instruct.fp16.gguf
```
### Start Ollama server and run inference

```bash
cd $ollama_root
OLLAMA_FLASH_ATTENTION=1 ./ollama.exe serve
```

Open another terminal, create, and run Ollama model
```bash
./ollama.exe create llama3.2-1b-f16 -f Modelfile
OLLAMA_FLASH_ATTENTION=1 ./ollama.exe run llama3.2-1b-f16
```

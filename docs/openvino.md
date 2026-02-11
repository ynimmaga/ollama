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

### Download models for testing

```bash
# Download model file: Llama-3.2-1B-Instruct.fp16.gguf
wget https://huggingface.co/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct.fp16.gguf \
     -O Llama-3.2-1B-Instruct.fp16.gguf
```

###  Create a file named 'Modelfile' and add the below line to the file:
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

### Prerequisites

- Download Microsoft.VisualStudio.2022.BuildTools: [Visual_Studio_Build_Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe). Select "Desktop development with C++" under workloads
- Install git
- Follow the guide to install OpenVINO Runtime from an archive file: [Windows](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-windows.html)
- **OpenCL:**
     - Install OpenCL using [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
     - Download cl2.hpp and opencl.hpp from [OpenCL-CLHPP](https://github.com/KhronosGroup/OpenCL-CLHPP/tree/main/include/CL) and paste at following path: `C:\~\oneAPI\compiler\latest\include\CL\`
- Build requires use of both MSYS2 and x64 Native Tools Command Prompt terminals. Download and install [msys2.org](https://msys2.org).


### 1. Open "MSYS2 UCRT64" terminal to install dependencies

- Use the below command to install dependencies
     ```bash
     pacman -S mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-make base-devel rsync git mingw-w64-ucrt-x86_64-go mingw-w64-ucrt-x86_64-cmake
     ```

- Set GOROOT for MSYS2's native Go and verify the installation by checking the version
     ```bash
     export GOROOT=/ucrt64/lib/go
     export PATH=$GOROOT/bin:$PATH
     go version
     ```


### 2. Clone Ollama

Clone the OpenVINO-enabled Ollama fork from "MSYS2 UCRT64" terminal:
```bash
git clone https://github.com/ynimmaga/ollama.git
cd ollama
git checkout ov_backend
```


### 3. Vendor the required OpenVINO patches

- Login to git with username and email
     ```bash
     git config --global user.name "Your Name"
     git config --global user.email “you@example.com”
     ```
- verify
     ```bash
     git config --global --list
     ```
- Apply patch now
     ```bash
     ./run_ov.sh
     ```
- Edit manually:
     ```powershell
     Find ~/ml/backend/ggml/ggml/src/CMakeLists.txt
     #comment out mem_hip.cpp and mem_nvml.cpp
     ```


### 4. Build GGML OpenVINO backend and Add to the Library path

For building GGML OpenVINO backend, 

- Open x64 Native Tools Command Prompt at `ollama` directory, source OpenVINO and OpenCL variables using
     ```bash
     "c:\Program Files (x86)\Intel\<openvino_toolkit_windows_folder>\setupvars.bat"
     "c:\Program Files (x86)\Intel\oneAPI\setvars.bat"
     ```
- To build OpenVINO backend
     ```bash
     mkdir build && cd build
     cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_OPENVINO=ON -DGGML_VULKAN=OFF -DVulkan_FOUND=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=ON -DBUILD_SHARED_LIBS=OFF
     cmake --build . --target INSTALL --config Release
     copy "ml\backend\ggml\ggml\src\ggml-openvino\Release\ggml-openvino.lib" .
     copy "%INTEL_OPENVINO_DIR%\runtime\lib\intel64\Release\openvino.lib" .
     ```


### 5. Build Ollama

- Now go back to the "MSYS2 UCRT64" terminal and execute the following commands:
     ```bash
     export INTEL_OPENVINO_DIR=<path to OpenVINO dir that has setupvars.bat>
     # but substitute “PROGRA~2” for “Program Files (x86)” in the path
     export CGO_CXXFLAGS="-DGGML_USE_OPENVINO -I$INTEL_OPENVINO_DIR/runtime/include"
     export CGO_LDFLAGS="-L./ -lopenvino -lggml-openvino"
     export CC=cl
     export CXX=cl
     export CGO_ENABLED=1
     go clean -cache -modcache
     
     export PATH="/C/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:$PATH"
     pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-go
     ```
- Now open “MSYS MINGW64” and build ollama.exe
     ```bash
     export PATH="/mingw64/bin:$PATH"
     go clean -cache
     cd /home/Administrator/ollama
     go mod tidy
     go build .
     ```


### 6. To run models

- Create a file named 'Modelfile.txt' in `ollama` folder and add below line to the file
     ```powershell
     FROM <model_directory>\MODEL.gguf
     ```

- Open a command prompt and start a ollama server instance
     ```powershell
     set OLLAMA_FLASH_ATTENTION=1
     ollama.exe serve
     ```

- Open another terminal, create, and run Ollama model
     ```powershell
     ollama.exe create TEST -f Modelfile.txt
     set OLLAMA_FLASH_ATTENTION=1
     ollama.exe run TEST
     ```

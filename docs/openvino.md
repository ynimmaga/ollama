# OpenVINO Backend in Ollama

OpenVINO is a high-performance AI inference toolkit to optimize performance on Intel CPUs, Intel integrated and discrete GPUs, and NPUs. This branch contains the OpenVINO backend for Ollama. OpenVINO converts the GGML compute graph to OpenVINO IR and accelerates inference on Intel AI PCs.

# Instructions to build and run OpenVINO Backend

## Prerequisites

- Linux or Windows system with Intel hardware (CPU, GPU, or NPU)
- **For Intel GPU or NPU Usage**: Install the appropriate hardware drivers for your Intel GPU or NPU. For detailed instructions, see: [Additional Configurations for Hardware Acceleration](https://docs.openvino.ai/2025/get-started/install-openvino/configurations.html).
- Git, CMake, and Ninja software tools are needed for building.
```bash
  sudo apt-get update
  sudo apt-get install -y build-essential libcurl4-openssl-dev libtbb12 cmake ninja-build python3-pip curl wget tar
```

## Install OpenVINO Runtime

- Follow the guide to install OpenVINO Runtime from an archive file: [Linux](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-linux.html) | [Windows](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-windows.html)

<details>
<summary>📦 Click to expand OpenVINO 2025.2 installation commands on Linux</summary>
<br>

```bash
export OPENVINO_VERSION_MAJOR=2025.2
export OPENVINO_VERSION_FULL=2025.2.0.19140.c01cd93e24d
sudo apt-get update
sudo apt-get install -y build-essential libcurl4-openssl-dev libtbb12 cmake ninja-build python3-pip curl wget tar
sudo mkdir -p /opt/intel
wget -O openvino_${OPENVINO_VERSION_MAJOR}.tgz https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/linux/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz
tar -xf openvino_${OPENVINO_VERSION_MAJOR}.tgz
sudo mv openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64 /opt/intel/openvino_${OPENVINO_VERSION_MAJOR}
rm openvino_${OPENVINO_VERSION_MAJOR}.tgz
cd /opt/intel/openvino_${OPENVINO_VERSION_MAJOR}
echo "Y" | sudo -E ./install_dependencies/install_openvino_dependencies.sh && cd -
sudo ln -s /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} /opt/intel/openvino
source /opt/intel/openvino/setupvars.sh
```
</details>

- Verify OpenVINO is initialized properly
```bash
echo $OpenVINO_DIR
```

## Build Ollama with OpenVINO Backend

### Clone Ollama

Clone the OpenVINO-enabled Ollama fork:

```bash
git clone https://github.com/ynimmaga/ollama.git
cd ollama
git switch poc_openvino_backend
```

### Build GGML OpenVINO Backend and Add to the Library path

```bash
 mkdir build && cd build
cmake .. -DGGML_OPENVINO=ON -DBUILD_SHARED_LIBS=ON
make -j8
export LD_LIBRARY_PATH=$PWD/lib/ollama:$LD_LIBRARY_PATH
```
### Build Ollama

```bash
cd $ollama_root
go clean -cache
go mod tidy
go build .
```

## Download models for testing:

```bash
# Download model file: Llama-3.2-1B-Instruct.fp16.gguf
wget https://huggingface.co/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct.fp16.gguf \
     -O Llama-3.2-1B-Instruct.fp16.gguf
```

## Create Modelfile and add the below text:
```bash
# 1. Reference the local model file
FROM ./Llama-3.2-1B-Instruct.fp16.gguf

# 2. Set the default max tokens to generate (from your -n 50 argument)
PARAMETER num_predict 50

# 3. Define the official prompt template for Llama 3.2 Instruct
TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# 4. Set a default system message
SYSTEM """You are a helpful AI assistant."""

# 5. Define the model's special stop tokens
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"

# 6. Set the context window size
PARAMETER num_ctx 4096
```
## Start Ollama server and run inference

```bash
cd $ollama_root
./ollama serve
```

Open another terminal, create, and run Ollama model
```bash
./ollama create llama3.2-1b-f16 -f Modelfile
./ollama run llama3.2-1b-f16
```

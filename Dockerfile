# 1. Base Image
# Using NVIDIA CUDA 12.1.1 with cuDNN 8, Ubuntu 22.04 (devel image for build tools)
# This aligns with the mention of CUDA 12.1 in your README.
# If your PyTorch version is strictly for CUDA 11.8, you might need to change this to a CUDA 11.8 base.
# Refer to https://hub.docker.com/r/nvidia/cuda/ for available tags.
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# 2. Set Environment Variables
# Non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Python configuration (targeting Python 3.10 as per README)
ENV PYTHON_VERSION=3.10
ENV VENV_PATH=/opt/venv

# Application directory and PYTHONPATH
# Adding /app to PYTHONPATH ensures modules in the project root (like qwen_vl_utils if it's a top-level dir) are found.
ENV APP_DIR=/workspace
ENV PYTHONPATH=""
ENV PYTHONPATH="${APP_DIR}:${PYTHONPATH}"

# Triton cache directory (as per your working setup)
ENV TRITON_CACHE_DIR=/root/.triton/autotune

# Add virtual environment's bin to PATH
ENV PATH="${VENV_PATH}/bin:${PATH}"

# 3. Install System Dependencies & Python
# Includes build-essential and other dependencies from your working setup.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git-lfs \
    build-essential \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    python${PYTHON_VERSION}-dev \
    libaio-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libmpich-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Create Python Virtual Environment
RUN python${PYTHON_VERSION} -m venv ${VENV_PATH}

# 5. Upgrade Pip and Install Python Packages from requirements.txt
# First, copy only requirements.txt to leverage Docker layer caching.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    # Install packages from requirements.txt using the venv's pip
    # --no-cache-dir helps reduce image size
    pip install --no-cache-dir -r /tmp/requirements.txt

# 6. Set Up Application Directory and Triton Cache
WORKDIR ${APP_DIR}
# Create Triton cache directory as per your working setup
RUN mkdir -p ${TRITON_CACHE_DIR} && \
    chmod -R 700 ${TRITON_CACHE_DIR}

# 7. Copy Project Code
# This copies all files from your project's build context into the image at /workspace.
COPY . ${APP_DIR}

# 8. (Optional) Expose Ports or Set Default Command
# If your application runs a web service, you might expose a port:
# EXPOSE 8000

# For a development/research environment, a CMD like bash is often useful
# to allow interactive sessions. Otherwise, you might set it to run your main.py.
# CMD ["python", "main.py", "--help"]
CMD ["bash"]
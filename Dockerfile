FROM hub.oepkgs.net/openeuler/openeuler:22.03-lts-sp4

####################### os #######################
RUN yum clean all && \
    yum makecache && \
    yum install -y \
        kmod \
        sudo \
        wget \
        curl \
        cmake \
        make \
        git \
        vim \
        gcc && \
    yum clean all

####################### python #######################
WORKDIR /root
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_25.1.1-2-Linux-aarch64.sh && \
    bash /root/Miniconda3-py311_25.1.1-2-Linux-aarch64.sh -b && \
    rm /root/Miniconda3-py311_25.1.1-2-Linux-aarch64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
ENV PYTHONPATH="/root/miniconda3/lib/python3.11/site-packages"
RUN pip config set global.index-url 'https://pypi.tuna.tsinghua.edu.cn/simple' && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

####################### CANN #######################
WORKDIR /root
RUN echo "UserName=HwHiAiUser" >> /etc/ascend_install.info && \
    echo "UserGroup=HwHiAiUser" >> /etc/ascend_install.info && \
    echo "Firmware_Install_Type=full" >> /etc/ascend_install.info && \
    echo "Firmware_Install_Path_Param=/usr/local/Ascend" >> /etc/ascend_install.info && \
    echo "Driver_Install_Type=full" >> /etc/ascend_install.info && \
    echo "Driver_Install_Path_Param=/usr/local/Ascend" >> /etc/ascend_install.info && \
    echo "Driver_Install_For_All=no" >> /etc/ascend_install.info && \
    echo "Driver_Install_Mode=normal" >> /etc/ascend_install.info && \
    echo "Driver_Install_Status=complete" >> /etc/ascend_install.info
RUN curl -s "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-toolkit_8.0.0_linux-aarch64.run" -o Ascend-cann-toolkit.run && \
    curl -s "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-kernels-910b_8.0.0_linux-aarch64.run" -o Ascend-cann-kernels-910b.run && \
    curl -s "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-nnrt_8.0.0_linux-aarch64.run" -o Ascend-cann-nnrt.run && \
    chmod a+x *.run && \
    bash /root/Ascend-cann-toolkit.run --install -q && \
    bash /root/Ascend-cann-kernels-910b.run --install -q && \
    bash /root/Ascend-cann-nnrt.run --install -q && \
    rm /root/*.run
RUN echo "source /usr/local/Ascend/nnrt/set_env.sh" >> /root/.bashrc && \
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc

####################### dev env #######################
RUN pip install --no-cache-dir \
    cmake>=3.26 \
    decorator \
    ray==2.42.1 \
    protobuf==3.20.0 \
    ml_dtypes \
    wheel \
    setuptools \
    wrap \
    deprecated \
    packaging \
    ninja \
    "setuptools-scm>=8" \
    numpy \
    build

WORKDIR /workspace

RUN git clone -b br_infer_deepseek_os https://gitee.com/mindspore/mindformers.git /workspace/mindformers && \
    cd mindformers && \
    sed -i 's/-i https:\/\/pypi.tuna.tsinghua.edu.cn\/simple//g' build.sh && \
    bash build.sh && \
    PACKAGE_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cp -a research "$PACKAGE_PATH" && \
    rm -rf /workspace/mindformers

RUN git clone https://gitee.com/mindspore/golden-stick.git /workspace/golden-stick && \
    cd golden-stick && \
    bash build.sh && \
    pip install --no-cache-dir /workspace/golden-stick/output/*.whl && \
    rm -rf /workspace/golden-stick

ENV USE_TORCH="FALSE"
ENV USE_TF="FALSE"
RUN git clone -b v0.6.6.post1 https://gitee.com/mirrors/vllm.git /workspace/vllm && \
    cd vllm && \
    VLLM_TARGET_DEVICE=empty pip install --no-cache-dir . && \
    rm -rf /workspace/vllm

RUN git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git /workspace/msadapter && \
    cd /workspace/msadapter && \
    bash scripts/build_and_reinstall.sh && \
    rm -rf /workspace/msadapter

ADD . /workspace/vllm_mindspore
RUN cd /workspace/vllm_mindspore && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install . && \
    rm -rf /workspace/vllm_mindspore

RUN wget -O mindspore-2.5.0-cp311-cp311-linux_aarch64.whl \
https://repo.mindspore.cn/mindspore/mindspore/version/202503/20250303/br_infer_deepseek_os_20250303004707_705727d59236c8c197b25ad0e464c4908434d42f_newest/unified/aarch64/mindspore-2.5.0-cp311-cp311-linux_aarch64.whl && \
pip install --no-cache-dir mindspore-2.5.0-cp311-cp311-linux_aarch64.whl && \
rm -f mindspore-2.5.0-cp311-cp311-linux_aarch64.whl

RUN pip uninstall torch torch-npu torchvision -y

CMD ["bash"]
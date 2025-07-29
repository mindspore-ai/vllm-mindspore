# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM hub.oepkgs.net/openeuler/openeuler:22.03-lts-sp4

RUN set -ex && \
    echo "[OS]" > /etc/yum.repos.d/openEuler.repo && \
    echo "name=OS" >> /etc/yum.repos.d/openEuler.repo && \
    echo "baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/OS/\$basearch/" >> /etc/yum.repos.d/openEuler.repo && \
    echo "enabled=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgcheck=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgkey=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/OS/\$basearch/RPM-GPG-KEY-openEuler" >> /etc/yum.repos.d/openEuler.repo && \
    echo "" >> /etc/yum.repos.d/openEuler.repo && \
    echo "[everything]" >> /etc/yum.repos.d/openEuler.repo && \
    echo "name=everything" >> /etc/yum.repos.d/openEuler.repo && \
    echo "baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/everything/\$basearch/" >> /etc/yum.repos.d/openEuler.repo && \
    echo "enabled=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgcheck=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgkey=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/everything/\$basearch/RPM-GPG-KEY-openEuler" >> /etc/yum.repos.d/openEuler.repo && \
    echo "" >> /etc/yum.repos.d/openEuler.repo && \
    echo "[update]" >> /etc/yum.repos.d/openEuler.repo && \
    echo "name=update" >> /etc/yum.repos.d/openEuler.repo && \
    echo "baseurl=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/update/\$basearch/" >> /etc/yum.repos.d/openEuler.repo && \
    echo "enabled=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgcheck=1" >> /etc/yum.repos.d/openEuler.repo && \
    echo "gpgkey=https://mirrors.huaweicloud.com/openeuler/openEuler-22.03-LTS-SP4/update/\$basearch/RPM-GPG-KEY-openEuler" >> /etc/yum.repos.d/openEuler.repo

RUN set -ex && \
    yum makecache && \
    yum install -y \
        bzip2 \
        bzip2-devel \
        cmake \
        curl \
        dpkg-devel \
        gcc \
        gdbm-devel \
        git \
        gperftools \
        gperftools-devel \
        kmod \
        libdb-devel \
        libffi-devel \
        llvm-toolset-17 \
        make \
        openssl-devel \
        pciutils \
        readline-devel \
        sqlite \
        sqlite-devel \
        sudo \
        tree \
        vim \
        wget \
        xz \
        xz-devel \
        zlib-devel \
        gzip \
        zip \
    && yum clean all \
    && rm -rf /var/cache/yum

WORKDIR /root

RUN set -ex && \
    wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_25.1.1-2-Linux-aarch64.sh && \
    bash /root/Miniconda3-py311_25.1.1-2-Linux-aarch64.sh -b && \
    rm /root/Miniconda3-py311_25.1.1-2-Linux-aarch64.sh

ENV PATH="/root/miniconda3/bin:$PATH"
ENV PYTHONPATH="/root/miniconda3/lib/python3.11/site-packages"

RUN set -ex && \
    pip config set global.index-url 'https://pypi.tuna.tsinghua.edu.cn/simple' && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

RUN set -ex && \
    echo "UserName=HwHiAiUser" >> /etc/ascend_install.info && \
    echo "UserGroup=HwHiAiUser" >> /etc/ascend_install.info && \
    echo "Firmware_Install_Type=full" >> /etc/ascend_install.info && \
    echo "Firmware_Install_Path_Param=/usr/local/Ascend" >> /etc/ascend_install.info && \
    echo "Driver_Install_Type=full" >> /etc/ascend_install.info && \
    echo "Driver_Install_Path_Param=/usr/local/Ascend" >> /etc/ascend_install.info && \
    echo "Driver_Install_For_All=no" >> /etc/ascend_install.info && \
    echo "Driver_Install_Mode=normal" >> /etc/ascend_install.info && \
    echo "Driver_Install_Status=complete" >> /etc/ascend_install.info

RUN set -ex && \
    curl -s -k "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run" -o Ascend-cann-toolkit.run && \
    curl -s -k "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-kernels-910b_8.1.RC1_linux-aarch64.run" -o Ascend-cann-kernels-910b.run && \
    curl -s -k "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnrt_8.1.RC1_linux-aarch64.run" -o Ascend-cann-nnrt.run && \
    chmod a+x *.run && \
    bash /root/Ascend-cann-toolkit.run --install -q && \
    bash /root/Ascend-cann-kernels-910b.run --install -q && \
    bash Ascend-cann-nnrt.run --install -q && \
    rm /root/*.run

RUN set -ex && \
    echo "source /usr/local/Ascend/nnrt/set_env.sh" >> /root/.bashrc && \
    echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc

RUN set -ex && \
    pip install --no-cache-dir \
        "cmake>=3.26" \
        decorator \
        ray==2.43.0 \
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
        numba \
        build

CMD ["bash"]


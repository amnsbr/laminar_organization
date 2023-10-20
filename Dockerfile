FROM --platform=linux/amd64 jupyter/base-notebook:ubuntu-20.04

USER root

# install Python, Connectome Workbench and Singularity dependencies
RUN apt-get update && \
    apt clean && \
    apt-get -y install git \
                   build-essential \
                   squashfs-tools \
                   uuid-dev \
                   libssl-dev \
                   libgpgme11-dev \
                   libseccomp-dev \
                   pkg-config \
                   python3-dev \
                   python3-pip \
                   connectome-workbench \
                   xvfb

# Install Singularity
RUN wget https://github.com/apptainer/singularity/releases/download/v3.8.7/singularity-container_3.8.7_amd64.deb && \
    dpkg -i singularity-container_3.8.7_amd64.deb && \
    rm singularity-container_3.8.7_amd64.deb

# build bigbrainwarp singularity image
# (the default path to bigbrainwarp is hardcoded in the scripts
#  and I am reproducing it in the docker image)
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    mkdir -p /data/group/cng/Tools && \
    singularity build \
        /data/group/cng/Tools/bigbrainwarp.simg \
        docker://caseypaquola/bigbrainwarp@sha256:cb62b57b95a7ea3e64a4e2b123cd496dba568e7aa7d33cb36019d7ded71d0404 && \
    singularity cache clean -f

# Copy everything (except .dockerignore [which includes .git folder])
COPY . /laminar_organization
WORKDIR /laminar_organization

# create conda environment (to specify Python version) and install packages with pip (faster than conda)
ENV PYTHONDONTWRITEBYTECODE=1
RUN conda create -y -p /laminar_organization/env python=3.9.12 && \
    conda install pip -y && \
    /laminar_organization/env/bin/pip install -r /laminar_organization/code/requirements.txt --no-cache-dir

# Open jupyter notebook
ENTRYPOINT ["bash", "-c", "/laminar_organization/env/bin/jupyter notebook --allow-root"]
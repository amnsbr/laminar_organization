FROM --platform=linux/amd64 jupyter/base-notebook:ubuntu-20.04

USER root

# install singularity
RUN apt-get update && \
    apt clean && \
    apt-get -y install git \
                   build-essential \
                   libtool \
                   squashfs-tools \
                   autotools-dev \
                   libarchive-dev \
                   automake \
                   autoconf \
                   debootstrap \
                   uuid-dev \
                   libssl-dev \
                   python3-dev \
                   python3-pip \
                   connectome-workbench

# Install Singularity from Github
WORKDIR /tmp
RUN pip3 install sregistry[all]
RUN git clone -b vault/release-2.6 https://github.com/sylabs/singularity.git && \
    cd /tmp/singularity && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make && make install


# Copy everything
COPY . /laminar_organization

WORKDIR /laminar_organization

# build bigbrainwarp singularity image
# (the default path to bigbrainwarp is hardcoded in the scripts
#  and I am reproducing it in the docker image)
# TODO: the proper way to do it would be through enviornment variables
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    mkdir -p /data/group/cng/Tools && \
    singularity build \
        /data/group/cng/Tools/bigbrainwarp.simg \
        docker://caseypaquola/bigbrainwarp@sha256:cb62b57b95a7ea3e64a4e2b123cd496dba568e7aa7d33cb36019d7ded71d0404
# May need to run these:
# sudo chown root:root /usr/local/libexec/singularity/bin/action-suid
# sudo chmod 4755 /usr/local/libexec/singularity/bin/action-suid

# install most Python dependencies using pip (faster than conda)
RUN conda create -y -p /laminar_organization/env python=3.9 && \
    conda install pip -y && \
    /laminar_organization/env/bin/pip install -r /laminar_organization/code/requirements.txt

# install osmesa build of vtk using conda (osmesa build is compatible with remote desktops)
RUN eval "$(conda shell.bash hook)" && \
    conda activate /laminar_organization/env && \ 
    conda install mesalib==21.2.5 vtk==9.1.0=osmesa_py39hd96a68f_115 -y --channel conda-forge --override-channels --freeze-installed && \
    /laminar_organization/env/bin/pip install brainspace==0.1.3

# install enigmatoolbox from github
RUN mkdir /laminar_organization/tools && \
    cd /laminar_organization/tools && \
    git clone --depth 1 --branch 1.1.3 https://github.com/MICA-MNI/ENIGMA.git && \
    cd ENIGMA && /laminar_organization/env/bin/pip install .

# reinstall numpy using conda
# when installed using pip, numpy (1.23.4) is incompatible
# with pycortex (1.2.3) and importing pycortex throws an error. 
RUN eval "$(conda shell.bash hook)" && \
    conda activate /laminar_organization/env && \
    conda install numpy==1.23.4 -y --channel conda-forge


# # # Open jupyter notebook
ENTRYPOINT ["bash", "-c", "/laminar_organization/env/bin/jupyter notebook --allow-root"]

# # docker build --platform linux/amd64 -t laminar_organization .  
# # docker run -it --platform linux/amd64 -p 8888:8888 laminar_organization
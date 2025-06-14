FROM tensorflow/tensorflow:nightly-custom-op-ubuntu16

ENV GITHUB_BRANCH="r1.0-tf1.15"
ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION=""
ENV PIP_MANYLINUX2010="1"

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-0.28.0-installer-linux-x86_64.sh > /dev/null
RUN bash bazel-0.28.0-installer-linux-x86_64.sh

RUN apt-get install -y python3.5
RUN apt-get install -y python3.6

RUN pip install --upgrade setuptools
RUN pip3 install --upgrade setuptools

# Install tensorflow
RUN pip uninstall -y tensorflow
RUN pip3 uninstall -y tensorflow
RUN pip install tensorflow==2.0.0
RUN pip3 install tensorflow==2.0.0

RUN pip3 install --upgrade auditwheel
COPY pip_pkg_scripts/build.sh /

VOLUME /tmp/pip_pkg_build

ENTRYPOINT ["/build.sh"]

# The default parameters for the build.sh
CMD []

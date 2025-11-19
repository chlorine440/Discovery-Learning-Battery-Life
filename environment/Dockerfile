# hash:sha256:2996d2425f29f4a5af7ccf8c857073a07cff86a717efb75692be68e6d4f5c44c
FROM registry.codeocean.com/codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        python==3.9.16 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    matplotlib==3.7.0 \
    numpy==1.23.5 \
    openpyxl==3.0.10 \
    optuna==3.1.0 \
    pandas==2.3.1 \
    scikit-learn==1.3.2

ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reditorsupport.r \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev

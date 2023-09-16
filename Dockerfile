FROM python:3.11.5-bullseye

# 安装依赖
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y libmagic-dev poppler-utils tesseract-ocr libreoffice pandoc
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# 工作目录，这个目录对应于镜像内的工作目录，后面的所有涉及到路径的操作都可以
# 使用WORKDIR的相对路径来指定
WORKDIR /app

# 拷贝requirements.txt 到 镜像中/app/requirements.txt  
COPY requirements.txt requirements.txt

# 升级pip
RUN pip install --upgrade pip

# 安装pip包
RUN pip install -r requirements.txt --upgrade

# 将当前文件中的目录复制到/app目录下
COPY . .

# 运行脚本
CMD ["streamlit", "run", "AI.py"]

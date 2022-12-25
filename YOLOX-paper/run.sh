# INSTALL YOLOX
if [ ! -e YOLOX ]; then
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git
    cd YOLOX
    pip3 install -v -e .
    cd ..
fi

# PREPARE COCO
# download dataset
if [ ! -e val2017.zip ]; then
    wget http://images.cocodataset.org/zips/val2017.zip
fi
if [ ! -e COCO/val2017 ]; then
    unzip val2017.zip
    mkdir -p COCO/val2017
    mv val2017/* COCO/val2017
    rm -r val2017
fi
if [ ! -e annotations_trainval2017.zip ]; then
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi
if [ ! -e COCO/annotations ]; then
    unzip annotations_trainval2017.zip
    mkdir -p COCO/annotations
    mv annotations/* COCO/annotations
    rm -r annotations
fi

# link dataset to base dir of YOLOX
ln -s COCO YOLOX/datasets/COCO

mdl=models/yolox_s.pth
if [ ! -e $mdl ]; then
    mkdir -p models
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
    mv yolox_s.pth models
fi
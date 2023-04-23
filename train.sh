export NGPUS=4
export PYTHONROOT=$HOME/.opt/miniconda/envs/cra/bin
export CROOT=$HOME/codes/CRA
export CFGROOT=$CROOT/configs
export SROOT=$CROOT/experiments

$PYTHONROOT/python test.py -cfg $CFGROOT/deeplabv2_r101_cda.yaml --saveres \
    resume pretrained/cda.pth \
    OUTPUT_DIR $SROOT/soft_labels/cda \
    DATASETS.TEST cityscapes_train

$PYTHONROOT/python -m torch.distributed.launch --nproc_per_node=$NGPUS train_cra.py \
    -cfg $CFGROOT/deeplabv2_r101_cra.yaml \
    DATASETS.FLAG cra \
    DATASETS.SOURCE_LABEL_DIR $SROOT/soft_labels/cda/inference/cityscapes_train \
    OUTPUT_DIR $SROOT/cra \
    resume pretrained/cda.pth 

$PYTHONROOT/python test.py -cfg $CFGROOT/deeplabv2_r101_cra.yaml resume $SROOT/cra/model_iter040000.pth >> $SROOT/cra/eval.txt

$PYTHONROOT/python test.py -cfg $CFGROOT/deeplabv2_r101_cra.yaml --saveres \
    resume $SROOT/cra/model_iter040000.pth \
    OUTPUT_DIR $SROOT/soft_labels/cra \
    DATASETS.TEST cityscapes_train

$PYTHONROOT/python -m torch.distributed.launch --nproc_per_node=$NGPUS train_self_distill.py -cfg $CFGROOT/deeplabv2_r101_self_distill.yaml \
    OUTPUT_DIR $SROOT/sd \
    DATASETS.FLAG sd \
    DATASETS.SOURCE_LABEL_DIR $SROOT/$OPATH/soft_labels/cra/inference/cityscapes_train

$PYTHONROOT/python test.py -cfg $CFGROOT/deeplabv2_r101_self_distill.yaml resume $SROOT/sd/model_iter020000.pth >> $SROOT/sd/eval.txt

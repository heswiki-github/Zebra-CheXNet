# Zebra-CheXNet

Reference: https://github.com/arnoweng/CheXNet

# CheXNet for Classification and Localization of Thoracic Diseases

This is a Python3 (Pytorch) reimplementation of [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/). The model takes a chest X-ray image as input and outputs the probability of each thoracic disease (胸部疾病) along with a likelihood map of pathologies.


## Dataset

The [ChestX-ray14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) comprises 112,120 frontal-view chest X-ray images of 30,805 unique patients with 14 disease labels. To evaluate the model, we randomly split the dataset into training (70%), validation (10%) and test (20%) sets, following the work in paper. Partitioned image names and corresponding labels are placed under the directory [labels](./ChestX-ray14/labels).

## Prerequisites

- Python 3.4+
- [PyTorch](http://pytorch.org/) and its dependencies

## Usage

1. Clone this repository.

2. Download images of ChestX-ray14 from this [released page](https://nihcc.app.box.com/v/ChestXray-NIHCC) and decompress them to the directory [images](./ChestX-ray14/images).

3. Specify FPGA or GPUs for hardware acceleration and run

   `python model.py`

## Comparsion

We followed the training strategy described in the official paper, and a ten crop method is adopted both in validation and test. Compared with the original CheXNet, the per-class AUROC of our reproduced model is almost the same. We have also proposed a slightly-improved model which achieves a mean AUROC of 0.847 (v.s. 0.841 of the original CheXNet).

| Pathology (病理) | [Wang et al.](https://arxiv.org/abs/1705.02315) | [Yao et al.](https://arxiv.org/abs/1710.10501) | [CheXNet](https://arxiv.org/abs/1711.05225) | [Implemented Model](https://github.com/arnoweng/CheXNet) | [Improved Model](https://github.com/arnoweng/CheXNet) | Zebra Implementaiton |
| :----------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :---------------------: | :----------------: | :----------------: |
| Atelectasis (肺不張)          | 0.716 | 0.772 | 0.8094 | 0.8294 | 0.8311 | 0.6138 |
| Cardiomegaly (心臟腫大)       | 0.807 | 0.904 | 0.9248 | 0.9165 | 0.9220 | 0.5337 |
| Effusion (積液)               | 0.784 | 0.859 | 0.8638 | 0.8870 | 0.8891 | 0.8328 |
| Infiltration (浸潤)           | 0.609 | 0.695 | 0.7345 | 0.7143 | 0.7146 | 0.3729 |
| Mass (大量的)                 | 0.706 | 0.792 | 0.8676 | 0.8597 | 0.8627 | 0.3908 |
| Nodule (結核)                 | 0.671 | 0.717 | 0.7802 | 0.7873 | 0.7883 | 0.6040 |
| Pneumonia (肺炎)              | 0.633 | 0.713 | 0.7680 | 0.7745 | 0.7820 | 0.5178 |
| Pneumothorax (氣胸)           | 0.806 | 0.841 | 0.8887 | 0.8726 | 0.8844 | 0.6200 |
| Consolidation (合併)          | 0.708 | 0.788 | 0.7901 |  0.8142| 0.8148 | 0.6152 |
| Edema (浮腫)                  | 0.835 | 0.882 | 0.8878 | 0.8932 | 0.8992 | 0.6514 |
| Emphysema (氣腫)              | 0.815 | 0.829 | 0.9371 | 0.9254 | 0.9343 | 0.3718 |
| Fibrosis (纖維化)             | 0.769 | 0.767 | 0.8047 | 0.8304 | 0.8385 | 0.7409 |
| Pleural Thickening (胸膜增厚) | 0.708 | 0.765 | 0.8062 | 0.7831 | 0.7914 | 0.5483 |
| Hernia (疝)                  | 0.767 | 0.914 | 0.9164 | 0.9104 | 0.9206 | 0.4104 |

####
Area Under the Receiver Operating Characteristic (AUROC) is a performance metric that you can use to evaluate classification models. 

AUROC tells you about the model’s ability to discriminate between cases (positive examples) and non-cases (negative examples.) An AUROC of 0.8 means that the model has good discriminatory ability: 80% of the time, the model will correctly assign a higher absolute risk to a randomly selected patient with an event than to a randomly selected patient without an event. 

## Zebra Implementaiton

The test report after running the applciaiton in Zebra V2022.2.5 release and Alveo U50LV, the average AUROC is 0.559

### Step 1: Check Hardware Installation

#### lspci
```
demo@cx:/zebra/V2022.2.5$ sudo lspci -v -d 10ee:
01:00.0 Serial controller: Xilinx Corporation Device 8022 (prog-if 00 [8250])
        Subsystem: Xilinx Corporation Device 0007
        Flags: bus master, fast devsel, latency 0
        Memory at fc000000 (32-bit, non-prefetchable) [size=4M]
        Memory at fa000000 (32-bit, non-prefetchable) [size=32M]
        Capabilities: [40] Power Management version 3
        Capabilities: [48] MSI: Enable- Count=1/1 Maskable- 64bit+
        Capabilities: [70] Express Endpoint, MSI 00
        Capabilities: [100] Advanced Error Reporting
        Capabilities: [1c0] #19
        Capabilities: [350] Vendor Specific Information: ID=0001 Rev=1 Len=02c <?>
        Kernel driver in use: zebra
        Kernel modules: zebra
```
#### zebra_tools --checkCores
```
demo@cx:/zebra/V2022.2.5$ source ./settings.sh
demo@cx:/zebra/V2022.2.5$ zebra_tools --checkCores 
```
```
[ZEBRA] Log file: /home/demo/.mipsology/zebra/log/zebra_tools.20220904-182224.25379.log
[ZEBRA] ======================
[ZEBRA] MIPSOLOGY SAS (c) 2022
[ZEBRA] Zebra V2022.2.5
[ZEBRA] ======================
[ZEBRA] The command line is: "zebra_tools --checkCores".
[ZEBRA] Detect XIL_AU50 board 0 on PCIe slot 0000:01:00.
[ZEBRA] Check if board 0 system 0 core 0 can be used ... OK.
[ZEBRA] Check if board 0 system 1 core 0 can be used ... OK.
[ZEBRA] Check if board 0 system 2 core 0 can be used ... OK.
[ZEBRA] Check if board 0 system 3 core 0 can be used ... OK.
[ZEBRA]
[ZEBRA] No HW assertion detected.
```

### Setp 2:  Zebra Settings
```
$ zebra_config --add runSession.enableTimeStatistics=true 
$ zebra_config --add runOptimization.frequency=500 
$ zebra_config --add debug.enableSubBatch=False 
$ zebra_config --add runSession.directory=quant_zebra 
$ zebra_config --add quantization.minimalBatchSize=100 
```

```
$ zebra_config --add quantization.mode=dynamic (default=constrainedCalibrationV1.5)
$ zebra_config --add quantization.forceSatCheckOnLastLayer=false (default=true)
$ zebra_config --add quantization.algorithmVersion=1.0 (default=3.1)
$ zebra_config --add quantization.ignoreNegativeValueOnLastLayer=false (default=true)
$ zebra_config --add runOptimization.addOptimizers=PrecisionRecovery:RUN
```

#### For performance optimization
```
zebra_config --add runOptimization.frequency=575
zebra_config --add memoryTuning.algorithm=PMN
```
#### zebra.ini
```
demo@cx9:/nvme/zebra/V2022.2.5$ zebra_config --list-all
[ZEBRA] ======================
[ZEBRA] MIPSOLOGY SAS (c) 2022
[ZEBRA] Zebra V2022.2.5
[ZEBRA] ======================
[ZEBRA] List all config files:
[ZEBRA]
[ZEBRA] [runSession]
        enableTimeStatistics=true
        directory=quant_zebra
[debug]
        enableSubBatch=False
[quantization]
        minimalBatchSize=100
        mode=dynamic
[runOptimization]
        addOptimizers=PrecisionRecovery:RUN
[ZEBRA] [runSession]
        precision=INT8
[ZEBRA] [runSession]
        enableTimeStatistics=true
        directory=quant_zebra
[debug]
        enableSubBatch=False
[quantization]
        minimalBatchSize=100
        mode=dynamic
[runOptimization]
        addOptimizers=PrecisionRecovery:RUN
[ZEBRA] [log]
        enable=false
```

### Setp 3: Launch Zebra Docker
```
## /zebra/run_docker.sh
$ cd /zebra/V2022.2.5
$ source ./settings.sh
$ ./examples/docker/run.sh
```
### Setp 4: Run application inside Zebra Docker
```
## /zebra/CheXNet/run.sh
$ cd zebra/CheXNet
$ unset LD_PRELOAD 
$ ZEBRA_DEBUG_NN3=true python3 model_custom.py 
```

#### Step 5: Example of log file
```
Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /home/demo/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
100%|███████████████████████████████████████████████████████████████████████████████| 30.8M/30.8M [00:03<00:00, 8.67MB/s]
=> loading checkpoint
=> loaded checkpoint
./model_custom.py:113: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  input_var = torch.autograd.Variable(inp.view(-1, c, h, w), volatile=True)
Running inference 0
Len of in: 4 torch.Size([4, 10, 3, 224, 224])
Len of input: 40 torch.Size([40, 3, 224, 224])
/usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
Running inference 1
Len of in: 4 torch.Size([4, 10, 3, 224, 224])
Len of input: 40 torch.Size([40, 3, 224, 224])
...
Running inference 5607
Len of in: 4 torch.Size([4, 10, 3, 224, 224])
Len of input: 40 torch.Size([40, 3, 224, 224])
Running inference 5608
Len of in: 1 torch.Size([1, 10, 3, 224, 224])
Len of input: 10 torch.Size([10, 3, 224, 224])
INFO :  Total Images   = 224320
INFO :  Execution Time = 8127.62

The average AUROC is 0.559
The AUROC of Atelectasis is 0.6138711490423786
The AUROC of Cardiomegaly is 0.533704529002345
The AUROC of Effusion is 0.8328125750171148
The AUROC of Infiltration is 0.3729686595322937
The AUROC of Mass is 0.3908945257304344
The AUROC of Nodule is 0.6040564932757174
The AUROC of Pneumonia is 0.5178404170255904
The AUROC of Pneumothorax is 0.6200046283676344
The AUROC of Consolidation is 0.6152587184878242
The AUROC of Edema is 0.6514502004561119
The AUROC of Emphysema is 0.3718691181430833
The AUROC of Fibrosis is 0.7409481605196289
The AUROC of Pleural_Thickening is 0.5483650284364993
The AUROC of Hernia is 0.41044765009750944


```

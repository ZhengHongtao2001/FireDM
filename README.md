# FireDM
Official code for 'FireDM: A weakly-supervised approach for massive generation of multi-scale and multi-scene fire segmentation datasets'
<p align="center">
<img src="./asset/1.jpg" width="800px"/>  
<br>
</p>


## Our Contribution
- [x] We propose a data augmentation framework FireDM, that can train to generate an infinite fire segmentation dataset with minimal reliance on a training dataset.

### Installation
```sh
conda create -n FireDM python=3.8
```
CUDA==11.3, CUDNN==8.2.0
Install the corresponding torch==1.10.2, please refer to [pytorch](https://pytorch.org/get-started/previous-versions/).
Such as:
```sh
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html  
```
Then install other packages:
```sh
python -m pip install -r requirements.txt
```

Select one of the following pre-training weight files: [Stable Diffusion XL Base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main) and place them in the ```./dataset/ckpts``` directory.


### Training Dataset Prepare
- I have placed the [BoWFire](https://ieeexplore.ieee.org/abstract/document/7314551) dataset used for training in a standardized format under the [Link](https://drive.google.com/file/d/1lilG-1MkrV6wLILSVZDuMMz-4wQiqjCb/view?usp=sharing), and you can follow this example to configure the contents of the dataset yourself to fit your project.

The overall format arrangement is consistent with the VOC dataset:


```
data/
    PascalVOC12/
	JPEGImages
	SegmentationClassAug
	splits/
	     train_aug.txt
```


Besides, you also need to order the prompt txt files as follows:

```
dataset/
	Prompts_From_GPT/
		VOC/
                  fire.txt
```

### Running Command
Training the Fire-deocder


```sh
# For Training Mask-Decoder
sh ./script/train.sh
```


```sh
# Generate synthetic data for VOC
sh ./script/generation.sh
```


 ## Acknowledgements
This work draws inspiration from the following contributions. We express our gratitude for these outstanding contributions and encourage citation of the following works when referencing our own:

- [Grounded-Diffusion](https://github.com/Lipurple/Grounded-Diffusion)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) 
- [GenPromp](https://github.com/callsys/GenPromp)
- [Depthformer](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/633580d0ed7395734a034abfebf2ddb6ad579d75/configs/depthformer)
- [DeepFashion-MM](https://github.com/yumingj/DeepFashion-MultiModal)
- [DatasetDM](https://github.com/showlab/DatasetDM)


## Citation

```
@article{zheng2024firedm,
  title={FireDM: A weakly-supervised approach for massive generation of multi-scale and multi-scene fire segmentation datasets},
  author={Zheng, Hongtao and Wang, Meng and Wang, Zilong and Huang, Xinyan},
  journal={Knowledge-Based Systems},
  pages={111547},
  year={2024},
  publisher={Elsevier}
}
```


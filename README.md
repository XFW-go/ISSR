# ISSR
This is a Tensorflow implement of ISSR

Integrating Semantic Segmentation and Retinex Model for Low Light Image Enhancement. In ACMMM'20 <br>
[Minhao Fan*](https://xfw-go.github.io/), [Wenjing Wang*](https://daooshee.github.io/website/), [Wenhan Yang](https://flyywh.github.io/), [Jiaying Liu](http://www.icst.pku.edu.cn/struct/people/liujiaying.html). <br>

[Project Page & Dataset](https://mm20-semanticreti.github.io/)

<img src="figs/Fig-1.png" width="800px"/>

### Requirements ###
1. Python >= 3.5.0
2. Tensorflow >= 1.9.0
3. numpy, PIL
An available config of env for conda is in the environment.yaml

### Testing  Usage ###
To quickly test your own images with our model, you can just run through
```shell
python main.py --use_gpu=1 --gpu_idx=0 --gpu_mem=0.5 --phase=test --test_dir=/path/to/your/test/dir/ --save_dir=/path/to/save/results/ --decom=0
```

### Training Usage ###
First, download train/val/test data set from [our project page](https://mm20-semanticreti.github.io/) and unzip the files.
You can organize your dataset structure and modify the corresponding part in `main.py`.
Run
```shell
python main.py --use_gpu=1 --gpu_idx=0 --gpu_mem=0.5 --phase=train \
    --epoch=100 --batch_size=10 --patch_size=48 --start_lr=0.001 --eval_every_epoch=20 \
    --checkpoint_dir=./ckpts --sample_dir=./sample
 ```
 Tips:
 1. The enhancement performance is highly dependent on training parameters. So if you change the default parameters, you might get some weird results.
 
 ### Citation ###
 ```
 @inproceedings{FanWY020,
  author    = {Minhao Fan and
               Wenjing Wang and
               Wenhan Yang and
               Jiaying Liu},
  title     = {Integrating Semantic Segmentation and Retinex Model for Low-Light
               Image Enhancement},
  booktitle = {{MM} '20: The 28th {ACM} International Conference on Multimedia, Virtual
               Event / Seattle, WA, USA, October 12-16, 2020},
  pages     = {2317--2325},
  publisher = {{ACM}},
  year      = {2020},
}
```
 

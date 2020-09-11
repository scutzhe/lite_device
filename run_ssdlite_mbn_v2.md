> # 训练记录

# 1. 训练示例代码

```
python train.py \
--datasets ~/work/YOLOv3/datasets/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007 ~/work/YOLOv3/datasets/VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012 \
--validation_dataset ~/work/YOLOv3/datasets/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007 \
--net mb2-ssd-lite \
--base_net models/mb2-imagenet-71_8.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 200 \
--validation_epochs 5 \
--num_epochs 200
```

## 运行demo
```
python detection_demo.py \
mb2-ssd-lite \
saved_model/mb2-ssd-lite-Epoch-15-Loss-2.1904499530792236.pth \
saved_model/voc-model-labels.txt \
/home/qindanfeng/work/deep_learning/datasets/vehicle_datasets/JPEGImages/0000f77c-6257be58.jpg
```

## 性能评估
```
python eval.py \
--net mb2-ssd-lite  \
--dataset ~/work/YOLOv3/datasets/VOC/VOCtest_06-Nov-2007/VOCdevkit/VOC2007 \
--trained_model saved_model/mb2-ssd-lite-Epoch-150-Loss-2.8759542611929088.pth \
--label_file models/voc-model-labels.txt
```

------


# 2. 2020.08.04 训练车辆检测模型
```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net mb2-ssd-lite \
--base_net models/mb2-imagenet-71_8.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 200 \
--validation_epochs 5 \
--num_epochs 200
```

## resume
```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net mb2-ssd-lite \
--resume saved_model/mb2-ssd-lite-Epoch-35-Loss-2.0251203179359436.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 200 \
--validation_epochs 5 \
--num_epochs 200
```

## 性能测试
```
python eval.py \
--net mb2-ssd-lite  \
--dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--trained_model saved_model/mb2-ssd-lite-Epoch-30-Loss-2.0793707370758057.pth \
--label_file saved_model/voc-model-labels.txt
```

------

# 3. 2020.08.08 代码优化，车辆检测模型，新的训练脚本如下 

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite \
--base_net models/mb2-imagenet-71_8.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 100 \
--validation_epochs 5 \
--num_epochs 100
```

## resume
```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net mb2-ssd-lite \
--resume saved_model/mb2_ssd_lite/last.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 100 \
--validation_epochs 5 \
--num_epochs 100
```

## 性能测试
```
python eval.py \
--dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--trained_model saved_model/mb2_ssd_lite/best.pth \
--label_file saved_model/voc-model-labels.txt
```

## 4. 训练车辆检测模型，epoch185，mAP为0.6056，测试在实际视频的效果

```
mb2-ssd-lite-Epoch-185-Loss-1.5812279284000397.pth
```

![](assets/epoch_185_mAP.svg)

![](assets/1.gif)

![2](assets/2.gif)

![3](assets/3.gif)

------

# 5. 2020.08.11 使用小样本车辆数据集进行训练，训练集5000，测试集1000

- [x] 1） 设置epoch=100， 余弦学习器lr=0.01，t_max=100，使用预训练模型，小样本训练

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample \
--base_net models/mb2-imagenet-71_8.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 100 \
--validation_epochs 5 \
--num_epochs 100 \
--use_small_sample
```

![image-20200812085052060](assets/image-20200812085052060.png)

![image-20200812085109251](assets/image-20200812085109251.png)

![image-20200812085120253](assets/image-20200812085120253.png)

看到模型并没有收敛，可能余弦学习器设置不正确，将t_max设置比epoch大很多再重新训练。

设置epoch=100， lr=0.01，t_max=300，重新训练

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_tmax_300 \
--base_net models/mb2-imagenet-71_8.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 300 \
--validation_epochs 5 \
--num_epochs 100 \
--gpus 1 \
--use_small_sample
```

resume，设置epoch=200，继续训练

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_tmax_300 \
--resume saved_model/mb2_ssd_lite_small_sample_tmax_300/last.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 300 \
--validation_epochs 5 \
--num_epochs 200 \
--gpus 1 \
--use_small_sample
```

resume，设置epoch=400

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_tmax_300 \
--resume saved_model/mb2_ssd_lite_small_sample_tmax_300/last.pth  \
--scheduler cosine \
--lr 0.01 \
--t_max 300 \
--validation_epochs 5 \
--num_epochs 400 \
--gpus 1 \
--use_small_sample
```

![image-20200818144606499](assets/image-20200818144606499.png)

![image-20200818144658957](assets/image-20200818144658957.png)

![image-20200818144713200](assets/image-20200818144713200.png)

![image-20200818144723427](assets/image-20200818144723427.png)

- [x] 2）根据余弦学习器lr下降曲线，使用SGD重新训练一次，看是否可以得到同样的效果。

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--base_net models/mb2-imagenet-71_8.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 100 \
--gpus 2 \
--use_small_sample
```

resume，epoch=150

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 150 \
--gpus 2 \
--use_small_sample
```

resume，epoch=200

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 201 \
--gpus 2 \
--use_small_sample
```

resume, epoch=250

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 251 \
--gpus 2 \
--use_small_sample
```

resume, epoch=350, milestones=252

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 351 \
--milestones 252 \
--gpus 2 \
--use_small_sample
```

resume, epoch=400, milestones=252

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 401 \
--milestones 252 \
--gpus 2 \
--use_small_sample
```

resume,epoch=450,milestones=252,402,gamma=0.5

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 451 \
--milestones 252,402 \
--gamma 0.5 \
--gpus 2 \
--use_small_sample
```

resume,epoch=550,milestones=252,402,gamma=0.5

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 551 \
--milestones 252,402 \
--gamma 0.5 \
--gpus 2 \
--use_small_sample
```

resume,epoch=600,milestones=252,402,552 gamma=0.5

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 601 \
--milestones 252,402,552 \
--gamma 0.5 \
--gpus 2 \
--use_small_sample
```

resume,epoch=650,milestones=252,402,552 gamma=0.5

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 651 \
--milestones 252,402,552 \
--gamma 0.5 \
--gpus 2 \
--use_small_sample
```

resume,epoch=700,milestones=252,402,552 gamma=0.5

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 701 \
--milestones 252,402,552 \
--gamma 0.5 \
--gpus 2 \
--use_small_sample
```

![image-20200818144151643](assets/image-20200818144151643.png)

![image-20200818144204163](assets/image-20200818144204163.png)

![image-20200818144218238](assets/image-20200818144218238.png)

![image-20200818144233065](assets/image-20200818144233065.png)

结论：1)与2两种学习器的最终效果差不多，但余弦学习器学习曲线更加平滑。

- [ ] 3）对比大批量训练和小批量训练的区别。

epoch=100, batch_size=8

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep_minibatch \
--base_net models/mb2-imagenet-71_8.pth  \
--scheduler multi-step \
--batch_size 8 \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 100 \
--gpus 1 \
--use_small_sample
```

学习率太大，曲线震荡的很厉害，调整学习率。epoch=150，milestones=100，gamma=0.5

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep_minibatch \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep_minibatch/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 151 \
--milestones 100 \
--gamma 0.5 \
--gpus 1 \
--use_small_sample
```

epoch=250

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_small_sample_mulstep_minibatch \
--resume saved_model/mb2_ssd_lite_small_sample_mulstep_minibatch/last.pth  \
--scheduler multi-step \
--lr 0.1 \
--validation_epochs 5 \
--num_epochs 251 \
--milestones 100 \
--gamma 0.5 \
--gpus 1 \
--use_small_sample
```



- [ ] 4）使用超参搜索框架找到合适的超参数。

- [ ] 5）将小数据集上搜索的超参数应用到大数据集上，看下效果如何。

```
python train.py \
--datasets /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--validation_dataset /home/qindanfeng/work/deep_learning/datasets/vehicle_datasets \
--net_name mb2_ssd_lite_mulstep \
--base_net models/mb2-imagenet-71_8.pth  \
--scheduler cosine \
--lr 0.1 \
--t_max 350 \
--validation_epochs 5 \
--num_epochs 200 \
--gpus 2 
```



- [ ] 6）使用iou_loss、giou_loss、diou_loss、ciou_loss。并形成性能对比。

- [ ] 7）修改难例挖掘，原始是选择loss最大的负样本加入反向梯度求解中，但是车辆数据集中的大样本远远小于小样本，因此可能会造成大部分在学习小样本，而我们的学习目标集中在大、中目标上，因此需要对难例的选定重新设计，给出两种方案，方案一：设定图片中的有效区域，存在该区域的负样本且loss较大时选为反向传播的目标；方案二，对目标的大小进行约束，只是大于某个尺度，且为loss较大的负样本才参与梯度求解。
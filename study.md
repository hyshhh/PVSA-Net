#执行代码
## 训练指令：
```
python tools/train.py  ${配置文件} [可选参数]
```
```buildoutcfg
--work-dir ${工作路径}: 重新指定工作路径
--amp: 使用自动混合精度计算
--resume: 从工作路径中保存的最新检查点文件（checkpoint）恢复训练
```


python tools/train.py configs/biformer/biformer_mm-20k_chase_db1-512x512.py --work-dir mmseg_log\biformer
python setup.py install重新编译
```
如果希望从指定的检查点上恢复训练：
```buildoutcfg
python tools/train.py ${配置文件} --resume --cfg-options load_from=${检查点}
```

## 测试指令：
```buildoutcfg
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${OUTPUT_DIR}
```
测试biformer组成的分割模型指令如下：
```buildoutcfg
python tools/test.py configs/biformer/biformer_mm-20k_chase_db1-512x512.py mmseg_log/biformer/iter_2000.pth --show-dir mm
seg_log/biformer/visualizations
python tools/get_flops.py configs/biformer/biformer_mm-20k_chase_db1-512x512.py --shape 512 512
```
可以指定可视化结果输出目录，得到测试样例。

## 配置文件
configs文件夹下的_base_文件夹下存放了基础模型配置文件、数据集配置文件、学习调度配置文件

configs文件夹下还存放了指定任务配置文件，执行训练和测试任务时需要输入指定任务配置文件（包括模型、数据集和学习器等参数）
 
模型结构和参数需要更改文件中的_base_/models文件

数据集图像尺寸、路径等参数在_base_/dataset文件

batch_size文件中, 迭代次数，学习率衰减策略等参数在_base_/schedules文件中

更多配置文件的相关信息可见：https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/4_train_test.html

## 数据集
具体数据集图片文件和标签文件存放在 data 文件夹中

数据集转换脚本存放在tools/dataset_converters中
# lite_device
1 实验记录:  
```
1. model=models/80_1.871617062886556.pth模型保存  
2. model=models/100_1.826955000559489.pth这个模型比epoch80的效果要好  
```
2 生成MNN的脚本  cd /home/zhex/package/MNN/build
```
./MNNConvert -f ONNX --modelFile /home/zhex/git_me/lite_device/onnx_model/electronic_tag.onnx --MNNModel /home/zhex/git_me/lite_device/onnx_model/electronic_tag.mnn --bizCode biz
```
3 mnn推断程序示例  
```
# 检测推断程序
mnn_inference_detection.py
# 识别推断
mnn_inference_device_recognition.py(尝试写法1,能否正确推断不能保证)
mnn_inference_device_recognition_2.py(可以正常推断)
mnn_inference_device_recognition_3.py(尝试写法3,能否正确推断不能保证)
```

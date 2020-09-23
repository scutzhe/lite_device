# lite_device
实验记录:  
1. model=models/80_1.871617062886556.pth模型保存  
2. model=models/100_1.826955000559489.pth这个模型比epoch80的效果要好  
3. 


#生成MNN的脚本  cd /home/zhex/package/MNN/build
./MNNConvert -f ONNX --modelFile /home/zhex/git_me/lite_device/onnx_model/electronic_tag.onnx --MNNModel /home/zhex/git_me/lite_device/onnx_model/electronic_tag.mnn --bizCode biz
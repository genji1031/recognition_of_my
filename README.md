# recognition_of_my
识别人脸
# 前期准备
  input_data.py 中  train_dir = 是需要我们遍历的训练数据图片存放位置，每一类图片请放置在同一个文件夹中。比如有4类明星人脸需要训练就放4个文件夹，并在每一个文件夹上标注该明星名字即可
  
  train.py
  train_dir = 'D:/myself/recognitionData/train_photos'  # 训练样本的读入路径
  logs_train_dir = 'D:/myself/recognitionData/savedata'  # 训练好的模型pickle文件存储路径
  
  test.py
  logs_train_dir = 'D:/myself/recognitionData/savedata' 更改这里的路径
  最后将测试路径修改即可
  img = Image.open('D:/myself/recognitionData/test/wwa.jpg')
  

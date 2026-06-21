这个我采取的另一个思路，我不用视频了，我用的是图片训练集，就是让ai学习五官，嘴巴眼睛和手
系统能够检测以下危险驾驶行为：
  闭眼驾驶（Eye Closure Detection）
  打哈欠行为（Yawning Detection）
  手部干扰行为（Hand Distraction Detection）
然后规则是我自己定义的
  用了HOG：特征提取（Histogram of Oriented Gradients），其实就是提取出嘴巴和眼睛特征，然后，然后数据集也是关注的嘴巴眼睛，所以ai学出来可以判断
  然后用的Linear SVM 课上有 眼睛状态分类Open（睁眼） Closed（闭眼） 嘴巴状态分类Yawn（打哈欠） No Yawn（正常状态）

系统工作流程：
  数据集图片
  ↓
  HOG特征提取
  ↓
  SVM模型训练
  ↓
  生成模型文件（.pkl）
  ↓
  摄像头实时检测
  ↓
  危险等级判断
  ↓
  报警提示


要运行的话：
pip install -r requirements.txt
python realtime_detection.py
训练模型：
python train_models.py

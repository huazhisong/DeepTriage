# TextCNN
使用CNN进行文本分类，工具为TensorFlow

for i in range(1, 11):
   t = np.loadtxt('class_'+str(i)+'.csv', delimiter=',',dtype=str)
   classes.append(t)
Eclipse文件分析
每个训练集要预测的类别数目
[259, 571, 873, 1137, 1436, 1692, 1921, 2122, 2310, 2499]
每个文件的句子长度
194, 217, 220, 222, 222, 225, 229, 231, 234, 236

Mozzila文件分析
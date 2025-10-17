# Great_Idea_LAB-Handwriting_Digit_Recognition
It's a project that reproduces the CNN model of handwriting digit recognition using C++/OpenCV
关于CNN_model的原理与架构参见`slide.pdf` 
\newline
关于代码的实现：
\newline
暂时用结构体储存各层参数，没有写成类。
\newline
然后是各层的前向传播以及必要的激活函数等，并用`CNN_forward`集成。
\newline
然后是误差的计算。
\newline
然后是各层的前向传播以及必要的激活函数等，并用`CNN_backward`集成。
\newline
然后是卷积核/偏置参量的更新，并用`CNN_update`集成。
\newline
最后是MINST数据集的读入以及训练/测试函数。

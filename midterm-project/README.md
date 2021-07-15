本次作业模板为IEEE CVPR $\LaTeX$ Template，可以自行下载。其使用方法如下:

其它部分遵守$\LaTeX$语法即可，主要难上手的地方在于参考文献：

1.在百度学术上搜索引用的文章，点击引用，选择BibTeX格式，将内容复制到文件egbib.bib内；

2.在tex文件前加\usepackage{cite}，在引用处加\cite{article name}；

3.前面的egbib.bib结合文献模板ieee_fullname.bst，即可完成参考文献。

编译时，有的可能只需要编译一次XeLaTeX，有的可能需要XeLaTeX->BibTeX->XeLaTeX->XeLaTeX，视情况而定。

src：测例代码文件夹

picture：测试结果文件夹

运行代码时需改一下引用文件所在目录（不过也不需要运行代码。。。）

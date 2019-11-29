# Size encoding/decoding in COCO dataset


This code was used in the paper [1] as part of experiments to answer a question if size of an object can be encoded in an image by resampling (rescaling) the image. This code uses COCO dataset (http://cocodataset.org/). For our experiments we used three categories. Nevertheless, the code provides tools for repeating the described experiments with all categories that defined in the dataset. Moreover, some filtering tools are also provided.<br>



<H2>Required Libraries</H2><br>


To process the COCO dataset we used COCO API library probided by the COCO team (https://github.com/cocodataset/cocoapi/).<br><br>

Tensorflow version: 1.11.0<br><br>

Keras version: 2.2.4<br><br>

Python: 3.6.7<br><br>

sklearn: 0.20.0<br><br>

<H2>Execution parameters</H2><br>

<b>--tmp_dir:&lt;path to a directory&gt;</b><br>
The command is not required. Takes as an argument path to a folder.<br>
Define <b>Working Derectory</b>.The application stores intermediate data as well as results of experiments in files. By default all the data is stored in the same folder where the code is located. Because it can be messy if several experiments performed each experiment can be assigned to a particular forlder such that all temporal and resulting data will be stored in the defined folder.<br><br>

<b>--load</b><br>
The command is not required. Takes no arguments.<br>
If provided then application will <b>load COCO dataset</b> from the Internet. Result will be stored to <b>&lt;Working Derectory&gt;/data/</b> folder. Note: default tools provide small downloading speed as well as there were cases of connection errors that required to restart loading again. Thus, <b>we recommend download COCO source files manually</b> and place them in <b>&lt;Working Derectory&gt;/data/</b> folder.<br><b>The application requires:</b><br>
2014 Train images (http://images.cocodataset.org/zips/train2014.zip).<br>
2014 Validation images (http://images.cocodataset.org/zips/val2014.zip).<br>
2017 Train images (http://images.cocodataset.org/zips/train2017.zip).<br>
2017 Validation images (http://images.cocodataset.org/zips/val2017.zip).<br>
2014 Train/Val annotations (http://images.cocodataset.org/annotations/annotations_trainval2014.zip).<br>
2017 Train/Val annotations (http://images.cocodataset.org/annotations/annotations_trainval2017.zip).<br><br>

<b>--unzip_preloaded</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br>

<b>--category:&lt;&gt;</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br>

<b>--filter</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br>

<b>--smallest_axe:&lt;&gt;</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br>

<b>--largest_axe:&lt;&gt;</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br>

<b>--median:&lt;&gt;</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br>

<b>--resize:&lt;&gt;</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br>

<b>--run_cv</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br>

<b>--auc</b><br>
The command is not required. Takes as an argument path to a folder.<br>
text<br><br><br><br>


Examples:<br>
text<br><br>

<H2>Execution issues</H2><br>

In case if you have problems with running the code, please contact the corresponding authore of the paper [1].


<H2>Reference</H2><br>

[1] TBA


<H2>Citation</H2><br>

If you use this code in your work, please cite it as:
TBA



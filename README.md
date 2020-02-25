# Size encoding/decoding in COCO dataset


This code was used in the paper [1] as part of experiments to answer a question if size of an object can be encoded in an image by resampling (rescaling) the image. This code uses COCO dataset (http://cocodataset.org/). For our experiments we used three categories. Nevertheless, the code provides tools for repeating the described experiments with all categories that defined in the dataset. Moreover, some filtering tools are also provided.<br>



<H2>Required Libraries</H2><br>


To process the COCO dataset we used COCO API library probided by the COCO team (https://github.com/cocodataset/cocoapi/).<br>
<b>NOTE: COCO API library is not supported for Windows OS.</b><br>
Tensorflow version: 1.11.0<br>
Keras version: 2.2.4<br>
Python: 3.6.7<br>
sklearn: 0.20.0<br><br>

<H2>Execution parameters</H2><br>

<b>--tmp_dir:&lt;path to a directory&gt;</b><br>
The command is not required. Takes as an argument path to a folder.<br>
Define <b>Working Derectory</b>.The application stores intermediate data as well as results of experiments in files. By default all the data is stored in the same folder where the code is located. Because it can be messy if several experiments performed each experiment can be assigned to a particular forlder such that all temporal and resulting data will be stored in the defined folder.<br>
Example:<br>
python3 main.py --tmp_dir:./experiment1/<br>
Will create <b>experiment1</b> folder and will store there all temporal results.
<br><br>

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
The command is required if all data files were downloaded manually. If defined, application will unzip all the files from <b>&lt;Working Derectory&gt;/data/</b> folder.<br>
Unzip source files.<br><br>

<b>--category:&lt;category1,category2,...,category_n&gt;</b><br>
The command is not required. Takes as an argument a list of category names provided in COCO dataset and which must be used for experiments in the application.<br>
COCO team annotated multiple objects from multiple categories (http://cocodataset.org/#explore). In our experiments we used three categories: <b>cat, bear, dog</b>. They are used as <b>default categories</b>.<br>
Example:<br>
python3 main.py --category:person,backpack,bicycle<br>
Set the application to perform the experiments on <b>person,backpack,bicycle</b> categories.
<br><br>

<b>--filter</b><br>
The command is required. Takes no argument.<br>
<b>Filter</b> images of defined object categories. After filtering compute the largest bounding box resolution for each category.<br>
Example:<br>
python3 main.py --filter<br>
Create a folder and store objects images of defined categories.
<br><br>

<b>--resize:&lt;Integer value: rescaling resolution&gt;</b><br>
The command is required. Rescale filtered images to the uniform size. Images will be rescaled into square image with image size equal to defined values. <b>NOTE:</b>In our experiments we used 640x640 pixel image resolution as an input for the CNN. Thus, usage other values but 640 require changes in code i.e. change CNN input image size.<br>
Example:<br>
python3 main.py --resize:640<br>
Resize <b>filtered</b> images into 640x640 images. Result will be used in the experiment.<br><br>

<b>--smallest_axe:&lt;Integer value: the smallest height/width image value&gt;</b><br>
The command is not required. Remove from the dataset images which height or width is smaller than defined value.<br>
Example:<br>
python3 main.py --smallest_axe:80<br>
<b>Remove</b> images which height or width is smaller than 80 pixels.<br><br>

<b>--largest_axe:&lt;Integer value: the largers height/width image value&gt;</b><br>
The command is not required. Remove from the dataset images which height or width is larger than defined value.<br>
Example:<br>
python3 main.py --largest_axe:600<br>
<b>Remove</b> images which height or width is larger than 600 pixels.<br><br>

<b>--median:&lt;Integer value: percent value of images in the dataset&gt;</b><br>
The command is not required. Labeling of images is done automatically. The median area of a source images before rescaling is used as a threshold for labeling. It may create cases where two images with a small difference in area are labeled into different classes. This command create a margine between two classes by removing defined percent of images which are the closest in term of area to the median.<br>
Example:<br>
python3 main.py --median:5<br>
<b>Remove</b> 5 percent of the dataset images which area is closest to the dataset median value.<br><br>

<b>--run_cv</b><br>
The command is required. Takes no arguments. Perform 5 times 2-fold cross-validation on the resulting dataset and computes CNN output predictions as well as accuracy for each cross-validation iteration for each defined category.<br>
Example:<br>
python3 main.py --run_cv<br>
Perform 5 times 2-fold cross-validation.<br><br>


<b>--auc</b><br>
The command is not required. Takes no argumens.<br>
Has to be executed if/after "--run_cv" is performed.
If defined, then application will compute Area Under Receiver Operating Characteristic Curve (AUROC) for each iteration of cross-valitation of for each selected category.
Example:<br>
python3 main.py --auc<br>
Computes AUROC for each iteration of cross-valitation of for each selected category.
<br><br>

<b>--regression</b><br>
The command is required. Takes no arguments. Perform 5 times 2-fold cross-validation regression training on the resulting dataset for each cross-validation iteration for each defined category.<br>
Example:<br>
python3 main.py --regression<br>
Perform 5 times 2-fold cross-validation regression training.<br><br>


<b>--mse</b><br>
The command is not required. Takes no argumens.<br>
Has to be executed if/after "--regression" is performed.
If defined, then application will compute Mean Square Error for the regression and Pearson Correlation Coefficient for labels and predictions.
Example:<br>
python3 main.py --mse<br>
Computes AUROC for each iteration of cross-valitation of for each selected category.
<br><br><br><br>


<b>Examples:</b><br>
It is possible to run the application iteratively. First is it possible to download the dataset and then, with the second run, filter categories from the dataset. Nevertheless, we recommend to run the application with all the defined parameters at once. For example, in case when the required files are manually downloaded the experement performace can be done by running following command:<br>
python3 main.py --unzip_preloaded --filter --resize:640 --run_cv --tmp_dir:./example_dir/<br>
The application will unzip COCO dataset files from the ./example_dir/data/ folder and perform the experiment.<br>
For filtering, the same command can be modified:<br>
python3 main.py --unzip_preloaded --filter --resize:640 --run_cv --tmp_dir:./example_dir/ --smallest_axe:80 --median:10<br>
The application will unzip COCO dataset files from the ./example_dir/data/ folder, remove images according to defined filtering commands and perform the experiment.
<br><br>

<H2>Issues</H2><br>

In case if you have problems with running the code, please contact the corresponding authore of the paper [1].


<H2>Reference</H2><br>

[1] TBA


<H2>Citation</H2><br>

If you use this results of this code in your work, please cite it as:
TBA






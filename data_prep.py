import sys
ver = sys.version
import zipfile
if ver[0] == '3':
    import urllib.request  as urllib2 
else:
    import urllib2
import os.path
from os import listdir
from os.path import isfile, join
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import math
from skimage.transform import resize


def resize_patches_to_size(params):

    catName = params['category']
    save_dir = params['tmp_dir'] + 'training_data/'
    resize_value = params['resize_to']

    for curCat in catName:
        median_area = get_category_median_area(curCat, params)
        labels_file = params['tmp_dir'] + '{}_labels.txt'.format(curCat)
        check_folder(save_dir + '{}/'.format(curCat))
        cur_folder = save_dir + '{}/'.format(curCat)
        src_folder = params['tmp_dir'] + 'temporal_patches/{}/'.format(curCat)
        onlyfiles = [f for f in listdir(src_folder) if (isfile(join(src_folder, f)) and ('.jpg' in f))]
        labelFile = open(labels_file,"w")
        for img_path in onlyfiles:
            I = io.imread(src_folder+img_path)
            cur_area = I.shape[0]*I.shape[1]
            resized_image = resize(I, (resize_value, resize_value),
                       anti_aliasing=False)
            io.imsave(cur_folder + img_path, resized_image)
            if params['regression']:

                c_label = (float(cur_area) / (params['resize_to']*params['resize_to']))*2 - 1
                txt = src_folder + img_path + ':' + str(c_label) + '\n'
                labelFile.write(txt)

            else:

                if cur_area < median_area:
                    txt = src_folder+img_path+':0\n'
                    labelFile.write(txt)
                else:
                    txt = src_folder+img_path+':1\n'
                    labelFile.write(txt)
        labelFile.close()


def get_category_median_area(curCat,params):
    areaListPath = params['tmp_dir'] + 'areaList_{}.txt'.format(curCat)
    lfile = open(areaListPath)
    lines = []
    for line in lfile: 
        line = int(line)
        lines.append(line)
    
    lfile.close()
    medianArea = median(lines)

    if params['median']>0:
        lines = np.array(lines)
        lines = abs(lines - medianArea)
        lines.sort(0)
        num_to_cut = len(lines)/100*params['median']
        num_to_cut = int(num_to_cut)
        threshold = lines[num_to_cut]
        create_median(curCat, medianArea, threshold, params)
    return medianArea

def filter_dataset(params):
    catName = params['category']
    setType = ['train2014','train2017','val2014','val2017']
    for cur_cat in catName:
        extract_temporal_caterories(setType,cur_cat,params)
    print_max_width_height(catName, params)



def extract_temporal_caterories(data_sets,curCat,params):

    save_dir = params['tmp_dir'] + 'temporal_patches'
    check_folder(save_dir)
    img_id = 0
    for cur_set in data_sets:
        save_path = save_dir + '/{}/'.format(curCat)
        check_folder(save_path)
        dataDir=os.getcwd()
        annFile='{}/data/annotations/instances_{}.json'.format(dataDir,cur_set)
        coco=COCO(annFile)
        catIds = coco.getCatIds(catNms=[curCat])[0]
        imgIds = coco.getImgIds(catIds=catIds )
        
        
        
        for cImgID in imgIds:
            img = coco.loadImgs(cImgID)[0]
            IPath = '{}/data/{}/{}'.format(os.getcwd(),cur_set,img['file_name'])
            I = io.imread(IPath)
            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)
            #show_images(I)
            cAnn_id = 0
            
            inc_id = False

            for cAnn in anns:
                cID = cAnn['category_id']
                if cID != catIds:
                    continue
                bbox = cAnn['bbox']
                bbox[0] = math.trunc(bbox[0])
                bbox[1] = math.trunc(bbox[1])
                bbox[2] = math.trunc(bbox[2]+1)
                bbox[3] = math.trunc(bbox[3]+1)
                cropped = I[bbox[1]:(bbox[1]+bbox[3]),bbox[0]:(bbox[0]+bbox[2])]
                cur_im_path = save_path +'{}_{}.jpg'.format( str(img_id), str(cAnn_id))

                #show_images(cropped)
                # filtering
                is_accepted = accepted_by_size(cropped, params)
                if is_accepted:
                    io.imsave(cur_im_path, cropped)
                    cAnn_id = cAnn_id + 1
                    inc_id = True
            if inc_id:
                img_id +=  1


def accepted_by_size(cropped, params):
    result = True

    smallest_size = min(cropped.shape[0],cropped.shape[1])
    largest_size = max(cropped.shape[0],cropped.shape[1])

    if params['smallest_axe']>0:
        if smallest_size<=params['smallest_axe']:
            result = False
    if params['largest_axe']>0:
        if largest_size>=params['largest_axe']:
            result = False
    return result

            
def save_area(cur_area_path, segArea):
    res_file = open(cur_area_path,"a") 
    res_file.write(str(segArea))
    res_file.close() 

def print_max_width_height(catName,params):
    result = []
    maxH = 0
    maxW = 0
    for curCat in catName:

        path = params['tmp_dir'] + 'temporal_patches/{}/'.format(curCat)
        onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)) and ('.jpg' in f))]
        cur_file_name = params['tmp_dir'] + 'areaList_{}.txt'.format(curCat)

        fl = open(cur_file_name,"w") 
        areaList = []
        for impath in onlyfiles:
            cfilePath = path + impath
            I = io.imread(cfilePath)
            maxH=max(maxH, I.shape[0])
            maxW=max(maxW, I.shape[1])
            fl.write(str(I.shape[0]*I.shape[1])+'\n' )
            areaList.append(I.shape[0]*I.shape[1])
        message = 'Category Name: ' + curCat + '\n'
        message += 'Maximum Height: ' + str(maxW)+'\n'
        message += 'Maximum Width: ' + str(maxW)+'\n'
        message += 'Image Area Saved to: ' + cur_file_name+'\n\n\n'
        
        fl.close()
    return result


def create_median(catName, med_val, threshold, params):


    path = params['tmp_dir'] + 'temporal_patches/{}/'.format(catName)
    onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)) and ('.jpg' in f))]
    for fname in onlyfiles:
        fpath = path +fname
        img = io.imread(fpath)
        cArea = img.shape[0]*img.shape[1]
        if abs(cArea-med_val)<threshold:
            os.remove(fpath)
    update_areaList(catName,params)


def update_areaList(catName, params):
    path = params['tmp_dir'] + 'temporal_patches/{}/'.format(catName)
    flist_path = params['tmp_dir'] + 'areaList_{}.txt'.format(catName)
    os.remove(flist_path)
    new_areas = []
    onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)) and ('.jpg' in f))]
    for fname in onlyfiles:
        fpath = path +fname
        img = io.imread(fpath)
        cArea = img.shape[0]*img.shape[1]
        new_areas.append(cArea)
    f = open(flist_path,'w')
    for cArea in new_areas:
        line = "{}\n".format(cArea)
        f.write(line)
    f.close()


def prepare_data(params):

    img, ann = get_links()
    unzip_path = list()
    
    for url in img:
        cpath = os.getcwd()
        cpath = cpath + '/data/'
        file_name = cpath + url.split('/')[-1]
        load_data(url, file_name)
        unzip_path.append(file_name)

    for url in ann:
        cpath = os.getcwd()
        cpath = cpath + '/data/'
        file_name = cpath + url.split('/')[-1]
        load_data(url, file_name)
        unzip_path.append(file_name)
    
    unzip_files(unzip_path)


def unzip_preloaded():
    path = os.getcwd()+'/data/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for j in range(len(onlyfiles)):
        onlyfiles[j] = path + onlyfiles[j]
    unzip_files(onlyfiles)

def unzip_files(unzip_path):
    for path in unzip_path:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd() + '/data/')



def check_folder(dir_name):
    if os.path.exists(dir_name):
        return
    else:
        os.makedirs(dir_name)


def get_links():
    images = ('http://images.cocodataset.org/zips/train2014.zip',
    'http://images.cocodataset.org/zips/val2014.zip',
    'http://images.cocodataset.org/zips/train2017.zip',
    'http://images.cocodataset.org/zips/val2017.zip')

    annotations = ('http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip')

    return images, annotations






def load_data(url,file_name):
    
    check_folder('data')
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: {} Bytes: {}".format(file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        status = r"[%3.2f%%]" % (file_size_dl * 100. / file_size)
        print(status,)

    f.close()

def median(lst):
    n = len(lst)
    s = sorted(lst)
    return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None


def show_images(I):
    plt.axis('off')
    plt.imshow(I)
    plt.show()
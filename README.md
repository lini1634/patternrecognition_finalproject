# patternrecognition_finalproject

+ Beyond bags of features spatial pyramid matching for recognizing natural scene categories, CVPR 2006

> 요약: 장면을 인식하여 분류하는데, 하나의 이미지를 분할하고 각 분할이미지마다 로컬피쳐로 히스토그램을 계산한다. Orderless bag of features의 확장판.


*이미지를 분할해서 히스토그램을 계산하는것은 장면 분류의 정확도를 높이기 위해 원래 있던 아이디어 인데, spatial pyramid로 했다는 것이 이 논문의 컨트리뷰션임.*

  + Bag of feature와의 차이점: 피쳐의 공간정보가 있다. 분할된 고정된 직사각형 윈도우를 이용한다. 
  + Multiresolution Histogram과 차이점: 로컬피쳐가 계산된 고정된 해상도에 공간 위치가 다름. 


### 단계
	1. 피쳐 추출: Dense SFIT로 얻어진 피쳐 정보를 k-means 군집화를 통해 시각적 vocabulary를 만든다.
	2. X이미지와 Y이미지의 히스토그램을 구하고 두 히스토그램의 최솟값으로 교차점을 찾는다.(-> 커널)
	3. 분할 이미지 셀의 넓이에 반비례하게 가중치를 둔다. (더 작은 셀일수록 공간정보를 많이 담고 있으므로 가중치를 높게 둠).
	4. 이미지의 히스토그램 분포를 SVM 분류기로 학습함. 
 
 
 
#### 참고

(1) https://github.com/CyrusChiu/Image-recognition  
(2) https://github.com/wihoho/Image-Recognition/blob/5dc8834dd204e36172815345f0abe5640a4a37ef/recognition/classification.py#L10  
(3) https://darkpgmr.tistory.com/125

```
!ls -lha kaggle.json

!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6

# 캐글연동을 위한 토큰 입력
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json


# 버전이 1.5.6 이 아니면, 진행할 수 없다
! kaggle -v

! kaggle competitions download -c 2019-ml-finalproject
! unzip 2019-ml-finalproject.zip

! yes | pip3 uninstall opencv-python
! yes | pip3 uninstall opencv-contrib-python
! yes | pip3 install opencv-python==3.4.2.16
! yes | pip3 install opencv-contrib-python==3.4.2.16
! yes | pip3 install kmc2
```

## 라이브러리  
```
import cv2
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report
import scipy.cluster.vq as vq
import pandas as pd
import kmc2
from sklearn.cluster import MiniBatchKMeans

```

## 데이터 로드
```
df_data=pd.read_csv('./Label2Names.csv',header=None)

DATA_ROOT_TRAIN="./train"
train_des=list()
train_labels=list()

for cls in tqdm(os.listdir(DATA_ROOT_TRAIN)):
  img_list=os.listdir(DATA_ROOT_TRAIN+'/'+cls)
  img_list.sort()
  

  if cls=='BACKGROUND_Google':
    label=102
  else:
    label=(df_data.index[df_data[1]==cls]+1).tolist()[0]

  for img in img_list:
    image=cv2.imread(DATA_ROOT_TRAIN+'/'+cls+'/'+img)
    image=cv2.resize(image,(256,256))
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    train_des.append(image)
    train_labels.append(label)
    
train_des=np.array(train_des)
train_labels=np.array(train_labels)

DATA_ROOT_TEST="./testAll_v2"
test_des=list()
img_list=os.listdir(DATA_ROOT_TEST)
img_list.sort()

for img in tqdm(img_list):
  image=cv2.imread(DATA_ROOT_TEST+'/'+img)
  image=cv2.resize(image,(256,256))
  #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  test_des.append(image)
```

## Dense SIFT 기술자  

```
DSIFT_STEP_SIZE=8
def extract_DenseSift_descriptors(img):
  #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  sift=cv2.xfeatures2d.SIFT_create()
  dsift_step_size=DSIFT_STEP_SIZE
  keypoints=[cv2.KeyPoint(x,y,dsift_step_size)
    for y in range(0,img.shape[0],dsift_step_size)
      for x in range(0,img.shape[1],dsift_step_size)]
  keypoints, descriptors=sift.compute(img,keypoints)
  return descriptors
```

## 히스토그램  
```
def input_vector_encoder(feature,codebook):
  code,_=vq.vq(feature,codebook)
  word_hist,bin_edges=np.histogram(code,bins=range(codebook.shape[0]+1),normed=True)
  return word_hist
  
```

## 피라미드   

```
def build_spatial_pyramid(image,descriptor,level):
  step_size=DSIFT_STEP_SIZE
  h=image.shape[0]//step_size
  w=image.shape[1]//step_size
  idx_crop=np.array(range(len(descriptor))).reshape(h,w)
  size=idx_crop.itemsize
  height,width=idx_crop.shape
  bh,bw=2**(3-level),2**(3-level)
  shape=(height//bh,width//bw,bh,bw)
  #print(shape)
  strides=size*np.array([width*bh,bw,width,1])
  #print(strides)
  #print(idx_crop)
  crops=np.lib.stride_tricks.as_strided(
      idx_crop,shape=shape,strides=strides
  )
  des_idxs=[col_block.flatten().tolist() for row_block in crops
            for col_block in row_block]
  pyramid=[]
  for idxs in des_idxs:
    pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
  return pyramid
  
def spatial_pyramid_matching(image,descriptor,codebook,level):
  pyramid=[]
  if level==0:
    pyramid+=build_spatial_pyramid(image,descriptor,level=0)
    code=[input_vector_encoder(crop,codebook) for crop in pyramid]
    return np.asarray(code).flatten()
  if level==1:
    pyramid+=build_spatial_pyramid(image,descriptor,level=0)
    pyramid+=build_spatial_pyramid(image,descriptor,level=1)
    code=[input_vector_encoder(crop,codebook) for crop in pyramid]
    code_level_0=0.5*np.asarray(code[0]).flatten()
    code_level_1=0.5*np.asarray(code[1:]).flatten()
    return np.concatenate((code_level_0,code_level_1))
  if level==2:
    pyramid+=build_spatial_pyramid(image,descriptor,level=0)
    pyramid+=build_spatial_pyramid(image,descriptor,level=1)
    pyramid+=build_spatial_pyramid(image,descriptor,level=2)
    code=[input_vector_encoder(crop,codebook) for crop in pyramid]
    code_level_0=0.25*np.asarray(code[0]).flatten()
    code_level_1=0.25*np.asarray(code[1:5]).flatten()
    code_level_2=0.5*np.asarray(code[5:]).flatten()
    return np.concatenate((code_level_0,code_level_1,code_level_2))
    
 ```
 
 ## 피라미드 매치 커널  
 
 ```
def histogramIntersection(M,N):
  m=M.shape[0]
  n=N.shape[0]

  result=np.zeros((m,n))
  for i in range(m):
    for j in range(n):
      temp=np.sum(np.minimum(M[i],N[j]))
      result[i][j]=temp
  return result
  
```

1. 8pixel * 8pixel의 DenseSIFT 기술자 추출  
```
from time import time

t0=time()

xtrain=[]
for img in train_des:
  x=extract_DenseSift_descriptors(img)
  xtrain.append(x)

x_train_dex=np.vstack((descriptor for descriptor in xtrain))

xtest=[]
for img in test_des:
  x=extract_DenseSift_descriptors(img)
  xtest.append(x)

x_test_dex=np.vstack((descriptor for descriptor in xtest))

print(time()-t0) 
> 456.36689496040344(7분~)
```

2. k-means 군집화로 코드북 만들기(K=400)  
```
t0=time()

codebooksize=400
seeding=kmc2.kmc2(np.array(x_train_dex).reshape(-1,128),codebooksize)
Kmeans=MiniBatchKMeans(codebooksize,init=seeding).fit(np.array(x_train_dex).reshape(-1,128))
codebook=Kmeans.cluster_centers_

print(time()-t0)
> 808.218183517456(13분~)
```

3. 피라미드 쌓기(Level=1)  
```
t0=time()

x_train=[spatial_pyramid_matching(train_des[i],xtrain[i],codebook,level=1) for i in range(len(train_des))]
x_test=[spatial_pyramid_matching(test_des[i],xtest[i],codebook, level=1) for i in range(len(test_des))] 

print(time()-t0)
> 308.98332715034485(5분~)

x_train=np.asarray(x_train)
x_test=np.asarray(x_test)
```

4. 피라미드매치 커널을 가진 SVM분류기로 학습  
```
t0=time()

from sklearn.svm import SVC
gramMatrix=histogramIntersection(x_train,x_train)

C_range=10.0**np.arange(-3,3)
gamma_range=10.0**np.arange(-3,3)
param_grid=dict(gamma=gamma_range.tolist(),C=C_range.tolist())

clf=GridSearchCV(SVC(kernel='precomputed'),param_grid,cv=5,n_jobs=-2)
clf.fit(gramMatrix,train_labels)

time()-t0
> 936.9802219867706(15분~)
```

+ 결과 예측  
```
predictMatrix=histogramIntersection(x_test,x_train)
label=clf.predict(predictMatrix)
```
+ 제출형식   
```
result=np.array(label).reshape(-1,1)
img_list=np.array(img_list).reshape(-1,1)
total_result=np.hstack([img_list,result])

df=pd.DataFrame(total_result,columns=["Id","Category"])
```

+ csv파일 변환후 확인, 제출  
```
df.to_csv('results-hrkim-v3.csv',index=False,header=True)
pd.read_csv('results-hrkim-v3.csv')

! kaggle competitions submit -c 2019-ml-finalproject -f results-hrkim-v3.csv -m "Final_Term_Project"
```

+ 성능: 0.55673~0.56855
  
  

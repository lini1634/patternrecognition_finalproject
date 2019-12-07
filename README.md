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

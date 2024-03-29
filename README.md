# patternrecognition_finalproject

+ Beyond bags of features spatial pyramid matching for recognizing natural scene categories, CVPR 2006

> 요약: 장면을 인식하여 분류하는데, 하나의 이미지를 분할하고 각 분할이미지마다 로컬피쳐로 히스토그램을 계산한다. Orderless bag of features의 확장판.

  + Bag of feature와의 차이점: 피쳐의 공간정보가 있다. 분할된 고정된 직사각형 윈도우를 이용한다. 
  + Multiresolution Histogram과 차이점: 로컬피쳐가 계산된 고정된 해상도에 공간 위치가 다름. 


### 단계
	1. 피쳐 추출: Dense SFIT로 얻어진 피쳐 정보를 k-means 군집화를 통해 시각적 vocabulary를 만든다.
	2. X이미지와 Y이미지의 히스토그램을 구하고 두 히스토그램의 최솟값으로 교차점을 찾는다.(-> 커널)
	3. 분할 이미지 셀의 넓이에 반비례하게 가중치를 둔다. (더 작은 셀일수록 공간정보를 많이 담고 있으므로 가중치를 높게 둠).
	4. 이미지의 히스토그램 분포를 SVM 분류기로 학습함. 

### 실험

|image color|sift descriptor|codebooksize|pyramid level|svm kernel|svm parameters|score|
|-----------|---------------|------------|-------------|----------|-------------|------|
|gray|dense 8 * 8|200|0|SVC|C: 0.001 ~ 1000, gamma: 0.001 ~ 1000|0.37647~0.38238|
|gray|dense 8 * 8|400|0|SVC|C: 0.001 ~ 1000, gamma: 0.001 ~ 1000|0.40248~0.41725|
|color|dense 8 * 8|200|2|histogram intersection|C: 0.001 ~ 1000, gamma: 0.001 ~ 1000|0.50591~0.50768|
|color|dense 8 * 8|200|2|LinearSVC|C: 0.000307~0.001|0.52482~0.52955|
|color|dense 8 * 8|400|1|histogram intersection|C: 0.001 ~ 1000, gamma: 0.001 ~ 1000|0.55673~0.56855|


### 참고

(1) https://github.com/CyrusChiu/Image-recognition  
(2) https://github.com/wihoho/Image-Recognition/blob/5dc8834dd204e36172815345f0abe5640a4a37ef/recognition/classification.py#L10  
(3) https://darkpgmr.tistory.com/125  
(4) https://github.com/TrungTVo/spatial-pyramid-matching-scene-recognition/blob/master/spatial_pyramid.ipynb


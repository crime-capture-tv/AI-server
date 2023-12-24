# Crime Capture TV : 무인점포 이상행동 탐지 시스템 (AI part)

메타버스 아카데미 9월 프로젝트

#### 🎥 시연 영상 보러가기(Click)
#### 📙 발표자료 보러가기(Click)

<img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/ace435a8-d4b3-4291-8627-dc59e052e55d" width="70%">

<br/>

# 팀원 소개 및 역할

**개발기간: 2023.09.04 ~ 2023.09.27**

#### AI
| AI | AI | AI | AI |
|:--:|:--:|:--:|:--:|
| 정민교 | 김종민 | 김찬영 | 최눈솔 |

#### Server
| server | server | server |
|:------:|:------:|:------:|
| 박태근 | 김나영 | 이주원 |

#### 기획
| 기획 | 기획 | 기획 |
|:---:|:---:|:---:|
| 김영식 | 이성균 | 이지수 |

<br/>

### AI 세부 역할 분담

<table>
    <tbody>
        <tr>
            <td><b>정민교</b></td>
            <td>데이터 전처리 및 VideoMAE model fine tuning</td>
        </tr>
        <tr>
            <td><b>김종민</b></td>
            <td>Yolo-v8을 이용한 humam detecting, 라즈베리 파이 CCTV 제작</td>
        </tr>
        <tr>
            <td><b>김찬영</b></td>
            <td>데이터 전처리 및 model serving</td>
        </tr>
        <tr>
            <td><b>최눈솔</b></td>
            <td>데이터 전처리</td>
        </tr>
    </tbody>
</table>

<br/>

# 융합 구조도

<img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/8331fda9-c20e-4ec2-b15c-c77a56ed916d" width="70%">

<br/>

# 프로젝트 소개

**cctv를 이용하여 무인점포 내에서 일어나는 이상행동을 감지하는 시스템을 제작**

<img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/f3c49525-2b6c-48aa-84a0-1ff56e1ddd0f" width="70%">

<img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/16740b64-0e7b-4a54-ae72-5bc4e7433f30" width="70%">

<br/>

# 주요 내용

### 1. Prepare data set

- #### Source data
  
  [ai hub](https://www.aihub.or.kr/)에 있는 [실내(편의점, 매장) 구매행동 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=71549)와
  [실내(편의점, 매장) 사람 이상행동 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71550)를 이용하였다.
  
- #### Preprocessing
  1) 원본 데이터에서 원하는 부분 편집

     원본 데이터는 1분짜리 긴 영상이었으므로 우리가 원하는 행동을 하는 구간만 찾아서 편집하였다. 사람의 행동 당 라벨을 부여하여 catch, put, insert, normal로 분류하였고 라벨당 약 300개의 2~5초짜리 영상을 준비하였다.
     
  3) 편집된 영상의 화질 조절

     원본 영상의 크기는 1920 x 1080 /3 fps 였으나 편집과정에서 1080 x 720 / 30 fps로 낮아졌다. 여기서 영상을 크롭하였을 때 사람의 모습이 꽉 차도록 640 x 480 으로 낮추었다.
     
  5) 편집된 영상 증강

     영상이 라벨당 약 300개 정도 준비하였지만 모델 학습시에 과적합이 발생하여 데이터를 증강 시켰다.
     영상 데이터 증강에 관한 부분은 ['3차원 의료 영상의 영역 분할을 위한 효율적인 데이터 보강 방법'](https://koreascience.kr/article/JAKO202109156813970.pdf)이라는 논문을 참고하였음.
	    
	    증강 방법
	    - Rotation  :  -10° ~ 10°
	    - Brightness  :  -50 ~ 50
	    - RGB  :  -30 ~ 30
       
  7) 사람이 있는 부분만 Crop

     모델이 input으로 받는 영상의 사이즈가 224 x 224 / 16 fps 의 영상이기 때문에 처음에는 기존 영상의 사이즈를 줄여서 넣는 방향으로 진행하였지만
     성능이 좋지 못하여 사람이 있는 부분을 찾아서 사람을 중심으로 224 x 224 크기로 자른 다음에 학습을 진행하였다. 사람을 찾기 위해서 [YOLOv8](https://docs.ultralytics.com/)을 사용.

- #### Result
    
    <table>
        <tbody>
            <tr>
                <td align="center"><img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/f64fe674-3c0b-4e4a-ab35-cb05ee36f3c0" width="200" height="200"></td>
                <td align="center"><img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/a35db24a-c3f2-466a-b477-0dd3ebd0289d" width="200" height="200"></td>
                <td align="center"><img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/36da3032-8ad7-4f84-9c33-61a5cfa5b1cf" width="200" height="200"></td>
                <td align="center"><img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/ff9656e6-c790-42f1-b09d-d6efbff133ba" width="200" height="200"></td>
            </tr>
            <tr>
                <td align="center"><b>Source</b></td>
                <td align="center"><b>편집</b></td>
                <td align="center"><b>증강</b></td>
                <td align="center"><b>Crop</b></td>
            </tr>
            <tr>
                <td align="center">1920x1080 / 3fps</td>
                <td align="center">640x480 / 30fps</td>
                <td align="center">x4 or x6</td>
                <td align="center">224x224 / 16fps</td>
            </tr>
        </tbody>
    </table>

### 2. Model train

- #### VideoMAE

  [VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae) 모델은 [Vision Transformer(ViT)](https://huggingface.co/docs/transformers/model_doc/vit)를 이용한 [ImageMAE(ViTMAE)](https://huggingface.co/docs/transformers/model_doc/vit_mae) 모델을 영상으로 확장한 모델이다. 원본 논문인 ['VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training'](https://arxiv.org/abs/2203.12602)과 [github](https://github.com/MCG-NJU/VideoMAE).
  
- #### ViViT

  [ViViT](https://huggingface.co/docs/transformers/model_doc/vivit) 모델은 [Vision Transformer(ViT)](https://huggingface.co/docs/transformers/model_doc/vit)에서 직접 확장된 영상 분류 모델이다. 논문[ 'ViViT: A Video Vision Transformer'](https://arxiv.org/abs/2103.15691)과 [github](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit).
	
	결론적으론 ViViT 모델이 VideoMAE 모델보다 성능이 좋지 못해서 테스트만 거치고 사용하지 않았음.

- #### Fine turning

  커스텀 데이터셋으로 VideoMAE를 fine turning하는 방법은 [이 노트](https://github.com/huggingface/notebooks/blob/main/examples/video_classification.ipynb)를 참고하였다. ViViT 모델 또한 같은 코드에서 VideoMAEImageProcessor와 VideoMAEForVideoClassification를 VivitImageProcessor와 VivitForVideoClassification로 교체하여 학습하였다.
  
- #### Train result

  처음 증강과 크롭을 하지 않은 데이터 셋을 이용하였을 때 테스트 데이터의 결과가 하나의 라벨로 편향되는 결과를 볼 수 있었다.
  
  이는 모델이 사람의 행동이 아니라 배경을 학습하여 이런 결과가 나오는 것으로 판단하여 크롭 작업을 진행하였다.
  
  또한 fine turning된 모델을 학습시키기 때문에 과적합이 일어나 데이터를 증강시켜서 모델을 학습 시켰다. 실제로 2배, 4배, 6배 증강된 데이터로 학습을 시킨 결과 증강을 할수록 정확도가 증가하고 편향이 줄어드는 결과를 얻었다.
  
- #### 아쉬운 점
   1) insert와 catch를 잘 구분하지 못했다. insert 데이터 중 물건을 넣는 과정만 들어가야 하는데 물건을 집어서 넣는 부분까지 포함된 영상이 많이 있었기 때문으로 보여짐.
   2) 사람이 돌면 put 으로 인식하는 경우가 있었다. put영상에서 사람이 물건을 놓고 뒤돌아 가는 장면이 포함되어 있어서 물건을 놓지 않아도 돌기만 하면 put으로 판단하는 것으로 보여짐.

- #### Result
    <table>
        <tbody>
            <tr>
                <td align="center"><img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/1b8a7b55-65e5-455b-8990-e681b4af893a" width="200" height="200"></td>
                <td align="center"><img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/e457d204-01ea-44f2-b098-13134b40f4f3" width="200" height="200"></td>
                <td align="center"><img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/8b295e33-0a69-49da-a2b8-538198eabd27" width="200" height="200"></td>
                <td align="center"><img src="https://github.com/crime-capture-tv/AI-server/assets/141614581/428ab861-1214-43bd-9aaa-d961de8569f6" width="200" height="200"></td>
            </tr>
            <tr>
                <td align="center"><b>catch</b></td>
                <td align="center"><b>put</b></td>
                <td align="center"><b>insert</b></td>
                <td align="center"><b>walking</b></td>
            </tr>
        </tbody>
    </table>

<br/>

### 3. Model train

# 기술 스택

### - 언어
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">

### - 주요 라이브러리
 <img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white"> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">

### - 개발 툴
<img src="https://img.shields.io/badge/VS code-2F80ED?style=for-the-badge&logo=VS code&logoColor=white"> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white">

### - 협업 툴
<img src="https://img.shields.io/badge/Github-181717?style=for-the-badge&logo=Github&logoColor=white"> <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white">

# 참고자료

### Papers

1. [Isola, P., Zhu, J., Zhou, T., & Efros, A. A. (2017, November 22). Image-to-Image Translation with Conditional Adversarial Networks. Arxiv. https://arxiv.org/abs/1611.07004](https://arxiv.org/abs/1611.07004)

### GitHub

1. [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
2. [pytorch-hed](https://github.com/sniklaus/pytorch-hed)
3. [FloorplanToBlender3d](https://github.com/grebtsew/FloorplanToBlender3d)

### Blog

1. [[논문실습] Pix2Pix](https://velog.io/@wilko97/%EB%85%BC%EB%AC%B8%EC%8B%A4%EC%8A%B5-Pix2Pix)
2. [딥러닝 기반 건축도면 생성 모델 개발](https://brunch.co.kr/@ddkddk35/10)

<br/>

<br/>

---

special thanks to [정민](https://github.com/min731)

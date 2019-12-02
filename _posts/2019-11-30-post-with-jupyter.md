---
title: "post-with-jupyter"
search: true
categories:
 - Notebook
tags:
 - Need_modify
last_modified_at: 2019-11-08 01:51
classes: wide
---

<style type="text/css">
.input_area div.highlighter-rouge {
  background-color: #263238  !important;
}

div.highlighter-rouge, figure.highlight {
  font-size: 0.675em  !important;
}

.output_stream, .output_data_text, .output_traceback_line {
  margin-left: 3% !important;
  border: none !important;
  border-radius: 4px !important;
  background-color: #fafafa !important;
  box-shadow: none !important;
  color: #000000  !important;
  font-size: 0.6em !important;
}

.output_stream:before, .output_data_text:before, .output_traceback_line:before{
  content: none !important;
}

table.dataframe {
    background-color: #fafafa;
    width: 100%;
    max-height: 240px;
    display: block;
    overflow: auto;
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 20px;
    text-align: center;
}
table.dataframe th {
  font-weight: bold;
  padding: 4px;
}
table.dataframe td {
  padding: 4px;
}
table.dataframe tr:hover {
  background: #b8d1f3;
}

</style>
# jupyter(ipynb)로 지킬 포스트 작성하기

이번 포스트는 jupyter notebook으로 jekyll 포스트를 작성하는 방법에 대해 다룹니다. 이 포스트는 주피터 노트북으로 작성되었습니다.

지킬의 장점은 마크다운 파일에 YAML Front Matter만 추가하면 사전 설정된 테마에 따라 멋있는 블로그 포스트를 만들어주기 때문에 별도의 프론트엔드 지식 없이도 손쉽게 블로그 포스트를 만들 수 있다는 것입니다. 하지만, ipynb 파일을 별도의 설정 없이 마크다운으로 변환해 포스트 폴더에 넣는 경우 대참사가 일어나는 것을 볼 수 있는데 이는 기본 변환기가 input과 output을 구분하는 class를 포함하고 있지 않아 전부 코드 블럭으로 표현되고, output의 이미지 경로가 쉽게 꼬인다는 문제입니다. (지킬은 현재 `_posts` 폴더에 png 파일이 포함되는 경우 정상적으로 이미지를 참조하지 못하는 에러가 있습니다. [관련 이슈](https://github.com/jekyll/jekyll/issues/5181))

ipynb 파일을 markdown이나 html로 변환할 때  주피터는 **nbconvert**를 사용합니다. nbconvert는 내부적으로 파이썬의 템플릿 엔진인 **jinja 템플릿**을 사용하는데, nbconvert 사용시에 기본으로 적용되는 템플릿을 수정하고 반영하여 ipynb 파일이 지금 이 글과 같이 블로그 스타일로 변환될 수 있도록 하려고 합니다. 목적에 따라 마크다운 파일이 아닌 html 파일로 변환을 할 수도 있으나, 주로 참조한 [Chris Holdgraf](https://github.com/choldgraf)의 포스트가 마크다운으로 변환하는 방법을 알려주어 이 포스트도 마크다운으로 변환하는 방식을 포스팅했습니다. 목적에 따라 html로 변환해서 포스팅을 하는 것도 물론 가능합니다.

### nbconvert 설치

nbconvert는 `pip install nbconvert`, 콘다 환경인 경우 `conda install nbconvert`를 통해 간단하게 설치할 수 있습니다. 만약 주피터 파일을 html이나 md가 아닌 pdf, tex 등으로 변환하는 등 nbconvert의 모든 기능을 이용하기 위해서는  pandoc 과 XeLaTeX를 추가로 설치해야 합니다. 해당 내용은 [nbconvert 공식문서](https://nbconvert.readthedocs.io/en/latest/install.html)에서 확인하실 수 있습니다.

### nbconvert의 markdown 템플릿 변환하기

nbconvert가 노트북 파일을 마크다운으로 변환할 때 사용하는 템플릿을 커스터마이징하고 싶은 경우 Jinja 템플릿에 대한 이해가 조금은 필요합니다. [Jinja 공식문서](https://jinja.palletsprojects.com/en/2.10.x/templates/)를 읽어볼 수 있으면 좋겠지만, 목적에 비해 그 내용이 방대하므로 [다음의 글](https://www.datacamp.com/community/tutorials/jinja2-custom-export-templates-jupyter)을 참조하시면 간략하게 작동원리를 이해하실 수 있습니다.

nbconvert는 html로 변환하는 경우에는 `basic.tpl`, 마크다운으로 변환하는 경우에는 `markdown.tpl`을 기본값으로 사용하여 렌더링을 합니다. 

`markdown.tpl` to `jekyll.tpl`

### 쉘 함수로 경로 설정 자동화하기

```sh
function new_post {
    #change these 3 lines to match your specific setup
    POST_PATH="/Users/jungsooyun/Desktop/dev/frontend/jungsooyun.github.io/_posts"
    IMG_PATH="/Users/jungsooyun/Desktop/dev/frontend/jungsooyun.github.io/images"

    FILE_NAME="$1"
    CURR_DIR=`pwd`
    FILE_BASE=`basename $FILE_NAME .ipynb`

    POST_NAME="${FILE_BASE}.md"
    IMG_NAME="${FILE_BASE}_files"

    POST_DATE_NAME=`date "+%Y-%m-%d-"`${POST_NAME}

    # convert the notebook
    jupyter nbconvert --to markdown --template jekyll.tpl $FILE_NAME

    # change image paths
    sed -i .bak "s:\[png\](:[png](/images//images/:" $POST_NAME

    # move everything to blog area
    mv  $POST_NAME "${POST_PATH}/${POST_DATE_NAME}"
    mv  $IMG_NAME "${IMG_PATH}/"
```

그런데, 마크다운 같은 경우는 이미지를 별도로 구분한 다음 추가해주어야 한다.(https://predictablynoisy.com/jekyll-markdown-nbconvert) 구현한 템플릿을 이용하면 자동으로 폴더 생성, 그런데 _posts 폴더에 png 파일이 들어가는 경우 에러가 발생하는 문제가 생긴다.(http://www.leeclemmer.com/2017/07/04/how-to-publish-jupyter-notebooks-to-your-jekyll-static-website.html) 이를 해결하기 위해 쉘 함수를 정의해준다 (자동화 짱짱)(https://blomadam.github.io/tutorials/2017/04/09/ipynb-to-Jekyll-Post-tools.html)

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

<div class="input_area" markdown="1">

```python
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

%matplotlib inline
```

</div>

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

<div class="input_area" markdown="1">

```python
meta = load_iris()
print(meta.keys())
```

</div>

{:.output_stream}

```
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

```

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

<div class="input_area" markdown="1">

```python
data = pd.DataFrame(meta['data'], columns=meta['feature_names'])
data.head()
```

</div>




<div markdown="0">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>
</div>



<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

<div class="input_area" markdown="1">

```python
sns.set_style('whitegrid')

sns.scatterplot(data['sepal length (cm)'], data['petal length (cm)'])
```

</div>




{:.output_data_text}

```
<matplotlib.axes._subplots.AxesSubplot at 0x1a16ecd748>
```




![png](/images/post-with-jupyter_files/post-with-jupyter_5_1.png)


<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

<div class="input_area" markdown="1">

```python
Error
```

</div>


{:.output_traceback_line}

`---------------------------------------------------------------------------`


{:.output_traceback_line}

`NameError                                 Traceback (most recent call last)`


{:.output_traceback_line}

`<ipython-input-6-24f271892eea> in <module>
----> 1 Error
`


{:.output_traceback_line}

`NameError: name 'Error' is not defined`



<div class="prompt input_prompt">
In&nbsp;[None]:
</div>

<div class="input_area" markdown="1">

```python

```

</div>

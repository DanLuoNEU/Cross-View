# Cross-View
Code for *Muti-view Action Recognition via View-invariant Dynamic Representations*

Environment Install

'bash
pip install virtualenv
virtualenv /path/to/env-cross-view
source /path/to/env-cross-view/bin/activate
pip install -r requirements.txt
'

## Train
trainClassifier_Multi_CS.py \
&nbsp; -train with multi sampler for Cross-Subject


trainClassifier_Multi.py \
&nbsp; -train with multi sampler for Cross-View

## Test 
testClassifier_Multi_CS.py \
&nbsp; -test with multi sampler for Cross-Subject


testClassifier_Multi.py \
&nbsp; -test with multi sampler for Cross-View

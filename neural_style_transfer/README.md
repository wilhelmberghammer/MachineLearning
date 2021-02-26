# Neural Style Transfer

I tried to follow the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf) but I did some things differently. 

For example I didn't use their, more complicated loss function, I simply used MSE.

I also didn't use noise as input, I used the content.

This might be the reason for some differences. I didn't notice any differences when changing the alpha and beta values. The biggest change happens when changing the learning rate. A higher learning rate leads to more significant changens to the original content.

<br>

### Setup
You will need a virtualenv with `python>=3.8`.

```bash
pip install poetry
poetry install
```

### To run style transfer
```bash
poetry run python neural_style_transfer/nst.py
```

## Result Example
**Example 1:**

![content alisha](https://github.com/wilhelmberghammer/MachineLearning/blob/main/neural_style_transfer/readme_results/result_1.png)

**Example 2:**

![style dog](https://github.com/wilhelmberghammer/MachineLearning/blob/main/neural_style_transfer/readme_results/result_2.png)

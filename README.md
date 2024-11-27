# FCN

Link to Paper:

***“Fully Convolutional Networks for Semantic Segmentation” - 2015***

[https://arxiv.org/pdf/1411.4038](https://arxiv.org/pdf/1411.4038)

---

# 1. Introduction

- **Pioneering Fully Convolutional Approach**:
    
    Introduced the concept of Fully Convolutional Networks (FCNs), which replace fully connected layers with convolutional layers, enabling end-to-end dense predictions for pixel-level tasks like semantic segmentation.
    
- **The first work to train FCNs end-to-end**:
    - (1) for **pixelwise prediction** and
    - (2) from **supervised pre-training**
- **Skip Connections for Multi-Scale Fusion**:
    
    Introduced **skip connections** to combine high-level semantic information from deeper layers with low-level spatial details from shallower layers, progressively refining predictions (e.g., FCN-32s, FCN-16s, FCN-8s).
    
<br><br><br><br>
# 2. Architecture


![Alt text](https://github.com/h383kim/FCN/blob/main/images/image1.png)

<br><br><br>
> **Transposed Convolution**
> 

---

Also called as:

- Deconvolution
- Transposed Convolution
- Fractional Stride Convolution

A transposed convolution "reverses" the spatial reduction caused by a standard convolution. While standard convolution downsamples an input, transposed convolution upsamples it, enabling reconstruction of higher-resolution outputs.

Take a look at example:

![Alt text](https://github.com/h383kim/FCN/blob/main/images/image4.png)
<br>
[img src](https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967)

$$
\text{Input Feature} = \begin{bmatrix}
55 & 52 \\
57 & 50 \\
\end{bmatrix}
$$

$$
\text{Kernel} = \begin{bmatrix}
1 & 2  & 1\\
2 & 2  & 2\\
1 & 1 & 2\end{bmatrix}
$$

$$
\begin{align*}
\text{Output} = \begin{bmatrix}
55 & (110 + 52)  & (55 + 104) & 52\\
(110 + 57) & (55 + 104 + 114 + 50) & (110 + 52 + 57 + 100) & (104 + 50)\\
(55 + 114) & (55 + 52 + 57 + 100) & (110 + 52 + 114 + 50) & (104 + 100)\\
57 & (57 + 50) & (114 + 50) & 100
\end{bmatrix}
\end{align*}
$$

<br><br><br>
> **General Formula for Transposed Convolution Output Size**
> 
---
The formula to compute the output size of a transposed convolution is:

`output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding`


- **`input_size`**: Spatial size (height or width) of the input tensor.
- **`stride`**: Upsampling factor.
- **`padding`**: Amount of zero-padding added to both sides of the input.
- **`kernel_size`**: Size of the transposed convolution kernel.
- **`output_padding`**: Additional size added to one side of the output (usually set to 0 unless needed for specific cases).

<br><br><br>

<aside>

> **Other Upsampling Methods**
>
---

- Bed of Nails

![Alt text](https://github.com/h383kim/FCN/blob/main/images/image7.png)

- Nearest Neighbour Interpolation

![Alt text](https://github.com/h383kim/FCN/blob/main/images/image3.png)

- Max Unpooing

![Alt text](https://github.com/h383kim/FCN/blob/main/images/image9.png)
[img src](https://www.superannotate.com/blog/guide-to-semantic-segmentation)
</aside>

<aside>

<br><br><br>
**Difference Between other upsampling vs. Transposed Convolution**

---

In short, Transposed Convolutions have learnable parameters!

|  | **Upsampling(Interpolation, … and etc)** | **Transposed Convolution** |
| --- | --- | --- |
| **Learnable** **Parameters** | None | Kernel weights and biases |
| **Complexity** | Computationally simple | Computationally intensive |
| **Customization** | Not trainable (Fixed rules) | Trainable via backpropagation |
| **Flexibility** | Fixed scaling behavior | Can learn adaptive upsampling patterns |

</aside>

<br><br><br>

> **Shift-and-Stitch**

---

**This method is not used in the model**, but is worth mentioning as the paper does.

![Alt text](https://github.com/h383kim/FCN/blob/main/images/image5.png)

- **Shift the input image** slightly multiple times.
- Process each shifted version of the input through the network.
- **Combine (stitch) the outputs** to form a dense prediction.

**Shifting the Input**:

- If the network downsamples by a factor $`f`$, the input is shifted in $`f×f`$ different ways.
- For example, if $`f=2`$, you shift the input:
    - By $`(0,0)`$: No shift.
    - By $`(0,1)`$: Shift right by $`1`$ pixel.
    - By $`(1,0)`$: Shift down by $`1`$ pixel.
    - By $`(1,1)`$: Shift right and down by $`1`$ pixel.

**Run the Network**:

- Each shifted version of the input is processed by the same network.
- The outputs are coarse predictions at a downsampled resolution.

**Stitch the Outputs**:

- Each output prediction corresponds to specific pixels in the original input image (depending on the shift).
- By combining these predictions, you can "fill in the gaps" and produce a dense prediction that matches the resolution of the input.

<aside>

<br>

In computer vision tasks (e.g., semantic segmentation), we want a **prediction for every pixel** in the input image. This is called a **dense prediction**.

</aside>

<br><br><br>
>**Skip Net**
> 

---

The **FCN-8s, FCN-16s, and FCN-32s** architectures represent progressively refined variants of Fully Convolutional Networks (FCNs) that combine coarse, high-level semantic information with finer, lower-level spatial details to improve the accuracy of dense predictions in semantic segmentation.

![Alt text](https://github.com/h383kim/FCN/blob/main/images/image6.png)

### 1. **FCN-32s** (Coarsest Prediction)

- Predictions are made directly from the **conv7** layer (converted fully connected layer `fc7`).
- The output is at a stride of 32 (downsampled by a factor of 32 from the input image resolution).
- These coarse predictions are upsampled in a single step back to the original image resolution.
- While retaining high-level semantics, the spatial resolution is limited, resulting in coarse outputs.

---

### 2. **FCN-16s** (Finer Predictions by Adding Skip Connections)

- Adds predictions from the **pool4** layer (stride 16), which retains more spatial details.
- The **pool4 predictions** are combined with the predictions from **conv7**:
    1. Pool4 predictions are passed through a $`1×1`$ convolution to produce class-specific predictions.
    2. Conv7 predictions are upsampled by a factor of 2 (stride reduced from 32 to 16) using bilinear interpolation.
    3. The two predictions (pool4 and upsampled conv7) are summed together to fuse finer spatial details with high-level semantics.
- The fused predictions are upsampled back to the original image resolution.
- This approach improves the quality of segmentation, especially for finer structures, and increases mean Intersection over Union (mean IU) significantly.

---

### 3. **FCN-8s** (Finest Prediction)

- Further refines predictions by incorporating information from the **pool3** layer (stride 8), which has even more spatial detail than pool4.
- The process:
    1. Predictions from pool3 are combined with the fused predictions from **pool4 and conv7**.
    2. Pool3 predictions are passed through a $`1×1`$ convolution.
    3. The fused pool4 + conv7 predictions are upsampled by a factor of 2 to match the stride 8 resolution of pool3.
    4. All predictions (pool3, pool4, and conv7) are summed and upsampled back to the original image resolution.
- This hierarchical fusion of layers provides the highest precision, allowing the network to retain fine spatial details while leveraging high-level semantic information.

![Alt text](https://github.com/h383kim/FCN/blob/main/images/image8.png)

<br><br><br><br>

# 3. Experiments
Data Augmentation boosted around 2% on validation pixel accuracy.

## Metrics

| Metric           | Value    |
|------------------|----------|
| Pixel Accuracy   | 87.28%   |
<br>

## Sample Inference
![Alt text](https://github.com/h383kim/FCN/blob/main/results/segmentation_epoch_50.png)

![Alt text](https://github.com/h383kim/FCN/blob/main/images/image2.png)


CamVid(Cambridge-driving Labeled Video Database) Dataset:

https://www.kaggle.com/datasets/carlolepelaars/camvid


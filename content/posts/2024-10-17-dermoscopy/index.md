---
title: 'Dermoscopy images for neural networks' 
date: 2024-10-17T23:18:00+03:30
draft: false
tags: ["dermoscopy", "trichoscopy", "preprocessing", "shades of grey algorithm"]
description: "What are challenges specific to dermoscopy images for neural networks and how can we resolve them?"
canonicalURL: "https://advafaeian.github.io/2024-10-17-dermoscopy/"
cover:
    image: "cover.jpg" # image path/url
    alt: "Skin" # alt text
    caption: "Skin. Photo by Nsey Benajah on Unsplash"
    relative: true  
---

Here, I aim to summarize some of the most significant challenges related to training neural networks specifically on dermoscopy images and review a few conventional solutions for each.


## Lighting
As lighting conditions may vary across different settings, it is best practice to minimize the impact of lighting as accurately as possible. One method, called the **Gray World algorithm**, assumes that the average reflectance in a scene is gray and adjusts the image colors so that the overall average becomes neutral gray.
```python
import numpy as np

def gray_world_algorithm(image: np.ndarray):
    # Convert the image to a floating point representation for more precision
    img = image.astype(np.float32)
    
    # Calculate the mean values for each channel (R, G, B)
    mean_r = np.mean(img[:, :, 2])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 0])

    # Calculate the mean of all channels combined
    mean_gray = (mean_r + mean_g + mean_b) / 3

    # Scale each channel so that their means become equal to the mean_gray
    img[:, :, 2] = img[:, :, 2] * (mean_gray / mean_r)
    img[:, :, 1] = img[:, :, 1] * (mean_gray / mean_g)
    img[:, :, 0] = img[:, :, 0] * (mean_gray / mean_b)

    # Clip the values to ensure they remain in the valid range (0-255)
    img = np.clip(img, 0, 255)

    # Convert the image back to an unsigned 8-bit integer representation
    img = img.astype(np.uint8)
    
    return img
```

Another method, called the **Max-RGB algorithm**, adjusts the colors so that the brightest parts of the image appear white. 
```python
import numpy as np

def max_rgb_algorithm(image: np.ndarray):
    # Convert the image to a floating point representation for more precision
    img = image.astype(np.float32)
    
    # Find the maximum values for each channel across the entire image
    max_r = np.max(img[:, :, 2])
    max_g = np.max(img[:, :, 1])
    max_b = np.max(img[:, :, 0])
    
    # Scale each channel by its maximum to match the largest value among all channels
    img[:, :, 2] = img[:, :, 2] * (255.0 / max_r)
    img[:, :, 1] = img[:, :, 1] * (255.0 / max_g)
    img[:, :, 0] = img[:, :, 0] * (255.0 / max_b)
    
    # Clip the values to ensure they remain in the valid range (0-255)
    img = np.clip(img, 0, 255)
    
    # Convert the image back to an unsigned 8-bit integer representation
    img = img.astype(np.uint8)
    
    return img
```
These two methods represent different approaches to color correction: the Gray World algorithm is a mean-based method, while the Max-RGB algorithm is a maxima-based method. 


Another, more popular method that generalizes the previous ones is called the **Shades of Gray algorithm**. Before delving into the method, let’s introduce some notations. Let {{< rawhtml >}}$\mathbf{X} = [x_1, \ldots, x_n]^T${{< /rawhtml >}} be a vector in {{< rawhtml >}}$\mathbb{R}^N${{< /rawhtml >}}. For every {{< rawhtml >}}$p \geq 1${{< /rawhtml >}}, the quantity

{{< rawhtml >}}
$$


\| \mathbf{X} \|_p = \left( \sum_{i=1}^{N} |x_i|^p \right)^{1/p}


$$
{{< /rawhtml >}}

is called the [{{< rawhtml >}}$p${{< /rawhtml >}}-norm or {{< rawhtml >}}$L^p${{< /rawhtml >}}-norm of {{< rawhtml >}}$\mathbf{X}${{< /rawhtml >}}](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions). If we define the function $\mu_p(\mathbf{X}) = \frac{\text{L}^p\text{-norm of } \mathbf{X}}{N^{1/p}}$, then for $p = 1$, $\mu_1(\mathbf{X})$ equals the mean of the elements of $\mathbf{X}$. On the other hand, for $p = \infty$, $\mu_\infty(\mathbf{X}) = \max \{ |x_1|, |x_2|, \ldots, |x_n| \} = m$. This holds because:


{{< rawhtml >}}
$$
\begin{align*}
n \cdot m^p &\geq \sum_{i=1}^n |x_i|^p, \\
n^{1/p} \cdot m &\geq \left(\sum_{i=1}^N |x_i|^p\right)^{1/p} = \|\mathbf{X}\|_p, \\
m &\geq \lim_{p \to \infty} \|\mathbf{X}\|_p \tag{$\lim_{p \to \infty}n^{1/p} = 1$}
\end{align*}

$$
{{< /rawhtml >}}


Also,
{{< rawhtml >}}
$$
\begin{align*}
\|x\|_p &\geq \max\{|x_1|, \ldots |x_n|\} = m, \qquad &\text{(}\|x\|_p \geq |x_i| \text{ for all } i \text{)} \\
\lim_{p \to \infty} \|x\|_p &\geq m \qquad &\text{(the inequality holds for all } p \text{)} 
\end{align*}

$$
{{< /rawhtml >}}


Combining the inequalities:
{{< rawhtml >}}
$$
m \geq \lim_{p \to \infty} \|x\|_p \geq m

$$
{{< /rawhtml >}}

Therefore:
{{< rawhtml >}}
$$
\mu_\infty(\mathbf{X}) = \lim_{p \to \infty} \|\mathbf{X}\|_p = m = \max\{|x_1|, |x_2|, \ldots, |x_n|\} = \|x\|_\infty
$$
{{< /rawhtml >}}

In short, the Gray World algorithm uses the {{< rawhtml >}}${\displaystyle L^{1}}${{< /rawhtml >}}-norm, while the Max-RGB algorithm uses the {{< rawhtml >}}${\displaystyle L^{\infty}}${{< /rawhtml >}}-norm of {{< rawhtml >}}$\mathbf{X}${{< /rawhtml >}}. [Finlayson et al.](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008) demonstrated that between these two extremes, {{< rawhtml >}}$p = 6${{< /rawhtml >}} achieves the best performance in terms of angular error:

{{< rawhtml >}}
$$
\frac{\cos^{-1}\left(q_l \cdot q_e\right)}{\lvert q_l \rvert \cdot \lvert q_e \rvert}
$$
{{< /rawhtml >}}

where {{< rawhtml >}}$q_l = \begin{bmatrix} R_l & G_l & B_l \end{bmatrix}^T${{< /rawhtml >}} represents the measured light, and {{< rawhtml >}}$q_e = \begin{bmatrix} R_e & G_e & B_e \end{bmatrix}^T${{< /rawhtml >}} represents the estimated illuminant calculated through the algorithm. This equation uses the dot product to evaluate the alignment between the vectors of the estimated illuminant and the measured light in {{< rawhtml >}}$\mathbb{R}^3${{< /rawhtml >}}.



Therefore, the Shades of Gray method uses the {{< rawhtml >}}${\displaystyle L^{6}}${{< /rawhtml >}}-norm of {{< rawhtml >}}$\mathbf{X}${{< /rawhtml >}}.

```python
import numpy as np

def shades_of_grey_algorithm(image: np.ndarray, p=6):

    # Convert the image to a floating point representation for more precision
    img = image.astype(np.float32)
    
    # Calculate the p-norm for each channel
    r_norm = np.power(np.mean(np.power(img[:, :, 2], p)), 1/p)
    g_norm = np.power(np.mean(np.power(img[:, :, 1], p)), 1/p)
    b_norm = np.power(np.mean(np.power(img[:, :, 0], p)), 1/p)
    
    # Calculate the mean of the norms
    mean_norm = (r_norm + g_norm + b_norm) / 3

    # Scale each channel to normalize the norms
    img[:, :, 2] = img[:, :, 2] * (mean_norm / r_norm)
    img[:, :, 1] = img[:, :, 1] * (mean_norm / g_norm)
    img[:, :, 0] = img[:, :, 0] * (mean_norm / b_norm)
    
    # Clip the values to ensure they remain in the valid range (0-255)
    img = np.clip(img, 0, 255)
    
    # Convert the image back to an unsigned 8-bit integer representation
    img = img.astype(np.uint8)
    
    return img
```


&nbsp;

## Annotation and Labeling
The gold standard for diagnosing many skin diseases includes laboratory data, direct immunofluorescence (DIF), indirect immunofluorescence (IIF), and histopathology. Therefore, accurate labeling is not always possible using only dermoscopy images. Even for segmentation tasks, a confirmed diagnosis should be available. Additionally, experts may disagree on the diagnosis based on histopathology. A costly solution is to have multiple dermatopathologists review the histopathological slides and decide based on a majority opinion. \
Personally, I find this to be the most challenging aspect, particularly when dermoscopy images are obtained retrospectively, which is often the case—especially for rare diagnoses that even in a tertiary center, require years of data collection to gather enough images. In many instances, histopathological results are not available at the same center. For diagnoses such as melanoma, histopathological examinations can be inconclusive, leading to the excision of lesions due to the patient's overall high risk, which is determined by patient and lesion characteristics, including age. As a result, clinical decisions are made based on a high level of suspicion rather than a definitive diagnosis, and no ground truth diagnosis is available. One of our projects was completely canceled due to extremely difficult access to the histopathological reports, despite having hundreds of high-quality images available.


&nbsp;

## Cross-Dataset Generalization
A model trained on a specific dataset may perform poorly when applied to a dataset from a different population, with variations in skin tone, hair color, or imaging devices. To address this, multi-center datasets or those specifically designed to include diverse skin tones and hair colors, such as the [DDI dataset](https://ddi-dataset.github.io) by Daneshjou et al., can be used, if available. In this context, the importance of regularization is emphasized, as reducing overfitting improves the model's generalizability, allowing it to perform well across a broader range of skin tones and hair colors.

&nbsp;

## Data Imbalance
There are many rare diseases in dermatology. In trichoscopy, for example, a prepared dataset may contain a large number of alopecia areata images but a low number of folliculitis decalvans images. This imbalance can hurt training efficiency because, with conventional loss function designs, classes with fewer images have a less pronounced effect on the overall gradient, leading to a model that may not prioritize accurate diagnosis for these underrepresented classes. Simply discarding some of the more prevalent images is not ideal, as it means not utilizing the full dataset. One solution is to use data augmentation for classes with fewer images, though care should be taken not to overdo it, as this might distort key features that are crucial for accurate diagnosis. Another solution is to use a weighted loss function, which compensates for the lower number of images in certain diagnoses by assigning higher importance to them during training.

If you believe something should be added or if you notice any mistakes in this post, please don’t hesitate to reach out to me at [ad dot vafaeian at gmail dot com]. I will address any issues promptly. :)

## References
- Graham D. Finlayson, Elisabetta Trezzi. "Shades of Gray and Colour Constancy." IS&T/SID Twelfth Color Imaging Conference. (2004). 
- Roxana Daneshjou et al. ,Disparities in dermatology AI performance on a diverse, curated clinical image set.Sci. Adv.8,eabq6147(2022).
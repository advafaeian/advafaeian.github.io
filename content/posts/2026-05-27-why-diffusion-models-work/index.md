---
title: 'Why Diffusion Models Work' 
date: 2026-05-27
draft: false
tags: ["diffusion models", "generative models", "differential equations", "brownian motion"]
description: "A theoretical review of diffusion models, covering DDPMs, score-based generative modeling, stochastic differential equations, training objectives, and sampling algorithms."
canonicalURL: "https://advafaeian.github.io/2026-05-27-why-diffusion-models-work/"
cover:
    image: "images/cover.jpg" # image path/url
    alt: "Gas molecules" # alt text
    caption: "Gas molecules. Photo by Robert Zunikoff on Unsplash"
    relative: true  
math: true
---

This post was inspired by the Stanford course on diffusion models, [CME296](https://cme296.stanford.edu/), by Afshine Amidi and Shervine Amidi, and is intended to provide a detailed review of the theoretical background of diffusion models.

We review the derivation of the loss functions and the underlying principles behind the current training and inference algorithms. It requires a basic familiarity with diffusion models to begin with.

## PARADIGM 1: DDPM

### Basic equations

{{< rawhtml >}}
$$
x_{t+1} = \sqrt{1-\beta_t}\,x_t + \sqrt{\beta_t}\,\epsilon \qquad \text{with } \beta_t \text{ noise schedule}
$$
{{< /rawhtml >}}

which can be rewritten as:

{{< rawhtml >}}
$$


x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon 
\qquad
\text{with } \epsilon \sim \mathcal{N}(0,1),\quad
\alpha_t = 1-\beta_t,\quad
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s

$$
{{< /rawhtml >}}

---

### Deriving ELBO
The objective of our model is to *correctly* reconstruct the original image, {{< rawhtml >}}$x_0${{< /rawhtml >}}. Therefore, our goal is to maximize the expected value of {{< rawhtml >}}$\log p_\theta(x_0)${{< /rawhtml >}} over all possible {{< rawhtml >}}$x_0${{< /rawhtml >}}. However, computing {{< rawhtml >}}$p_\theta(x_0)${{< /rawhtml >}} requires marginalizing over all latent variables:

{{< rawhtml >}}
$$

p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}
=
\int p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)\,dx_{1:T}.

$$
{{< /rawhtml >}}

Directly optimizing this objective is intractable because it involves a high-dimensional integral over all latent variables, with the logarithm applied outside the integral. Therefore, instead of maximizing the exact log-likelihood, we derive and maximize a variational lower bound:

{{< rawhtml >}}
$$

p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}
=
\int q(x_{1:T}\mid x_0)\,
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\,dx_{1:T}.

$$
{{< /rawhtml >}}

This is an expectation over {{< rawhtml >}}$q(x_{1:T}\mid x_0)${{< /rawhtml >}}:

{{< rawhtml >}}
$$

p_\theta(x_0)
=
\mathbb{E}_{x_{1:T} \sim q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].

$$
{{< /rawhtml >}}

Take the logarithm:

{{< rawhtml >}}
$$

\log p_\theta(x_0)
=
\log
\mathbb{E}_{x_{1:T} \sim q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].

$$
{{< /rawhtml >}}

Now, apply Jensen's inequality:

{{< rawhtml >}}
$$

\log
\mathbb{E}_{x_{1:T} \sim q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right]
\ge
\mathbb{E}_{x_{1:T} \sim q(x_{1:T}\mid x_0)}
\left[
\log
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].

$$
{{< /rawhtml >}}

Therefore:

{{< rawhtml >}}
$$

\log p_\theta(x_0)
\ge
\mathbb{E}_{x_{1:T} \sim q(x_{1:T}\mid x_0)}
\left[
\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].

$$
{{< /rawhtml >}}


Take the expectation over {{< rawhtml >}}$x_0 \sim q(x_0)${{< /rawhtml >}}:

{{< rawhtml >}}
$$

\mathbb{E}_{x_0 \sim q(x_0)}\big[\log p_\theta(x_0)\big]
\;\ge\;
\boxed{
\mathbb{E}_{x_{0:T} \sim q(x_{0:T})}
\left[
\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right]
}
=
\text{ELBO}

$$
{{< /rawhtml >}}

---
### Writing ELBO as a tractable KL divergence

That ELBO could be rewrritten as:

{{< rawhtml >}}
$$

\mathcal{L}
=
\mathbb{E}_{x_{0:T}\sim q(x_{0:T})}
\left[
\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\mathbb{E}_{x_{0:T}\sim q(x_{0:T})}
\left[
\log
\frac{
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
}{
q(x_{1:T}\mid x_0)
}
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\mathbb{E}_{x_{0:T}\sim q(x_{0:T})}
\left[
\log
\frac{
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
}{
\prod_{t=1}^{T}q(x_t\mid x_{t-1},x_0)
}
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\mathbb{E}_{x_{0:T}\sim q(x_{0:T})}
\left[
\log
\frac{
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
}{
\prod_{t=1}^{T}q(x_t\mid x_{t-1})
}
\right] \text{(Markov chain property)}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\mathbb{E}_{x_{0:T}\sim q(x_{0:T})}
\left[
\log p(x_T)
+\sum_{t=1}^{T}\log p_\theta(x_{t-1}\mid x_t)
-\sum_{t=1}^{T}\log q(x_t\mid x_{t-1}, x_0)
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\mathbb{E}_{x_{0:T}\sim q(x_{0:T})}
\left[
\log p(x_T) + \log p_\theta(x_0\mid x_1) - \log q(x_1\mid x_0) + \sum_{t=2}^{T}\log\frac{p_\theta(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})}\right
]  \text{(Split off $t=1$)}

$$
{{< /rawhtml >}}


**Bayes' rule** {{< rawhtml >}}$q(x_t \mid x_{t-1}) = q(x_{t-1}\mid x_t, x_0)\,\dfrac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}${{< /rawhtml >}}:

{{< rawhtml >}}
$$

= \mathbb{E}_{x_{0:T}\sim q(x_{0:T})}\left[\log p(x_T) + \log p_\theta(x_0\mid x_1) - \log q(x_1\mid x_0) + \sum_{t=2}^{T}\log\frac{p_\theta(x_{t-1}\mid x_t)}{q(x_{t-1}\mid x_t,x_0)} + \underbrace{\sum_{t=2}^{T}\log\frac{q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}}_{\log q(x_1\mid x_0)\,-\,\log q(x_T\mid x_0)}\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\mathbb{E}_{x_{0:T}\sim q(x_{0:T})}
\left[
\log p_\theta(x_0\mid x_1) + \log\frac{p(x_T)}{q(x_T\mid x_0)} + \sum_{t=2}^{T}\log\frac{p_\theta(x_{t-1}\mid x_t)}{q(x_{t-1}\mid x_t,x_0)}\right] \text{($\log q(x_1|x_0)$ got cancelled)}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\boxed{
-\sum_{t=2}^{T}
\mathrm{KL}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)
}
+\text{extra terms}

$$
{{< /rawhtml >}}


--- 
### Why ELBO is tractable?

Now, we show that ELBO is tractable.

#### $q(x_{t-1}\mid x_t,x_0)$: 

{{< rawhtml >}}
$$

\boxed{
q(x_{t-1}\mid x_t,x_0)=\mathcal N(x_{t-1};\tilde\mu_t,\tilde\beta_t I)
}

$$
{{< /rawhtml >}}

with

{{< rawhtml >}}
$$

\tilde\mu_t
=

\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}},\epsilon_t
\right)

$$
{{< /rawhtml >}}

And
{{< rawhtml >}}
$$

\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t

$$
{{< /rawhtml >}}

where:

* {{< rawhtml >}}$x_t${{< /rawhtml >}}: the noisy image at timestep {{< rawhtml >}}$t${{< /rawhtml >}}
* {{< rawhtml >}}$\epsilon_t${{< /rawhtml >}}: the sampled Gaussian noise used to transform the clean image into a noisy image at timestep {{< rawhtml >}}$t${{< /rawhtml >}}


Start from:

{{< rawhtml >}}
$$

q(x_t \mid x_{t-1})=\mathcal N\left(x_t;\sqrt{\alpha_t},x_{t-1},,\beta_t I\right),
\qquad \alpha_t=1-\beta_t

$$
{{< /rawhtml >}}

and:

{{< rawhtml >}}
$$

q(x_t\mid x_0)=\mathcal N\left(x_t;\sqrt{\bar\alpha_t},x_0,,(1-\bar\alpha_t)I\right),
\qquad
\bar\alpha_t=\prod_{s=1}^t \alpha_s.

$$
{{< /rawhtml >}}

By Bayes:

{{< rawhtml >}}
$$

q(x_{t-1}\mid x_t,x_0)
=

\frac{q(x_t\mid x_{t-1},x_0),q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}.

$$
{{< /rawhtml >}}

And because the forward process is Markov,

{{< rawhtml >}}
$$

q(x_t\mid x_{t-1},x_0)=q(x_t\mid x_{t-1}),

$$
{{< /rawhtml >}}

So (from here you see both terms are tractable):

{{< rawhtml >}}
$$

q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1}),q(x_{t-1}\mid x_0).
$$
{{< /rawhtml >}} 

Now plug in both Gaussians:

{{< rawhtml >}}
$$

q(x_t\mid x_{t-1})
=

\mathcal N(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I),

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

q(x_{t-1}\mid x_0)
=

\mathcal N(x_{t-1};\sqrt{\bar\alpha_{t-1}}x_0,(1-\bar\alpha_{t-1})I).

$$
{{< /rawhtml >}}

Thus

{{< rawhtml >}}
$$

q(x_{t-1}\mid x_t,x_0)
\propto
\mathcal N(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I),
\mathcal N(x_{t-1};\sqrt{\bar\alpha_{t-1}}x_0,(1-\bar\alpha_{t-1})I).

$$
{{< /rawhtml >}}

This is a product of Gaussians in {{< rawhtml >}}$x_{t-1}${{< /rawhtml >}}, so it must also be Gaussian in {{< rawhtml >}}$x_{t-1}${{< /rawhtml >}}:


**First term**:

{{< rawhtml >}}
$$

\mathcal N(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I)
\propto
\exp\left(
-\frac{1}{2\beta_t}|x_t-\sqrt{\alpha_t}x_{t-1}|^2
\right).

$$
{{< /rawhtml >}}

Expand:

{{< rawhtml >}}
$$

|x_t-\sqrt{\alpha_t}x_{t-1}|^2
=

x_t^\top x_t
-2\sqrt{\alpha_t}x_t^\top x_{t-1}
+\alpha_t x_{t-1}^\top x_{t-1}.

$$
{{< /rawhtml >}}

So this contributes

{{< rawhtml >}}
$$

-\frac{1}{2\beta_t}
\left(
x_t^\top x_t
-2\sqrt{\alpha_t}x_t^\top x_{t-1}
+\alpha_t x_{t-1}^\top x_{t-1}
\right).

$$
{{< /rawhtml >}}

**Second term**

{{< rawhtml >}}
$$

\mathcal N(x_{t-1};\sqrt{\bar\alpha_{t-1}}x_0,(1-\bar\alpha_{t-1})I)
\propto
\exp\left(
-\frac{1}{2(1-\bar\alpha_{t-1})}
|x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0|^2
\right).

$$
{{< /rawhtml >}}

Expand:

{{< rawhtml >}}
$$

|x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0|^2
=

x_{t-1}^\top x_{t-1}
-2\sqrt{\bar\alpha_{t-1}}x_0^\top x_{t-1}
+\bar\alpha_{t-1}x_0^\top x_0.

$$
{{< /rawhtml >}}

So this contributes:

{{< rawhtml >}}
$$

-\frac{1}{2(1-\bar\alpha_{t-1})}
\left(
x_{t-1}^\top x_{t-1}
-2\sqrt{\bar\alpha_{t-1}}x_0^\top x_{t-1}
+\bar\alpha_{t-1}x_0^\top x_0
\right).

$$
{{< /rawhtml >}}

Ignoring constants independent of {{< rawhtml >}}$x_{t-1}${{< /rawhtml >}}, the exponent of the product becomes:

{{< rawhtml >}}
$$

-\frac12
\left[
\left(
\frac{\alpha_t}{\beta_t}
+\frac{1}{1-\bar\alpha_{t-1}}
\right)x_{t-1}^\top x_{t-1}
-

2\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0
\right)^\top x_{t-1}
\right].

$$
{{< /rawhtml >}}

This has the canonical Gaussian form:

{{< rawhtml >}}
$$

-\frac12\left[
x^\top A x - 2b^\top x
\right]

$$
{{< /rawhtml >}}

with

{{< rawhtml >}}
$$

A=
\left(
\frac{\alpha_t}{\beta_t}
+\frac{1}{1-\bar\alpha_{t-1}}
\right)I,

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

b=
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0.

$$
{{< /rawhtml >}}

Therefore

{{< rawhtml >}}
$$

\boxed{
q(x_{t-1}\mid x_t,x_0)=\mathcal N(x_{t-1};\tilde\mu_t,\tilde\beta_t I)
}

$$
{{< /rawhtml >}}

with

{{< rawhtml >}}
$$

\tilde\beta_t = A^{-1}
=

\left(
\frac{\alpha_t}{\beta_t}
+\frac{1}{1-\bar\alpha_{t-1}}
\right)^{-1},

$$
{{< /rawhtml >}}

and

{{< rawhtml >}}
$$

\tilde\mu_t = A^{-1}b.

$$
{{< /rawhtml >}}

Simplify the variance:

{{< rawhtml >}}
$$

\tilde\beta_t
=

\left(
\frac{\alpha_t}{\beta_t}
+
\frac{1}{1-\bar\alpha_{t-1}}
\right)^{-1}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\left(
\frac{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}{\beta_t(1-\bar\alpha_{t-1})}
\right)^{-1}

$$
{{< /rawhtml >}}


{{< rawhtml >}}
$$

=
\frac{\beta_t(1-\bar\alpha_{t-1})}{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

\alpha_t(1-\bar\alpha_{t-1})+\beta_t
=

\alpha_t-\alpha_t\bar\alpha_{t-1}+\beta_t

$$
{{< /rawhtml >}}


{{< rawhtml >}}
$$

=
\alpha_t-\alpha_t\bar\alpha_{t-1}+1-\alpha_t

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
1-\alpha_t\bar\alpha_{t-1}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
1-\bar\alpha_t

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

\tilde\beta_t
=

\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

\boxed{
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
}

$$
{{< /rawhtml >}}


**Simplify the mean**


{{< rawhtml >}}
$$

\tilde\mu_t
=

\tilde\beta_t
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0
\right).

$$
{{< /rawhtml >}}

Substitute {{< rawhtml >}}$\tilde\beta_t=\dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t${{< /rawhtml >}}:

{{< rawhtml >}}
$$

\tilde\mu_t(x_t,x_0)
=

\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0.

$$
{{< /rawhtml >}}


Since {{< rawhtml >}}$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon_t${{< /rawhtml >}} and {{< rawhtml >}}$x_0=\frac{1}{\sqrt{\bar\alpha_t}}\left(x_t-\sqrt{1-\bar\alpha_t},\epsilon_t\right)${{< /rawhtml >}}:

{{< rawhtml >}}
$$

\tilde\mu_t(x_t,x_0)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t -
\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_t
\right)
.

$$
{{< /rawhtml >}}

Since {{< rawhtml >}}$1-\alpha_t=\beta_t${{< /rawhtml >}}:

{{< rawhtml >}}
$$

\boxed{
\tilde\mu_t
=

\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-

\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_t
\right)
}.

$$
{{< /rawhtml >}}


#### $p_\theta(x_t-1|x_t)$:

We assume that {{< rawhtml >}}$p_\theta${{< /rawhtml >}} is Gaussian:

{{< rawhtml >}}
$$

\boxed{
p_\theta(x_{t-1}\mid x_t)=\mathcal N\bigl(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\bigr)
}

$$
{{< /rawhtml >}}

Justification: if {{< rawhtml >}}$q(x_t\mid x_{t-1})${{< /rawhtml >}} is Gaussian, then {{< rawhtml >}}$q(x_{t-1}\mid x_t)${{< /rawhtml >}} is approximately Gaussian (because {{< rawhtml >}}$q(x_{t-1})\approx \mathcal N(m,C)${{< /rawhtml >}} becomes Gaussian-like after many small Gaussian noise steps.

### Loss function

From above:

{{< rawhtml >}}
$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(x_{t-1};\tilde\mu_t,\tilde\beta_t I)
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$
{{< /rawhtml >}}

Fix {{< rawhtml >}}$\Sigma_\theta(x_t,t)=\tilde\beta_t I${{< /rawhtml >}}, then KL between Gaussians with equal covariance:

{{< rawhtml >}}
$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I)
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta,\tilde\beta_t I)
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

D_{\mathrm{KL}}(q||p_\theta)
=
\mathbb E_q\left[
\log \frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)}
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\mathbb E_q\left[
\log
\frac{
\exp\left(-\frac{1}{2\tilde\beta_t}\left\|x_{t-1}-\tilde\mu_t\right\|^2\right)
}{
\exp\left(-\frac{1}{2\tilde\beta_t}\left\|x_{t-1}-\mu_\theta\right\|^2\right)
}
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\mathbb E_q\left[
-\frac{1}{2\tilde\beta_t}\left\|x_{t-1}-\tilde\mu_t\right|^2
+
\frac{1}{2\tilde\beta_t}\left\|x_{t-1}-\mu_\theta\right\|^2
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\frac{1}{2\tilde\beta_t}
\mathbb E_q\left[
\left\|x_{t-1}-\mu_\theta\right\|^2
-
\left\|x_{t-1}-\tilde\mu_t\right\|^2
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

\left\|x_{t-1}-\mu_\theta\right\|^2
=
\left\|x_{t-1}-\tilde\mu_t+\tilde\mu_t-\mu_\theta\right\|^2

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\left\|x_{t-1}-\tilde\mu_t\right\|^2
+
2(x_{t-1}-\tilde\mu_t)^\top(\tilde\mu_t-\mu_\theta)
+
\left\|\tilde\mu_t-\mu_\theta\right\|^2

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

D_{\mathrm{KL}}(q||p_\theta)
=
\frac{1}{2\tilde\beta_t}
\mathbb E_q\left[
2(x_{t-1}-\tilde\mu_t)^\top(\tilde\mu_t-\mu_\theta)
+
||\tilde\mu_t-\mu_\theta||^2
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

=
\frac{1}{2\tilde\beta_t}
\left[
2\mathbb E_q[x_{t-1}-\tilde\mu_t]^\top(\tilde\mu_t-\mu_\theta)
+
\left\|\tilde\mu_t-\mu_\theta\right\|^2
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

\mathbb E_q[x_{t-1}]=\tilde\mu_t

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

\mathbb E_q[x_{t-1}-\tilde\mu_t]=0

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

D_{\mathrm{KL}}(q||p_\theta)
=

\frac{1}{2\tilde\beta_t}
\left\|\tilde\mu_t-\mu_\theta\right\|^2

$$
{{< /rawhtml >}}

Therefore:
{{< rawhtml >}}
$$

\mathcal L_t
\propto
\mathbb E_{x_0,\epsilon,t}
\left[
\left\|\tilde\mu_t-\mu_\theta(x_t,t)\right\|^2
\right].

$$
{{< /rawhtml >}}

Fix and parameterize {{< rawhtml >}}$\mu_\theta(x_t,t)${{< /rawhtml >}} as:

{{< rawhtml >}}
$$

\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)
\right)

$$
{{< /rawhtml >}}


And given {{< rawhtml >}}$\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)${{< /rawhtml >}}:

{{< rawhtml >}}
$$

\mathcal L_t
\propto
\mathbb E_{x_0,\epsilon,t}
\left[
\left\|
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon
\right)
-
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)
\right)
\right\|^2
\right]

$$
{{< /rawhtml >}}


{{< rawhtml >}}
$$

=
\mathbb E_{x_0,\epsilon,t}
\left[
\left\|
\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}
\left(
\epsilon-\epsilon_\theta(x_t,t)
\right)
\right\|^2
\right]

$$
{{< /rawhtml >}}
{{< rawhtml >}}$\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}${{< /rawhtml >}} is constant, so:
{{< rawhtml >}}
$$

\propto
\mathbb E_{x_0,\epsilon,t}
\left[
\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2
\right]

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

x_t
=
\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

\boxed{
\mathcal L_{\mathrm{DDPM}}
=
\mathbb E_{t,x_0,\epsilon}
\left[
\left\|
\epsilon_\theta\left(
\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t
\right)
-\epsilon
\right\|^2
\right]
}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
t\sim \mathcal U\{1,T\},\qquad x_0\sim q_0(x_0),\qquad\epsilon\sim\mathcal N(0,I)
$$
{{< /rawhtml >}}

So maximizing {{< rawhtml >}}$p_\theta(x_0)${{< /rawhtml >}} is equivalent to minimizing the {{< rawhtml >}}$\ell_2${{< /rawhtml >}} distance between the noise predicted by the model and the actual noise that was added to obtain the noisy images.

The training process is to generate noisy images with {{< rawhtml >}}$t${{< /rawhtml >}} from {{< rawhtml >}}$0:T${{< /rawhtml >}}, then take each step, starting from {{< rawhtml >}}$T${{< /rawhtml >}}, give the noisy image at {{< rawhtml >}}$t${{< /rawhtml >}} to the model, model predicts how much noise was added to it, then compute and minimize the loss.

For inference, Take the image build from noise {{< rawhtml >}}$\mathcal N(0,I)${{< /rawhtml >}}, and construct {{< rawhtml >}}$x_t-1${{< /rawhtml >}} from {{< rawhtml >}}$x_t${{< /rawhtml >}} like this:
{{< rawhtml >}}
$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)+\sigma_t z.
$$
{{< /rawhtml >}}

(Previously, we showed that we parametarazied {{< rawhtml >}}$p_\theta(x_t-1|x_t)${{< /rawhtml >}} as:
{{< rawhtml >}}
$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N\bigl(\mu_\theta(x_t,t)\Sigma_\theta(x_t,t)\bigr)
$$
{{< /rawhtml >}}

with
{{< rawhtml >}}
$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
$$
{{< /rawhtml >}}

and

{{< rawhtml >}}
$$

\Sigma_\theta(x_t,t)=\sigma_t^2 I.

$$
{{< /rawhtml >}}
with 
{{< rawhtml >}}
$$

\sigma_t^2=\tilde\beta_t.

$$
{{< /rawhtml >}}
)


### DDIM

For DDIM, when generating {{< rawhtml >}}$x_t-1${{< /rawhtml >}} from {{< rawhtml >}}$x_t${{< /rawhtml >}}, the process is deterministic. Therefore (just rearrange {{< rawhtml >}}$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t},\epsilon${{< /rawhtml >}}):

{{< rawhtml >}}
$$
\hat x_0(x_t)=\frac{x_t-\sqrt{1-\bar\alpha_t},\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}},\hat x_0(x_t)+\sqrt{1-\bar\alpha_{t-1}}\epsilon_\theta(x_t,t)
$$
{{< /rawhtml >}}


One limitation of DDPM is that timestep skipping degrades sample quality, because each reverse step both denoises and injects Gaussian noise {{< rawhtml >}}$\sigma_t z_t${{< /rawhtml >}}; skipping many such steps breaks the approximation the reverse Markov chain relies on.


## PARADIGM 2: Score matching (Langevin Dynamics)

The idea is that we want to generate a realistic image, and therefore sample from the distribution of realistic images, {{< rawhtml >}}$p_{\text{data}}${{< /rawhtml >}}, which is unknown and difficult to sample from directly. One approach is to start from a simpler distribution that is easy to sample from, and then iteratively update the samples so that they move toward regions with higher probability under {{< rawhtml >}}$p_{\text{data}}${{< /rawhtml >}}.

To do this, we use the score function of {{< rawhtml >}}$p_{\text{data}}${{< /rawhtml >}}, which is easier to work with:

{{< rawhtml >}}
$$

\nabla_x \log p_{\text{data}}(x).

$$
{{< /rawhtml >}}

One important reason is that the normalization constant disappears during differentiation, since

{{< rawhtml >}}
$$

\nabla_x \log Z_\theta = 0,

$$
{{< /rawhtml >}}

meaning we do not need to compute the intractable normalizing constant:


{{< rawhtml >}}
$$

x_t = x_{t-1} + \frac{\alpha_i}{2}\, \nabla_x \log p_\text{data}(x_{t-1}) + \sqrt{\alpha_i}\epsilon_t

$$
{{< /rawhtml >}}

We add Gaussian noise to the images to "fill" the entire space with some probability mass. The parameter {{< rawhtml >}}$\alpha_i${{< /rawhtml >}} arises from the discretization of an underlying stochastic differential equation (SDE), which will be discussed in more detail below. The {{< rawhtml >}}$\sqrt{\alpha}${{< /rawhtml >}} term comes from the fact that the SDE contains a Wiener process whose variance is (dt) (equivalent to (\alpha) after discretization), and therefore whose standard deviation is (\sqrt{dt}).

Then why 1/2 for gradient update coefficient? That requires more depth.

There is at least one other phenomenon in the world around us that, much like machine learning diffusion models, consists of progressively adding randomness over multiple steps: the diffusion of gas molecules in a room 🌚. That is, in fact, where the name *diffusion* comes from. This formula {{< rawhtml >}}$x_t = x_{t-1} + \frac{\alpha_i}{2}\, \nabla_x \log p_\text{data}(x_{t-1}) + \sqrt{\alpha_i}\epsilon_t${{< /rawhtml >}} is exactly the location of a molecule at the time {{< rawhtml >}}$t${{< /rawhtml >}} which can be calculated using the posisiton at time {{< rawhtml >}}$t-1${{< /rawhtml >}}. In fact, this formula is the descretized form of this continous relationship:
{{< rawhtml >}}
$$

dx_t = \lim_{dt \to 0} x_t+dt - x_{t} =  \frac{1}{2} \nabla_x \log p_\text{data}(x_{t-1})dt + dW_t

$$
{{< /rawhtml >}}

Where {{< rawhtml >}}$W_t${{< /rawhtml >}} represents Brownian motion and is a normal varirable whose value is continously changing over time and its variance increasing with time distance. Therefore, {{< rawhtml >}}$W_t \sim \mathcal N(0, t)${{< /rawhtml >}} is the total accumulated randomness up to time {{< rawhtml >}}$t${{< /rawhtml >}}, and {{< rawhtml >}}$dW_t \sim \mathcal N(0,dtI)${{< /rawhtml >}} is tiny new random increment added during {{< rawhtml >}}$dt${{< /rawhtml >}}. Thus, when we descritize the process, {{< rawhtml >}}$dW_t${{< /rawhtml >}} turns into a normal variable with the variance of step size {{< rawhtml >}}$\sqrt{\alpha}\sigma${{< /rawhtml >}}.

{{< rawhtml >}}$\alpha_i${{< /rawhtml >}} controls the degree to which we discretize the underlying continuous process. This is reasonable because, in the molecular diffusion analogy, time is continuous, and the forces acting on each molecule can occur at any instant in time.

Therefore, if we explain where the {{< rawhtml >}}$\frac{1}{2}${{< /rawhtml >}} term comes from in {{< rawhtml >}}$dx = \frac{1}{2}\nabla_x \log p_{\text{data}}(x_{t-1})\,dt + dW_t${{< /rawhtml >}}, the same explanation also applies to the {{< rawhtml >}}$\frac{1}{2}${{< /rawhtml >}} term in the discretized equation {{< rawhtml >}}$x_t = x_{t-1} + \frac{\alpha_i}{2},\nabla_x \log p_{\text{data}}(x_{t-1}) + \sqrt{\alpha_i}\epsilon_t.${{< /rawhtml >}}



We can write (dx) in the more general form called an Itô stochastic differential equation:
{{< rawhtml >}}
$$

dx = f(x)dt + g(x)dW.

$$
{{< /rawhtml >}}

Now that we've introduced the “molecules” analogy, we can show that specific choices of the drift {{< rawhtml >}}$f(x)${{< /rawhtml >}} and diffusion {{< rawhtml >}}$g(x)${{< /rawhtml >}} determine the stationary final distribution of the molecules. In diffusion models, the goal is to choose the drift term so that the reverse process converges toward the data distribution {{< rawhtml >}}$p_{\text{data}}${{< /rawhtml >}}.

To find the stationary distribution of the process, we use Fokker-Planck equation of that Itô SDE. We take {{< rawhtml >}}$\rho(x,t)${{< /rawhtml >}} as the pdf of molecules, and want to find a distribution where {{< rawhtml >}}$\frac{\partial \rho(x,t)}{\partial t} = 0${{< /rawhtml >}}, meaning its stationary.

To find {{< rawhtml >}}$\frac{\partial \rho(x,t)}{\partial t} = 0${{< /rawhtml >}}, we have to define a test function {{< rawhtml >}}$\phi(x)${{< /rawhtml >}}. We want to reach an equation that is true for any smooth {{< rawhtml >}}$\phi(x)${{< /rawhtml >}}.

First, :
{{< rawhtml >}}
$$

d\phi(x) = \phi(x + dx) - \phi(x)  

$$
{{< /rawhtml >}}

Expand the first around {{< rawhtml >}}$x${{< /rawhtml >}}, cancel {{< rawhtml >}}$\phi(x)${{< /rawhtml >}} and note that {{< rawhtml >}}$u = x+dx${{< /rawhtml >}}, {{< rawhtml >}}$\frac{du}{dx}=1${{< /rawhtml >}} and {{< rawhtml >}}$\frac{d f(x+dx)}{dx} = \frac{d f(u)}{du}\frac{du}{dx} = \frac{d f(u)}{du}${{< /rawhtml >}}
{{< rawhtml >}}
$$

d\phi(x) = \sum_{j=1}^{\infty}{\frac{\phi(x)^{(j)}}{j}dx^j}

$$
{{< /rawhtml >}}

In ordinary differential equations, we usually treat terms of order {{< rawhtml >}}$dx^j${{< /rawhtml >}} for {{< rawhtml >}}$j > 1${{< /rawhtml >}} as negligible. However, for random motion in space, these higher-order terms are no longer negligible, because random infinitesimal movements can accumulate together (for example, two small movements occurring in the same direction). Therefore, unlike ordinary calculus, we cannot stop at the first derivative term alone:

{{< rawhtml >}}
$$
d\phi(x) = \frac{\partial \phi}{\partial x} dx + \frac{1}{2} \frac{\partial^2 \phi}{\partial x^2} (dx)^2
$$
{{< /rawhtml >}}

Also, please note that {{< rawhtml >}}$(dW)^2 \approx dt${{< /rawhtml >}} :), since: 
{{< rawhtml >}}
$$
Var(dW) = dt = E(dW^2) - E(dW)^2 = E(dW^2) \quad \text{since $dW \sim \mathcal N(0,dt)$}
$$
{{< /rawhtml >}}
but:
{{< rawhtml >}}
$$

Var(dW^2) = E(dW^4) - E(dW^2)^2 = 3dt^2 - dt^2 = 2dt^2 \text{which goes to zero faster than $dt$ as $dt\to0$.}

$$
{{< /rawhtml >}}
So {{< rawhtml >}}$(dW)^2${{< /rawhtml >}} is a random variable with mean of {{< rawhtml >}}$dt${{< /rawhtml >}} and a very tiny variance.

Substituting {{< rawhtml >}}$dx = f dt + g dW${{< /rawhtml >}} and using the Itô rule {{< rawhtml >}}$(dW)^2 = dt${{< /rawhtml >}} (while terms like {{< rawhtml >}}$dt^2${{< /rawhtml >}} and {{< rawhtml >}}$dt dW${{< /rawhtml >}} vanish):
{{< rawhtml >}}
$$
d\phi(x) = \left[ f(x) \frac{\partial \phi}{\partial x} + \frac{1}{2} g(x)^2 \frac{\partial^2 \phi}{\partial x^2} \right] dt + g(x) \frac{\partial \phi}{\partial x} dW
$$
{{< /rawhtml >}}
Now, taking expectation over {{< rawhtml >}}$x \sim \rho(x,t)${{< /rawhtml >}} from both sides, while {{< rawhtml >}}$E(dW) = 0${{< /rawhtml >}}
{{< rawhtml >}}
$$

\begin{align*}
\frac{d}{dt} E[\phi(x(t))] &= E\left[ f(x) \frac{\partial \phi}{\partial x} + \frac{1}{2} g(x)^2 \frac{\partial^2 \phi}{\partial x^2} \right] \\
\int \phi(x) \frac{\partial \rho(x,t)}{\partial t} dx &= \int \left[ f(x) \frac{\partial \phi}{\partial x} + \frac{1}{2} g(x)^2 \frac{\partial^2 \phi}{\partial x^2} \right] \rho(x,t) dx
\end{align*}

$$
{{< /rawhtml >}}

Integration by parts :
We assume {{< rawhtml >}}$\rho${{< /rawhtml >}} and its derivatives vanish at the boundaries ({{< rawhtml >}}$x \to \pm \infty${{< /rawhtml >}}). Because {{< rawhtml >}}$\rho${{< /rawhtml >}} is a pdf and can not have be positive everywhere in the world since it has to sum to 1.

*   **For the first term:**
    {{< rawhtml >}}
$$
\int_{-\infty}^{\infty} \left( \rho f \frac{\partial \phi}{\partial x} \right) dx = \rho f \phi\Big|_{-\infty}^{\infty} - \int \phi \frac{\partial}{\partial x} (f \rho) dx = - \int \phi \frac{\partial}{\partial x} (f \rho)
$$
{{< /rawhtml >}}
*   **For the second term (integrate by parts twice):**
    {{< rawhtml >}}
$$
\int_{-\infty}^{\infty} \left( \frac{1}{2} \rho g^2 \frac{\partial^2 \phi}{\partial x^2} \right) dx = \int \phi \frac{\partial^2}{\partial x^2} \left( \frac{1}{2} g^2 \rho \right) dx
$$
{{< /rawhtml >}}

Substitute these back into the equation:
{{< rawhtml >}}
$$
\int \phi(x) \frac{\partial \rho}{\partial t} dx = \int \phi(x) \left[ -\frac{\partial}{\partial x} (f \rho) + \frac{1}{2} \frac{\partial^2}{\partial x^2} (g^2 \rho) \right] dx
$$
{{< /rawhtml >}}


Since this equality must hold for **any** arbitrary test function {{< rawhtml >}}$\phi(x)${{< /rawhtml >}}, the terms inside the integrals must be equal:
{{< rawhtml >}}
$$
\frac{\partial \rho(x,t)}{\partial t} = -\frac{\partial}{\partial x} [f(x)\rho(x,t)] + \frac{1}{2} \frac{\partial^2}{\partial x^2} [g(x)^2 \rho(x,t)]
$$
{{< /rawhtml >}}

In multi-dimensional vector notation, the partial derivatives {{< rawhtml >}}$\frac{\partial}{\partial x}${{< /rawhtml >}} become the divergence {{< rawhtml >}}$\nabla \cdot${{< /rawhtml >}} and the Laplacian {{< rawhtml >}}$\nabla^2${{< /rawhtml >}} operators:
{{< rawhtml >}}
$$

\boxed{
\frac{\partial \rho}{\partial t} = -\nabla \cdot [f\rho] + \frac{1}{2} \nabla^2 [g^2 \rho]
}

$$
{{< /rawhtml >}}

Substitute our {{< rawhtml >}}$f(x)${{< /rawhtml >}} and {{< rawhtml >}}$g(x)${{< /rawhtml >}} into the equation:

{{< rawhtml >}}
$$

\frac{\partial \rho}{\partial t} = -\nabla \cdot \left( \frac{1}{2} (\nabla \log p) \rho \right) + \frac{1}{2} \nabla^2 \rho

$$
{{< /rawhtml >}}

To obtain a stationary state, set {{< rawhtml >}}$\frac{\partial \rho}{\partial t} = 0${{< /rawhtml >}}:
{{< rawhtml >}}
$$

0 = -\frac{1}{2} \nabla \cdot [(\nabla \log p) \rho] + \frac{1}{2} \nabla \cdot [\nabla \rho] \\
\nabla \cdot [(\nabla \log p) \rho] = \nabla \cdot [\nabla \rho]

$$
{{< /rawhtml >}}

This equation holds, if {{< rawhtml >}}$\rho=p${{< /rawhtml >}}. Therefore, for those specific {{< rawhtml >}}$f(x,t)=\frac12 \nabla_x \log p_{\mathrm{data}}(x)${{< /rawhtml >}} and {{< rawhtml >}}$g(x,t)=1${{< /rawhtml >}}, the stationary distribution, {{< rawhtml >}}$\rho(x,t)${{< /rawhtml >}} is equal to the distribution {{< rawhtml >}}$p${{< /rawhtml >}} that we want to sample from.

But one problem still remains, and thats we dont have {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}} and therefore its score function. :)

So, we have to find an estimation of score function of {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}}, {{< rawhtml >}}$s_\theta(x)${{< /rawhtml >}}, ideally through minimzing:
{{< rawhtml >}}
$$

\mathcal{L}_{\mathrm{SM}}=\mathbb{E}_x\left[\left\|s_\theta(x)-\nabla_x \log p_{\mathrm{data}}(x)\right\|^2\right]

$$
{{< /rawhtml >}}

But again, we do not have {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}}.


However, we can make {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}} easier to sample from by adding noise to its samples (for now, only one step). let:
{{< rawhtml >}}
$$

\tilde{x} = x + \sigma \epsilon \qquad \text{with} \qquad \epsilon \sim \mathcal{N}(0, I)

$$
{{< /rawhtml >}}

Therefore:

{{< rawhtml >}}
$$

q_\sigma(\tilde{x}\mid x) = \mathcal{N}(x, \sigma^2 I) \;\longrightarrow\; \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}\mid x) = - \frac{\tilde{x}-x}{\sigma^2} \\
\text{and} \\ 
q_\sigma(\tilde{x}) = \int q_\sigma(\tilde{x}\mid x)\, p_{\mathrm{data}}(x)\, dx

$$
{{< /rawhtml >}}


Then, you can replce {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}} with noised data distribution, {{< rawhtml >}}$\log q_\sigma(\tilde{x})${{< /rawhtml >}}, which is easier to sample from:
{{< rawhtml >}}
$$

\begin{align*}
\mathcal{L}_{\mathrm{SM}}(q_\sigma) &= \mathbb{E}_{\tilde{x}} \left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}) \right\|^2 \right] \\
&= \mathbb{E}_{x,\tilde{x}} \left[ \| s_\theta(\tilde{x}) - \underbrace{\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}\mid x)}_{-\frac{\tilde{x}-x}{\sigma^2}} \|^2 \right]
\end{align*}

$$
{{< /rawhtml >}}

The equality comes from:

{{< rawhtml >}}
$$

J_{\mathrm{SM}\,q_\sigma}(\theta)
=
\mathbb{E}_{q_\sigma(\tilde{x})} \left[ \frac12 \|s_\theta(\tilde{x})\|^2 \right]
-
\mathbb{E}_{q_\sigma(\tilde{x})}\left[ \left\langle s_\theta(\tilde{x}), \frac{\partial \log q_\sigma(\tilde{x})}{\partial \tilde{x}} \right\rangle \right]
+
\mathbb{E}_{q_\sigma(\tilde{x})}\left[\frac12 \left\| \frac{\partial \log q_\sigma(\tilde{x})}{\partial \tilde{x}} \right\|^2 \right]

$$
{{< /rawhtml >}}

Where {{< rawhtml >}}$\mathbb{E}_{q_\sigma(\tilde{x})}\left[\frac12 \left\| \frac{\partial \log q_\sigma(\tilde{x})}{\partial \tilde{x}} \right\|^2 \right]${{< /rawhtml >}} is a constant that does not depend on {{< rawhtml >}}$\theta${{< /rawhtml >}}, and can be ignored. Also, {{< rawhtml >}}$\mathbb{E}_{q_\sigma(\tilde{x})} \left[ \frac12 \|s_\theta(\tilde{x})\|^2 \right]${{< /rawhtml >}} appears identically in both sides of the equality above. Furthermore:

{{< rawhtml >}}
$$

\mathbb{E}_{q_\sigma(\tilde{x})}\left[ \left\langle s_\theta(\tilde{x}), \frac{\partial \log q_\sigma(\tilde{x})}{\partial \tilde{x}} \right\rangle \right]
=
\int_{\tilde{x}} q_\sigma(\tilde{x}) \left\langle s_\theta(\tilde{x}), \frac{\partial \log q_\sigma(\tilde{x})}{\partial \tilde{x}} \right\rangle d\tilde{x}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

= \int_{\tilde{x}} q_\sigma(\tilde{x}) \left\langle s_\theta(\tilde{x}), \frac{ \frac{\partial}{\partial \tilde{x}} q_\sigma(\tilde{x}) }{ q_\sigma(\tilde{x}) } \right\rangle d\tilde{x}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

= \int_{\tilde{x}} \left\langle s_\theta(\tilde{x}), \frac{\partial}{\partial \tilde{x}} q_\sigma(\tilde{x}) \right\rangle d\tilde{x}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

= \int_{\tilde{x}} \left\langle s_\theta(\tilde{x}), \frac{\partial}{\partial \tilde{x}} \int_x q_0(x)\, q_\sigma(\tilde{x}\mid x)\,dx \right\rangle d\tilde{x}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

= \int_{\tilde{x}} \left\langle s_\theta(\tilde{x}), \int_x q_0(x) \frac{\partial q_\sigma(\tilde{x}\mid x)}{\partial \tilde{x}} dx \right\rangle d\tilde{x}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

= \int_{\tilde{x}} \left\langle s_\theta(\tilde{x}), \int_x q_0(x)\, q_\sigma(\tilde{x}\mid x) \frac{\partial \log q_\sigma(\tilde{x}\mid x)}{\partial \tilde{x}} dx \right\rangle d\tilde{x}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

= \int_{\tilde{x}} \int_x q_0(x)\, q_\sigma(\tilde{x}\mid x) \left\langle s_\theta(\tilde{x}), \frac{\partial \log q_\sigma(\tilde{x}\mid x)}{\partial \tilde{x}} \right\rangle dx\,d\tilde{x}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

= \int_{\tilde{x}} \int_x q_\sigma(\tilde{x},x) \left\langle s_\theta(\tilde{x}), \frac{\partial \log q_\sigma(\tilde{x}\mid x)}{\partial \tilde{x}} \right\rangle dx\,d\tilde{x}

$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$

= \mathbb{E}_{q_\sigma(\tilde{x},x)} \left[ \left\langle s_\theta(\tilde{x}), \frac{\partial \log q_\sigma(\tilde{x}\mid x)}{\partial \tilde{x}} \right\rangle \right]

$$
{{< /rawhtml >}}


Therefore, we reached that's easy to compute, called denoising score matching (DSM) loss:
{{< rawhtml >}}
$$

\mathcal{L}_{DSM} = \mathbb{E}_{x,\tilde{x}} \left[ \| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}\mid x) \|^2 \right]

$$
{{< /rawhtml >}}

But the problem is that {{< rawhtml >}}$\log q_\sigma(\tilde{x})${{< /rawhtml >}} is not equal to {{< rawhtml >}}$\log p_\text{data}(x)${{< /rawhtml >}}, and the discrepancy between them increases with {{< rawhtml >}}$\sigma${{< /rawhtml >}}. One possible solution is to set {{< rawhtml >}}$\sigma${{< /rawhtml >}} to a very low value. In that case, however, {{< rawhtml >}}$q_\sigma(\tilde{x})${{< /rawhtml >}} becomes very similar to {{< rawhtml >}}$p_\text{data}(x)${{< /rawhtml >}}, and consequently {{< rawhtml >}}$\nabla \log q_\sigma(\tilde{x})${{< /rawhtml >}} becomes close to {{< rawhtml >}}$\nabla \log p_\text{data}(x)${{< /rawhtml >}}. As a result, the score function tends to direct samples toward nearby high-density regions more consistently. Therefore, points that lie in low-density regions are sampled very rarely (They are therefore rarely generated as model outputs and shown to the user as generated images.). In another viewpoint, the low density points have low {{< rawhtml >}}$q_\sigma(\tilde{x})${{< /rawhtml >}} (because they have low density in {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}} and {{< rawhtml >}}$q_\sigma(\tilde{x})${{< /rawhtml >}} is very alike {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}} when {{< rawhtml >}}$\sigma${{< /rawhtml >}} is set to a low value) and therefore, don't contribute to the loss {{< rawhtml >}}$\mathcal{L}_{\mathrm{SM}}(q_\sigma) = \int \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}) \right\|^2 q_\sigma(\tilde{x}) d\tilde{x}${{< /rawhtml >}} that much. Since we usually start from pure Gaussian noise :), it is very likely that the initial sample lies in a very low-density region. When {{< rawhtml >}}$\sigma${{< /rawhtml >}} is set to a very low value, {{< rawhtml >}}$q_\sigma(\tilde{x})${{< /rawhtml >}} becomes highly concentrated around the data manifold, so the model receives little training signal for points far from high-density regions. Therefore, the estimated score in such regions may be inaccurate and may fail to guide samples effectively toward higher-density regions. If you increase {{< rawhtml >}}$\sigma${{< /rawhtml >}}, your model will learn in which direction are the high density points from that noisy sample, by you noisy distribution will be far different than {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}}


![low and high values of sigma from https://github.com/afshinea/stanford-cme-296-diffusion-large-vision-models/blob/main/en/cheatsheet-diffusion-large-vision-models.pdf](images/image1.png)


The solution to balance this trade-off is to use multiple {{< rawhtml >}}$\sigma${{< /rawhtml >}} values, {{< rawhtml >}}$\sigma_i${{< /rawhtml >}}, for different stages and reduce them progressively as denoising proceeds. This means using high values of {{< rawhtml >}}$\sigma_i${{< /rawhtml >}} for early {{< rawhtml >}}$i${{< /rawhtml >}}'s, which allows very noisy initial samples to contribute to the loss and only to roughly know in which direction the high denstity points are. By reducing {{< rawhtml >}}$\sigma_i${{< /rawhtml >}} for later {{< rawhtml >}}$i${{< /rawhtml >}}'s, {{< rawhtml >}}$q_{\sigma_i}(\tilde{x}_i)${{< /rawhtml >}} is gradually refined to resemble {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}} more closely, leading to more accurate score function direction estimates. Applying Langevin sampling with reduced amount of noise is called *Annealed Langevin Dynamics* (ALD).

So the conceptual, but intractable, objective (loss) becomes:
{{< rawhtml >}}
$$
 
\mathcal{L}_{\mathrm{NCSN}} = \sum_{i=1}^{L} \lambda(i)\, \mathbb{E}_x \left[ \left\| s_\theta(x,\sigma_i) - \nabla_x \log q_{\sigma_i}(x) \right\|_2^2 \right] 

$$
{{< /rawhtml >}}

And the annealed Lengevin updates become:
{{< rawhtml >}}
$$

x \leftarrow x + \frac{\alpha_i}{2} s_\theta(x,\sigma_i) + \sqrt{\alpha_i}\,\epsilon

$$
{{< /rawhtml >}}



#### Connection between DDPM and score matching

DDPM is trying to guess and minimized the {{< rawhtml >}}$\mathcal{l}^2${{< /rawhtml >}} distance between the added noise and the actual noise for each step, while NCSN tries to find the score function at each step. Bascially, score function direction should be in the opposite direction of the the added noise:
{{< rawhtml >}}
$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
q(x_t \mid x_0) = \mathcal{N} \left( \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I \right)
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\nabla_{x_t}\log(q(x_t \mid x_0)) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$
{{< /rawhtml >}}


One difference: DDPM forward process is "variance preserving", while NCSN forward process is "variance exploding".



### Making things continuous (again!)

We can make the process of DDPM and NCSN continous to be able to later use some already discovered results from stochastic differential equations.

For DDPM:

{{< rawhtml >}}
$$

x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon \\[6pt]

x_t - x_{t-1}=\left(\sqrt{1-\beta_t}-1\right)x_{t-1} + \sqrt{\beta_t}\epsilon \\[6pt]

x_t - x_{t-1} = \left(\sqrt{1-\beta(t)dt}-1\right)x_{t-1} + \sqrt{\beta(t)dt}\epsilon \qquad \text{with $\beta_t = \beta(t)dt$} \\[6pt]

x_t - x_{t-1} \approx -\frac12\beta(t)x_{t-1}dt + \sqrt{\beta(t)}\sqrt{dt}\epsilon \qquad \text{with $\sqrt{1-\beta(t)dt} \approx 1-\frac12\beta(t)dt$} \\[6pt]

\boxed{dx=-\frac12\beta(t)xdt+\sqrt{\beta(t)}dW_t} \qquad \text{with $dW_t = \sqrt{dt}\epsilon$ and $x_t - x_{t-1} \to dx$}


$$
{{< /rawhtml >}}

For NCSN:
{{< rawhtml >}}
$$

\begin{aligned}
x_t&=x_{t-1}+\sqrt{\sigma_t^2-\sigma_{t-1}^2}\epsilon \\[6pt]

x_t-x_{t-1} &= \sqrt{\frac{d\sigma_t^2}{dt} dt}\epsilon \qquad \text{with $\frac{df(x)}{dx} = \lim_{h \to 0}\frac{f(x+h)-f(x)}{h}$} \\[6pt]

dx &= \boxed{\sqrt{\frac{d\sigma_t^2}{dt}}dW} \qquad \text{with $x_t - x_{t-1} \to dx$ and $dW = \sqrt{dt}\epsilon$}
\end{aligned}

$$
{{< /rawhtml >}}


#### The reverse process

Making things continuous make the process (forward and reverse), in terms of an SDE of the form:
{{< rawhtml >}}
$$

dx = f(x,t)dt + g(t)dW

$$
{{< /rawhtml >}}

The forward process for DDPM is when {{< rawhtml >}}$f = -\frac12 \beta(t) x${{< /rawhtml >}} and {{< rawhtml >}}$g = \sqrt{\beta(t)}${{< /rawhtml >}}.

For any "nice" forward process in the form of {{< rawhtml >}}$dx = f(x,t)dt + g(t)dW${{< /rawhtml >}}, the reverse SDE is as follows:
{{< rawhtml >}}
$$

dx = \left[f(x,t) - g(t)^2 \nabla_x \log(p_t(x))\right]dt + g(t)d\overline{W}

$$
{{< /rawhtml >}}

That is because we previously showed that the probability density {{< rawhtml >}}$\rho(x,t)${{< /rawhtml >}} of an SDE {{< rawhtml >}}$dx = f(x,t)dt + g(t)dW${{< /rawhtml >}} evolves according to the Fokker--Planck equation:

{{< rawhtml >}}
$$
\frac{\partial \rho}{\partial t} = -\nabla \cdot [f\rho] + \frac{1}{2} \nabla^2 [g^2 \rho] \qquad (1)
$$
{{< /rawhtml >}}
Since {{< rawhtml >}}$g(t)${{< /rawhtml >}} does not depend on {{< rawhtml >}}$x${{< /rawhtml >}}, we can simplify the second term: {{< rawhtml >}}$\nabla^2 [g^2 \rho] = g^2 \nabla^2 \rho${{< /rawhtml >}}.

We want to describe the process moving backward in time. Let {{< rawhtml >}}$\tau${{< /rawhtml >}} be the reverse time variable (for clarity, and the convenience of dealing with increasing time variable), where {{< rawhtml >}}$\tau = T - t${{< /rawhtml >}}. Let {{< rawhtml >}}$\bar{\rho}(x, \tau) = \rho(x, T-\tau) = \rho(x, t)${{< /rawhtml >}} be the density in reverse time.

By the chain rule, {{< rawhtml >}}$\frac{\partial \bar{\rho}}{\partial \tau} = -\frac{\partial \rho}{\partial t}${{< /rawhtml >}}. Substituting this into Equation (1):
{{< rawhtml >}}
$$

\begin{aligned}
-\frac{\partial \bar{\rho}}{\partial \tau} &= -\nabla \cdot [f\rho] + \frac{1}{2} g^2 \nabla^2 \rho \\[6pt]
\frac{\partial \bar{\rho}}{\partial \tau} &= \nabla \cdot [f\rho] - \frac{1}{2} g^2 \nabla^2 \rho. \qquad (2)
\end{aligned}

$$
{{< /rawhtml >}}

To find the reverse SDE, we need to rewrite the right-hand side of Equation (2) to look like a standard FPE. A general FPE for a process {{< rawhtml >}}$dx = \bar{f}d\tau + \bar{g}d\bar{W}${{< /rawhtml >}} has the form:
{{< rawhtml >}}
$$
\frac{\partial \bar{\rho}}{\partial \tau} = -\nabla \cdot [\bar{f}\bar{\rho}] + \frac{1}{2} \bar{g}^2 \nabla^2 \bar{\rho} \qquad (3)
$$
{{< /rawhtml >}}
set the terms of (2) and (3) equal to each other. We assume the diffusion coefficient remains the same magnitude: {{< rawhtml >}}$\bar{g} = g${{< /rawhtml >}}:
{{< rawhtml >}}
$$

\begin{aligned}
-\nabla \cdot [\bar{f}\rho] + \frac{1}{2} g^2 \nabla^2 \rho &= \nabla \cdot [f\rho] - \frac{1}{2} g^2 \nabla^2 \rho \\[6pt]
-\nabla \cdot [\bar{f}\rho] &= \nabla \cdot [f\rho] - g^2 \nabla^2 \rho \\[6pt]
\nabla \cdot [\bar{f}\rho] &= \nabla \cdot [f\rho - g^2 \nabla \rho] \qquad \text{with $\nabla^2 \rho = \nabla \cdot (\nabla \rho)$} \\[6pt]
\bar{f}\rho &= -f\rho + g^2 \nabla \rho \\[6pt]
\bar{f} &= -f + g^2 \frac{\nabla \rho}{\rho} \\[6pt]
\bar{f}(x, \tau) &= -f(x, t) + g(t)^2 \nabla_x \log \rho_t(x) \qquad \text{with $\nabla \log \rho = \frac{\nabla \rho}{\rho}$}
\end{aligned}

$$
{{< /rawhtml >}}

The SDE in reverse time {{< rawhtml >}}$\tau${{< /rawhtml >}} is (rewriting {{< rawhtml >}}$dx = \bar{f}d\tau + \bar{g}d\bar{W}${{< /rawhtml >}} given {{< rawhtml >}}$\bar{g} = g${{< /rawhtml >}}):
{{< rawhtml >}}
$$

\begin{aligned}
dx &= \bar{f} d\tau + g d\bar{W} \\[6pt]
dx &= \left[-f(x,t) + g(t)^2 \nabla_x \log p_t(x)\right] d\tau + g(t) d\bar{W} \\[6pt]
dx &= \left[-f(x,t) + g(t)^2 \nabla_x \log p_t(x)\right] (-dt) + g(t) d\bar{W} \qquad \text{with $d\tau = -dt$} \\[6pt]
dx &= [f(x,t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{W}
\end{aligned}

$$
{{< /rawhtml >}}

This confirms the reverse-time SDE:
{{< rawhtml >}}
$$
\boxed{dx = \left[f(x,t) - g(t)^2 \nabla_x \log(p_t(x))\right]dt + g(t)d\overline{W}}
$$
{{< /rawhtml >}}
where {{< rawhtml >}}$d\overline{W}${{< /rawhtml >}} is a standard Wiener process when time flows backward (and does not reconstruct the exact forward noise realization.).



### Training

We sample a clean image, sample Gaussian noise, and add it to the image to create a noisy sample. We then compute the conditional score function {{< rawhtml >}}$\nabla_{x_t}\log p(x_t \mid x_0)${{< /rawhtml >}}, which is tractable, compute the loss between this target score and the model prediction, and finally average the loss over many samples.

{{< rawhtml >}}
$$

\mathcal{L}_{\mathrm{DSM}} = \mathbb{E}_{t,x_0,x_t} \left[ \lambda_t \| s\theta(x_t,t)
- 
\underbrace{\nabla_{x_t}\log p(x_t\mid x_0)}_{-\frac{\epsilon}{\sigma_t}} \|^2 \right]

$$
{{< /rawhtml >}}

**Variance Preserving (DDPM-style)**

{{< rawhtml >}}
$$
 x_t \mid x_0 \sim \mathcal N \left( \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I \right)
$$
{{< /rawhtml >}}

**Variance Exploding (NCSN-style)**

{{< rawhtml >}}
$$
 x_t \mid x_0 \sim \mathcal N \left( x_0, \sigma_t^2 I \right)
$$
{{< /rawhtml >}}



### Inference

Sample an image {{< rawhtml >}}$x_T${{< /rawhtml >}} from pure Gaussian noise, {{< rawhtml >}}$\mathcal{N}(0,\sigma_T^2 I)${{< /rawhtml >}}, then use the Euler-Maruyama discretized form of the reverse SDE above to progressively produce less noisy images (by replacing {{< rawhtml >}}$f(x_{t_i}, t_i)${{< /rawhtml >}} and {{< rawhtml >}}$g(t_i)${{< /rawhtml >}} with their corresponding DDPM or NCSN forms):

{{< rawhtml >}}
$$
x_{t_{i-1}} = x_{t_i} + \left[ f(x_{t_i}, t_i) - g(t_i)^2 s_\theta(x_{t_i}, t_i) \right]\Delta t + g(t_i)\sqrt{\Delta t}\xi_i
$$
{{< /rawhtml >}}


### Converting score matching SDE to and ODE

SDEs contain a Wiener process and are stochastic, which limits how large the step size can be when skipping steps to accelerate the computations. However, we can convert that SDE into an ODE that produces the same probability density {{< rawhtml >}}$p${{< /rawhtml >}} (previously denoted by {{< rawhtml >}}$\rho${{< /rawhtml >}}). That makes different trajectories for each particle, but the same density.
We start by the forward SDE:
{{< rawhtml >}}
$$
dx = f(x,t)dt + g(t)dW
$$
{{< /rawhtml >}}

Fokker-Planck equation:
{{< rawhtml >}}
$$
\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot (f(x,t)p_t(x)) + \frac{1}{2}g(t)^2 \Delta p_t(x) \qquad \text{}
$$
{{< /rawhtml >}}

Given {{< rawhtml >}}$\Delta p_t = \nabla \cdot \left( p_t \nabla \log p_t \right)${{< /rawhtml >}}:
{{< rawhtml >}}
$$
\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot \left( \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) \right] p_t(x) \right)
$$
{{< /rawhtml >}}

Identify the velocity term in the flux equation, and:
{{< rawhtml >}}
$$
dx = \left[ f(x,t) - \frac{1}{2}g(t)^2 \nabla \log(p_t(x)) \right]dt
$$
{{< /rawhtml >}}

Thats probability flow ordinary differential equation: PF-ODE.

This makes the process fully deterministic, and therefore faster to solve, but lowers the quality. Now there multiple ways to solve it like Euler method and DPM-solver.


## PARADIGM 3: Flow matching

The idea is to obtain a vector field such that, if we place samples from an easy-to-sample distribution ({{< rawhtml >}}$p_0${{< /rawhtml >}} at time 0) into it, then at time 1, the resulting trajectories transport them to the hard-to-sample data distribution, {{< rawhtml >}}$p_1${{< /rawhtml >}}. Therefore, the goal is to map {{< rawhtml >}}$x_0 \sim p_0${{< /rawhtml >}} to {{< rawhtml >}}$x_1 \sim p_1${{< /rawhtml >}}:

- Training: Estimate {{< rawhtml >}}$u_t(x)${{< /rawhtml >}} for all time {{< rawhtml >}}$t${{< /rawhtml >}} and all locations {{< rawhtml >}}$x${{< /rawhtml >}} via {{< rawhtml >}}$u_t^\theta(x)${{< /rawhtml >}}

- Inference: Sample from the initial distribution and solve numerically the ODE using the learned vector field {{< rawhtml >}}$u_t^\theta(x)${{< /rawhtml >}}:
{{< rawhtml >}}
$$

\hat{x}_1 = x_0 + \int_0^1 u_t^\theta(x)dt.

$$
{{< /rawhtml >}}

### Estimating the vector field

The idea is to find a vector field, ideally, through minimizing the follwing loss (FM: flow matching):
{{< rawhtml >}}
$$

\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,x} \left[ \left\| u_t^\theta(x)-u_t(x) \right\|^2 \right]

$$
{{< /rawhtml >}}

But we dont know {{< rawhtml >}}$u_t(x)${{< /rawhtml >}} :)

However we could use the fact that optimizing:
{{< rawhtml >}}
$$

\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,x} \left[ \left\| u_t^\theta(x)-u_t(x) \right\|^2 \right] 

$$
{{< /rawhtml >}}
is equivalent to optimizing:
{{< rawhtml >}}
$$

\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{t,x_1,x} \left[ \left\| u_t^\theta(x)-u_t(x\mid x_1) \right\|^2 \right]

$$
{{< /rawhtml >}}


That's true because:
{{< rawhtml >}}
$$

\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{t,x_1,x} \left[ \left\| u_t^\theta(x)-u_t(x\mid x_1) \right\|^2 \right] = \mathbb{E}_{t,x_1,x} \left[ \left\| u_t^\theta(x) \right\|^2 + \left\| u_t(x|x_1) \right\|^2 - 2\langle u_t^\theta(x), u_t(x|x_1) \rangle  \right]

$$
{{< /rawhtml >}}

The first term is similar in both {{< rawhtml >}}$\mathcal{L}_{\mathrm{CFM}}${{< /rawhtml >}} and {{< rawhtml >}}$\mathcal{L}_{\mathrm{FM}}${{< /rawhtml >}}. The second term does not depend on {{< rawhtml >}}$\theta${{< /rawhtml >}}. Therefore, we can ignore them. Then:
{{< rawhtml >}}
$$

= \mathbb{E}_{t,x_1,x} \left[2\langle u_t^\theta(x), u_t(x|x_1) \rangle  \right] = \int_t \int_{x_1} \int_x  2 \langle u_t^\theta(x), u_t(x|x_1) \rangle p(x|x_1) p_\text{data}(x_1) dx dx_1 dt

$$
{{< /rawhtml >}}

Take the marginal vector field expression {{< rawhtml >}}$ u_t(x) = \int_{x_1} u_t(x \mid x_1) \frac{ p_t(x \mid x_1)\, p_{\mathrm{data}}(x_1) }{ p_t(x) } dx_1${{< /rawhtml >}}:

{{< rawhtml >}}
$$

= \int_t \int_x  2 \langle u_t^\theta(x), u_t(x) \rangle p(x) dx dt

$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$

= \mathbb{E}_{t,x} \left[2\langle u_t^\theta(x), u_t(x) \rangle  \right]

$$
{{< /rawhtml >}}

This makes the loss tractable:
{{< rawhtml >}}
$$

 \mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{t,x_1,x} \left[ \| u_t^\theta(x)-\underbrace{u_t(x\mid x_1)}_{x_1-x_0} \|^2 \right]

$$
{{< /rawhtml >}}

Now, what is {{< rawhtml >}}$u_t(x) = \int_{x_1} u_t(x \mid x_1) \frac{ p_t(x \mid x_1)\, p_{\mathrm{data}}(x_1) }{ p_t(x) } dx_1${{< /rawhtml >}}? Its a marginal vector field of the vector field which if we put our {{< rawhtml >}}$x_0 \sim \mathcal{N}(0,I)${{< /rawhtml >}} into it, they end up {{< rawhtml >}}$p_1 \sim p_\text{data}${{< /rawhtml >}} at {{< rawhtml >}}$t=1${{< /rawhtml >}}.

Why?


{{< rawhtml >}}
$$
p_t(x) = \int p_t(x|x_1) p_{\text{data}}(x_1) dx_1 \qquad \text{Marginal Probability Path}
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
\frac{\partial p_t(x)}{\partial t} = \int \frac{\partial p_t(x|x_1)}{\partial t} p_{\text{data}}(x_1) dx_1
$$
{{< /rawhtml >}}

Given conditional continuity equation {{< rawhtml >}}$\frac{\partial p_t(x|x_1)}{\partial t} = - \nabla \cdot p_t(x|x_1)v(x|x_1)${{< /rawhtml >}}:
{{< rawhtml >}}
$$
\frac{\partial p_t(x)}{\partial t} = \int -\nabla \cdot [p_t(x|x_1) u_t(x|x_1)] p_{\text{data}}(x_1) dx_1
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot \int u_t(x|x_1) p_t(x|x_1) p_{\text{data}}(x_1) dx_1
$$
{{< /rawhtml >}}

{{< rawhtml >}}$p_t(x|x_1) p_{\text{data}}(x_1) = p(x_1|x) p_t(x)${{< /rawhtml >}}:

{{< rawhtml >}}
$$
\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot \left( p_t(x) \int u_t(x|x_1) p(x_1|x) dx_1 \right)
$$
{{< /rawhtml >}}

Therefore:
{{< rawhtml >}}
$$
u_t(x) = \int u_t(x|x_1) p(x_1|x) dx_1
$$
{{< /rawhtml >}}

Therfore, {{< rawhtml >}}$u_t(x) = \int u_t(x|x_1) p(x_1|x) dx_1${{< /rawhtml >}} is the true conditional vector field that 

Assume we only have one determinisitic {{< rawhtml >}}$x_1${{< /rawhtml >}}, and therefore our {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}} would be dirac distribution. One pathway from gaussian noise {{< rawhtml >}}$\mathcal{N}(0, I)${{< /rawhtml >}}  to dirac distribuion is {{< rawhtml >}}$\mathcal{N}(tx_1, (1-t)^2I)${{< /rawhtml >}}. If you put {{< rawhtml >}}$t${{< /rawhtml >}} equal to 0 and 1, you can verify it. That probabilty path implies that {{< rawhtml >}}$x_t = tx_1 + (1-t)x_0${{< /rawhtml >}}, where {{< rawhtml >}}$x_0 \sim \mathcal{N}(0, I)${{< /rawhtml >}}. Therefore, the vector field that gives that probablity path is:
{{< rawhtml >}}
$$
\frac{\partial x_t}{\partial t} = x_1 - x_0
$$
{{< /rawhtml >}}

To prove that the conditional vector field {{< rawhtml >}}$u_t(x|x_1)=\frac{x_1 - x}/{1-t}${{< /rawhtml >}} generates the conditional probability path {{< rawhtml >}}$p_t(x|x_1) \sim \mathcal{N}(tx_1, (1-t)^2I)${{< /rawhtml >}}, we must show they satisfy the continuity equation:

{{< rawhtml >}}
$$
p_t(x|x_1) = \frac{1}{(2\pi)^{d/2} (1-t)^d} \exp\left( -\frac{\|x - tx_1\|^2}{2(1-t)^2} \right)
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\ln p_t = C - d \ln(1-t) - \frac{\|x - tx_1\|^2}{2(1-t)^2}
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\frac{\partial p_t}{\partial t} = p_t \left[ \frac{d}{1-t} + \frac{z \cdot x_1 (1-t) - \|z\|^2}{(1-t)^3} \right] \qquad \text{where $z=x - tx_1$} \qquad (4)
$$
{{< /rawhtml >}}


And the right hand side {{< rawhtml >}}$-\nabla \cdot (p_t u_t)${{< /rawhtml >}}:
{{< rawhtml >}}
$$
-\nabla \cdot (p_t u_t) = -\left( u_t \cdot \nabla p_t + p_t \nabla \cdot u_t \right) \qquad (5)
$$
{{< /rawhtml >}}
where:
{{< rawhtml >}}
$$
\nabla \cdot u_t = \nabla \cdot \left( \frac{x_1 - x}{1-t} \right) = \frac{1}{1-t} \nabla \cdot (x_1 - x) = \frac{-d}{1-t}
$$
{{< /rawhtml >}}
And:
{{< rawhtml >}}
$$
\nabla p_t = p_t \nabla \ln p_t = p_t \nabla \left[ -\frac{\|x - tx_1\|^2}{2(1-t)^2} \right] = p_t \left[ -\frac{x - tx_1}{(1-t)^2} \right] = p_t \left[ -\frac{z}{(1-t)^2} \right]
$$
{{< /rawhtml >}}
And:
{{< rawhtml >}}
$$
u_t \cdot \nabla p_t = \left( x_1 - \frac{z}{1-t} \right) \cdot \left( p_t \frac{-z}{(1-t)^2} \right) = p_t \left[ \frac{-z \cdot x_1}{(1-t)^2} + \frac{\|z\|^2}{(1-t)^3} \right] \qquad \text{since $x = z + tx_1$ and $x_t = x_1 - \frac{z}{1-t}$}
$$
{{< /rawhtml >}}
Replace in {{< rawhtml >}}$(5)${{< /rawhtml >}}:
{{< rawhtml >}}
$$
-\nabla \cdot (p_t u_t) = -\left( p_t \left[ \frac{-z \cdot x_1}{(1-t)^2} + \frac{\|z\|^2}{(1-t)^3} \right] + p_t \left[ \frac{-d}{1-t} \right] \right)
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
-\nabla \cdot (p_t u_t) = p_t \left[ \frac{d}{1-t} + \frac{z \cdot x_1}{(1-t)^2} - \frac{\|z\|^2}{(1-t)^3} \right] \qquad (6)
$$
{{< /rawhtml >}}

You can see that both sides, {{< rawhtml >}}$(4)${{< /rawhtml >}} and {{< rawhtml >}}$(6)${{< /rawhtml >}}, are equal. Therefore, the vector field {{< rawhtml >}}$u_t(x|x_1)=\frac{x_1 - x}/{1-t}${{< /rawhtml >}} leads to the density {{< rawhtml >}}$p_t(x|x_1) \sim \mathcal{N}(tx_1, (1-t)^2I)${{< /rawhtml >}}.

Also, it could be shown that every single particle with the probabilty path of {{< rawhtml >}}$p_t(x|x_1) \sim \mathcal{N}(tx_1, (1-t)^2I)${{< /rawhtml >}} is induced by the vector field:
{{< rawhtml >}}
$$
x_t = (1-t)x_0 + t x_1
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\frac{dx_t}{dt} = x_1 - x_0
$$
{{< /rawhtml >}}
Substitute {{< rawhtml >}}$x_0 = \frac{x_t - tx_1}{1-t}${{< /rawhtml >}}:
{{< rawhtml >}}
$$
\frac{dx_t}{dt} = x_1 - \left( \frac{x - tx_1}{1-t} \right)
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\frac{dx_t}{dt} = u_t(x|x_1) = \frac{x_1 - tx_1 - x + tx_1}{1-t} = \mathbf{\frac{x_1 - x}{1-t}}
$$
{{< /rawhtml >}}




### Training

So the training process will be become to sample a {{< rawhtml >}}$x_0 \sim \mathcal{N}(0, I)${{< /rawhtml >}}, and for timestep {{< rawhtml >}}$t${{< /rawhtml >}}, construct the noisy image {{< rawhtml >}}$x_t = (1-t)x_0 + tx_1${{< /rawhtml >}}, and use the noisy image to predict {{< rawhtml >}}$x_1 - x_0${{< /rawhtml >}} which is actaull {{< rawhtml >}}$u_t(x|x_1)${{< /rawhtml >}}

### Inference

Inference procedure is to sample {{< rawhtml >}}$x_0 \sim \mathcal{N}(0, I)${{< /rawhtml >}}, and use the learned vector field {{< rawhtml >}}$u^\theta_t${{< /rawhtml >}}, to construct the image at timestep {{< rawhtml >}}$t_i${{< /rawhtml >}}: {{< rawhtml >}}$x_{t_i} = x_{t_{i-1}} + u_{t_{i-1}}^{\theta}(x_{t_{i-1}})(t_i - t_{i-1})${{< /rawhtml >}}



## Latent space

Working directly with image representation have multiple problems: 
- **high dimensionality**
- **redundant information:** an image of a green apple may include many neighboring pixels with nearly identical green values
- **sparsity:** meaningful images occupy only a small and sparse subset of the overall image space. As a result, small perturbations in pixel space can move an image away from the manifold of realistic images, producing blurry or semantically meaningless outputs. The sparse distribution is hard to learn for generative models trying to learn {{< rawhtml >}}$p_\text{data}${{< /rawhtml >}}.

Therefore we introduce a latent space with rerduced dimension to work with. However, with a deterministic latent space, like in autoencoders, we don've any mean to enforce the latent representation to stay close and do not create a big space with sparse spikes. So, we introduce variational autoencoders, in which model lears to guess the distribution of the latent space (rather that the deterministic values). Therefore, we assume the latent space has the marginal distribution {{< rawhtml >}}$z \sim \mathcal{N}(0, I)${{< /rawhtml >}} and our encoder tries to guess parameters of {{< rawhtml >}}$q_\varphi(\cdot \mid x) = \mathcal{N}\left(\mu_\varphi(x), \sigma_\varphi^2(x)\right)${{< /rawhtml >}} and our decoder, the parametrs of {{< rawhtml >}}$p_\theta(. \mid z) = \mathcal{N}(\mu_\theta(z), \sigma_\theta^2(z))${{< /rawhtml >}} where {{< rawhtml >}}$\sigma_\varphi^2(x)${{< /rawhtml >}} and {{< rawhtml >}}$\sigma_\theta^2(z)${{< /rawhtml >}} are set to {{< rawhtml >}}$\sigma^2(z)I${{< /rawhtml >}} for simplicity and the models only try to guess the means. During the trainig, the latent variable will be sampled from {{< rawhtml >}}$q_\varphi(z \mid x)${{< /rawhtml >}}.

The loss can be obtained from the fact that all we want our model to do is to maximize the probability of generating the realsitic images, and therefore, maximzing the probability it asigns to the real image {{< rawhtml >}}$p_\theta(x)${{< /rawhtml >}}. So:
{{< rawhtml >}}
$$
p_\theta(x) = \int p_\theta(x, z) \, dz
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
p_\theta(x) = \int \frac{p(z) p_\theta(x|z)}{q_\phi(z|x)} q_\phi(z|x) \, dz
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\log p_\theta(x) = \log \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \frac{p(z) p_\theta(x|z)}{q_\phi(z|x)} \right]
$$
{{< /rawhtml >}}

However, that is computationally expensive, becuase there are a continous range of {{< rawhtml >}}$z${{< /rawhtml >}}, which are high dimensional vectors. So, we can define and ELBO:

{{< rawhtml >}}
$$
\log p_\theta(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \left( \frac{p(z) p_\theta(x|z)}{q_\phi(z|x)} \right) \right]
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
= \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{q_\phi(z|x)}{p(z)} \right]
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - \text{KL}[q_\phi(z|x) \| p(z)] = \text{ELBO} 
$$
{{< /rawhtml >}}

Therfore the VAE loss would be:
{{< rawhtml >}}
$$
\mathcal{L}_{\text{VAE}} = \underbrace{-\mathbb{E}_{z}[\log p_\theta(x|z)]}_{\substack{\mathcal{L}_{\text{rec}} \\ \text{\textbf{Reconstruction}}}} + \underbrace{\text{KL}(q_\varphi(z|x) \| p(z))}_{\substack{\mathcal{L}_{\text{KL}} \\ \text{\textbf{Regularization of}} \\ \text{\textbf{latent space}}}}
$$
{{< /rawhtml >}}

The term {{< rawhtml >}}$\text{KL}(q_\varphi(z|x) \| p(z))${{< /rawhtml >}} forces the latent space toward {{< rawhtml >}}$p(z) \sim \mathcal{N}(0, I)${{< /rawhtml >}} and therefore prevents sparse spikes.


The problem at this is point is that the above loss doesn't enforce truthfulness. That's because the first term reduces to {{< rawhtml >}}$\|x - \hat{x}\|^2${{< /rawhtml >}} which is the {{< rawhtml >}}$\ell_2 \text{ norm}${{< /rawhtml >}} of the pixel-wise difference between the actual image and the predicted image (the output of the decoder is mean of {{< rawhtml >}}$p_\theta(. \mid z) = \mathcal{N}(\mu_\theta(z), \sigma_\theta^2(z))${{< /rawhtml >}}). That penalizes the small difference in pixels very much which pushes the model toward the "safe" solution of choosing the average pixel values which make the image look blurry. And the structure of VAEs further exacerbates this issue, since both the latent representation and the reconstructed image are probabilistic rather than deterministic. 

One strategy is that instead of penalizing the pixel-to-pixel differece between the target and the decoder output, we penalize the {{< rawhtml >}}$\ell_2 \text{ norm}${{< /rawhtml >}} of the pixel-wise difference between the the feature maps of the actual image and the decoder model output:
{{< rawhtml >}}
$$
\mathcal{L}_{\mathrm{perc}} = \sum_l \frac{1}{H_l W_l} \left\| w^l \odot \left( \phi_l(x) - \phi_l(\hat{x}) \right) \right\|^2
$$
{{< /rawhtml >}}
(*{{< rawhtml >}}$w^l${{< /rawhtml >}}'s are tuned to match the human perception of the image*)
The feature maps are more closely correlated with the semantics of an image than the raw pixels. However if we choose {{< rawhtml >}}$\lambda_\text{perc}${{< /rawhtml >}} too high, the model tries to perfectly replicate the feature maps of the underlying network, which is a CNN, and looks like "checkerboards artifact"


Another strategy to mitigate blurriness is to use a discriminator model to push the generator toward producing more meaningful images. This adds an adversarial loss.

Therefore, {{< rawhtml >}}$\mathcal{L}_{\mathrm{VAE}}${{< /rawhtml >}} becomes:
{{< rawhtml >}}
$$
 \mathcal{L}_{\mathrm{VAE}} = \lambda_{\mathrm{rec}} \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{KL}} \mathcal{L}_{\mathrm{KL}} + \underbrace{\lambda_{\mathrm{perc}} \mathcal{L}_{\mathrm{perc}} + \lambda_{\mathrm{adv}} \mathcal{L}_{\mathrm{adv}}}_{\text{mitigate blurriness}}
$$
{{< /rawhtml >}}

### Training
(assume encoder and decoder are already trained separately)

Now we can apply our diffusion model in the latent space, since it does not suffer from the issues mentioned above for the pixel space:
- sample an image {{< rawhtml >}}$x_1 \sim p_\text{data}${{< /rawhtml >}}
- pass the image to the encoder and sample {{< rawhtml >}}$z_1 \sim q_\varphi(z \mid x) = \mathcal{N}\left(\mu_\varphi(x), \sigma_\varphi^2(x)\right)${{< /rawhtml >}}
- sample {{< rawhtml >}}$z_0 \sim p_0 = \mathcal{N}(0, I)${{< /rawhtml >}}
- for flow matching: obtain {{< rawhtml >}}$z_t = tz_1 + (1 - t)z_0${{< /rawhtml >}}.
- for flow matching: optimize the loss based of matching the velocity: {{< rawhtml >}}$\mathcal{L} = \|u_t^\theta(z_t) - (z_1 - z_0)\|^2${{< /rawhtml >}}



### Inference

- sample {{< rawhtml >}}$z_0 \sim p_0 = \mathcal{N}(0, I)${{< /rawhtml >}}
- flow matching: solve the ODE and arrive at {{< rawhtml >}}$z_1${{< /rawhtml >}}
- use the VAE decoder to arive at {{< rawhtml >}}$x_1${{< /rawhtml >}}


## Conditioning on text

But how do those paradigms integrate text? 

First, we need to make sure that similar text and images have similar embeddings. For this, we train image and text encoders using contrastive learning: we calculate the embedding of the image (CLS token) and the prompt (the last token), compute cosine similarity, and turn this similarities to probabilities through softmax. More generally, for an image {{< rawhtml >}}$i${{< /rawhtml >}} and a set of text candidates {{< rawhtml >}}${t_1,\dots,t_n}${{< /rawhtml >}}:

{{< rawhtml >}}
$$
P(t_k \mid i) = \frac{ \exp\left(s(i,t_k)\right) }{ \sum_{j=1}^{n} \exp\left(s(i,t_j)\right)}
$$
{{< /rawhtml >}}
Where
{{< rawhtml >}}
$$
s(i,t) = \frac{u_i^\top v_t}{|u_i|,|v_t|}
$$
{{< /rawhtml >}}


Then we use CLIP-style training: Suppose in a batch you have {{< rawhtml >}}$N${{< /rawhtml >}} image-text pairs: {{< rawhtml >}}$(I_1,T_1), (I_2,T_2), \dots, (I_N,T_N)${{< /rawhtml >}}. Then for each image {{< rawhtml >}}$I_i${{< /rawhtml >}}, it predicts which text is correct using softmax above. Therefore for each image {{< rawhtml >}}$i${{< /rawhtml >}} we'll have a series of probabilities:
{{< rawhtml >}}
$$
P(T_1 | I_i), P(T_2 | I_i), \dots, P(T_3 | I_N)
$$
{{< /rawhtml >}}

We want to maximize {{< rawhtml >}}$P(T_i | I_i)${{< /rawhtml >}} for all images, {{< rawhtml >}}$i${{< /rawhtml >}}'s.

So the loss for image-to-text matching is:

{{< rawhtml >}}
$$
\mathcal{L}_{\text{img}} = -\frac{1}{N} \sum_{i=1}^{N} \log P(T_i \mid I_i)
$$
{{< /rawhtml >}}

CLIP also does the reverse direction (text-to-image):

{{< rawhtml >}}
$$
P(I_j \mid T_i) = \frac{\exp(s_{ji}/\tau)} {\sum_{k=1}^{N}\exp(s_{ki}/\tau)}
$$
{{< /rawhtml >}}

with loss:

{{< rawhtml >}}
$$
\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{i=1}^{N} \log P(I_i \mid T_i)
$$
{{< /rawhtml >}}

Final training loss:

{{< rawhtml >}}
$$
\mathcal{L} = \frac{ \mathcal{L}_{\text{img}} + \mathcal{L}_{\text{text}} }{2}
$$
{{< /rawhtml >}}

In this form, the high probability of nonmatching pairs don't get diretly penalized. To remedy this, we can change the objective into trying to correctly guess whether two image and text pair match or not. And the loss function will be sigmoid.

Now that we have the right encoder models, we use them to guide the generative model toward correct output. Therefore the goal becomes to correctly guess:
{{< rawhtml >}}
$$
p(x_t \mid x_{t+1}) \;\longrightarrow\; p(x_t \mid x_{t+1}, y)
$$
{{< /rawhtml >}}

We can use a classifier weights to guide our {{< rawhtml >}}$p_{\theta, \phi}(x_t \mid x_{t+1}, y)${{< /rawhtml >}}, where {{< rawhtml >}}$\theta${{< /rawhtml >}} is the generator and {{< rawhtml >}}$\phi${{< /rawhtml >}} is the classifier weights. But {{< rawhtml >}}$p_{\theta, \phi}(x_t \mid x_{t+1}, y)${{< /rawhtml >}} has the same distribution as: 
{{< rawhtml >}}
$$
p_\theta(x_t \mid x_{t+1}, y)p_\phi(y \mid x_{t}).
$$
{{< /rawhtml >}}

- **First term**:  
    We know:
    {{< rawhtml >}}
$$
p_\theta(x_t \mid x_{t+1}, y) \sim \mathcal{N}(\mu_\theta, \sigma_{t+1}I) \qquad \text{DDPM generation process}
$$
{{< /rawhtml >}}
    Therefore:
    {{< rawhtml >}}
$$
 \log p_\theta(x_t \mid x_{t+1}) = -\frac{\|x_t - \mu_\theta\|^2}{2\sigma_{t+1}^2} + \text{constant}
$$
{{< /rawhtml >}}
- **Second term**:  
    We want to change {{< rawhtml >}}$p_\phi(y|x_t)${{< /rawhtml >}} to a form that keeps {{< rawhtml >}}$p_\theta(x_t \mid x_{t+1}, y)p_\phi(y \mid x_{t})${{< /rawhtml >}}.  
    Using first order taylor expansion (A Gaussian distribution's log-probability is a quadratic function (it looks like {{< rawhtml >}}$-(x-\mu)^2${{< /rawhtml >}}). If we use a complex, non-linear neural network for {{< rawhtml >}}$p(y|x_t)${{< /rawhtml >}}, adding its log-probability to our Gaussian would result in a very "messy" distribution that is no longer Gaussian. By using the first-order Taylor approximation, we treat the classifier's log-likelihood as a linear function of {{< rawhtml >}}$x_t${{< /rawhtml >}} locally):
    {{< rawhtml >}}
$$
\log p_\phi(y|x_t) \approx (x_t - \mu)^T \nabla_{x_t} \log p_\phi(y|\mu) + \text{constant}
$$
{{< /rawhtml >}}

Therefore, the conditional distribution of {{< rawhtml >}}$x_t${{< /rawhtml >}} is:
{{< rawhtml >}}
$$
x_t \sim \mathcal{N}(\mu_\theta + \sigma_{t+1}^2 \nabla_x \log p_\phi(y|\mu_\theta), \sigma_{t+1}^2 I)
$$
{{< /rawhtml >}}

To make the image to follow the classifier gradeints more rigourously, we introduce a parameter {{< rawhtml >}}$w${{< /rawhtml >}} usually {{< rawhtml >}}$>1${{< /rawhtml >}}:
{{< rawhtml >}}
$$
x_t \sim \mathcal{N}(\mu_\theta + w \sigma_{t+1}^2 \nabla_{x_t} \log p_\phi(y|\mu_\theta), \sigma_{t+1}^2 I)
$$
{{< /rawhtml >}}


One downside of the classifier guidance method is that, this procedure requires a classification model for {{< rawhtml >}}$\sigma_{t+1}^2 \nabla_x \log p_\phi(y|\mu_\theta)${{< /rawhtml >}} part. And that classifier should be able to classify the noisy images. So you need a freshly trained classifier on the noisy images. However, we know ({{< rawhtml >}}$i${{< /rawhtml >}} for implicit):
{{< rawhtml >}}
$$
p^i(y|x_t) \quad \propto \quad \frac{p(x_t|y)}{p(x_t)}
$$
{{< /rawhtml >}}
Therefore:
{{< rawhtml >}}
$$
\log p(x_t|y) = \log p(x_t) + \log p(y|x_t) + \text{const}
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\underbrace{\nabla_{x_t} \log p(x_t|y)}_{\text{Total Direction}} = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{Original Direction}} + \underbrace{\nabla_{x_t} \log p(y|x_t)}_{\text{Classifier "Push"}}
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
-\frac{\tilde{\epsilon}}{\sigma_t} = -\frac{\epsilon_\theta(x_t)}{\sigma_t} + \nabla_{x_t} \log p(y|x_t)
$$
{{< /rawhtml >}}
(where {{< rawhtml >}}$x_t =  \sqrt{\bar{\alpha}_t}x_0 + \sigma_t\epsilon${{< /rawhtml >}} for example and {{< rawhtml >}}$\tilde{\epsilon}${{< /rawhtml >}} is our new "guided" noise prediction).
{{< rawhtml >}}
$$
\tilde{\epsilon} = \epsilon_\theta(x_t) - \sigma_t \nabla_{x_t} \log p(y|x_t)
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\text{Guided Noise} = \epsilon_\theta(x_t) - w \sigma_t \nabla_{x_t} \log p_\phi(y|x_t)
$$
{{< /rawhtml >}}

Therefore, using the implicit classifier formula above:

classifier-based  
{{< rawhtml >}}
$$
 \epsilon_\theta(x_t) - w\sigma_t\nabla_{x_t} \log p_\phi(y|x_t)
$$
{{< /rawhtml >}}

classifier-free  
{{< rawhtml >}}
$$
\epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t, \emptyset))
$$
{{< /rawhtml >}}


**Training procedure for Classifier-Free Guidance (CFG)**:

1. sample {{< rawhtml >}}$x_0 \sim p_{\text{data}}, y${{< /rawhtml >}} pair, noise {{< rawhtml >}}$\epsilon \sim \mathcal{N}(0, I)${{< /rawhtml >}}, time step {{< rawhtml >}}$t \sim \mathcal{U}(0, T)${{< /rawhtml >}}, and create the noisy image {{< rawhtml >}}$x_t = \alpha_t x_0 + \sigma_t \epsilon.${{< /rawhtml >}}
2.
    *   **With probability {{< rawhtml >}}$(1 - p_{\text{uncond}})${{< /rawhtml >}}:** Keep the label {{< rawhtml >}}$y${{< /rawhtml >}}.
    *   **With probability {{< rawhtml >}}$p_{\text{uncond}}${{< /rawhtml >}}:** Replace the label {{< rawhtml >}}$y${{< /rawhtml >}} with a special **null token** {{< rawhtml >}}$\emptyset${{< /rawhtml >}} (e.g., an empty string or a vector of zeros).
3. Use {{< rawhtml >}}$x_t${{< /rawhtml >}} and {{< rawhtml >}}$t${{< /rawhtml >}} to predict {{< rawhtml >}}$\epsilon${{< /rawhtml >}} via {{< rawhtml >}}$\epsilon_\theta(x_t, y)${{< /rawhtml >}}
4. Compute loss {{< rawhtml >}}$\mathcal{L} = \|\epsilon_\theta(x_t, y) - \epsilon\|^2${{< /rawhtml >}} and backpropagate through {{< rawhtml >}}$\epsilon_\theta${{< /rawhtml >}}
(*Best results with {{< rawhtml >}}$w > 1${{< /rawhtml >}} and {{< rawhtml >}}$p_{\text{uncond}} = 10 - 20\%${{< /rawhtml >}}*)


## DiT

**End to end example (flow matching)**
- Sample noise from VAE latent space {{< rawhtml >}}$z_0 \sim mathcal{N}(0, I)${{< /rawhtml >}}.
- Divide the image into {{< rawhtml >}}$p \times p${{< /rawhtml >}} patches. (here we'll have {{< rawhtml >}}$\frac{I^2}{p^2}${{< /rawhtml >}} number of patches, which also be the number of sequence)
- Turn each patch into an embedding of dimension {{< rawhtml >}}$d${{< /rawhtml >}}.
- Turn timesteps and the condition to their corresponding embeddings and add them. 
- Now we need to introduce the condition embedding to the patches embeddings, but there are many way to do so, including:
    -  **original DiT**:  adaptive modulation of representation with learned gate {{< rawhtml >}}$\alpha${{< /rawhtml >}}, scale {{< rawhtml >}}$\gamma${{< /rawhtml >}}, and shift {{< rawhtml >}}$\beta${{< /rawhtml >}} from the condition embedding (passed to an MLP): {{< rawhtml >}}$x \leftarrow x + \alpha \ast \mathrm{Operation}\big(\mathrm{LN}(x) \ast (1+\gamma) + \beta\big)${{< /rawhtml >}}
        - but this has the problem that it applies the same modulation to all patches, which is a weakness. Thefore, people suggested the methods below
    - **MM-DiT**: cross-attention: using {{< rawhtml >}}$Q${{< /rawhtml >}} of patch embeddings and {{< rawhtml >}}$K${{< /rawhtml >}} and {{< rawhtml >}}$V${{< /rawhtml >}} of condition embeddings. (The attention output is then added back to the patch embeddings through a residual (skip) connection.)
    - **MM-DiT**: joint attention : consider both patch and condition embeddings as input.   
- Doing the previous step around self-attention and FFNN layers
- Output the {{< rawhtml >}}$u_\theta(z,t,c)${{< /rawhtml >}}
- Calculate new {{< rawhtml >}}$z_{t_i}${{< /rawhtml >}} : {{< rawhtml >}}
$$
z_{0+\Delta t} = z_0 + \underbrace{u_\theta(z_0, 0, c)} \Delta t
$$
{{< /rawhtml >}}
- Repeat the process to obtain {{< rawhtml >}}$u_\theta(z_{t_i}, t_i, c)${{< /rawhtml >}}
- Repeat the process to reach at {{< rawhtml >}}$z_1${{< /rawhtml >}} ({{< rawhtml >}}$t${{< /rawhtml >}} reaches 1).
- Pass the {{< rawhtml >}}$z_1${{< /rawhtml >}} through VAE decoder to obtain the image. 



### sampling noise

At the beginning timesteps, it is only necessary to know roughly where {{< rawhtml >}}$p_{\text{data}}${{< /rawhtml >}} lies. Near the end, the image is already nearly complete, making it relatively easy to remove the remaining noise. Therefore, tasks near the two endpoints are easier compared to those at the middle timesteps. For this reason, instead of sampling {{< rawhtml >}}$t${{< /rawhtml >}} from {{< rawhtml >}}$\mathcal{U}(0,1)${{< /rawhtml >}}, people sample it from a distribution with higher density around the middle timesteps and support over {{< rawhtml >}}$(0,1)${{< /rawhtml >}}. One such choice is the logit-normal distribution. 

If 

{{< rawhtml >}}
$$
X=\log\left(\frac{T}{1-T}\right),\quad X\sim\mathcal{N}(\mu,\sigma^2)
$$
{{< /rawhtml >}}

then (T) follows a logit-normal distribution with pdf of:

{{< rawhtml >}}
$$
f_T(t)=\frac{1}{t(1-t)\sigma\sqrt{2\pi}}\exp\left[-\frac{(\log(\frac{t}{1-t})-\mu)^2}{2\sigma^2}\right],\quad 0 &lt; t &lt; 1
$$
{{< /rawhtml >}}



### adjustment wrt resolution

Adding the same amount of noise to an already noisy image causes a larger loss of information than adding the same amount of noise to a high-resolution image. So, people adjust the noise level such that it is perceived similarly in a low-resolution image and a high-resolution one. The factor represnting the perceived noise is the variance of the average noise over total number of the image pixels. That's because a lower variance of the average noise across those pixels, making the image look "cleaner" even if the per-pixel noise level remains the same. 

Suppose we have two images {{< rawhtml >}}$\text{img}_m \in \mathbb{R}^{H \times W}${{< /rawhtml >}}, and {{< rawhtml >}}$\text{img}_n \in \mathbb{R}^{h \times w}${{< /rawhtml >}}, where {{< rawhtml >}}$m = H \times W${{< /rawhtml >}}, {{< rawhtml >}}$n = h \times w${{< /rawhtml >}}, and {{< rawhtml >}}$n < m${{< /rawhtml >}}. We want to make the perceived noise levels (the variance of the average noise) of both images equal. For that we need a higher noise level for {{< rawhtml >}}$\text{img}_m${{< /rawhtml >}} than for {{< rawhtml >}}$\text{img}_n${{< /rawhtml >}}. 

Here, suppose {{< rawhtml >}}$t=0${{< /rawhtml >}} and {{< rawhtml >}}$t=1${{< /rawhtml >}} are corresponding to the clean and noisy image, respectively. Take {{< rawhtml >}}$c${{< /rawhtml >}} as the original value of all pixels (all the pixels have the same value) and {{< rawhtml >}}$z_{i, t} = (1-t)c + t\epsilon${{< /rawhtml >}} when {{< rawhtml >}}$\epsilon \sim \mathcal{N}(0, I)${{< /rawhtml >}} as the noisy {{< rawhtml >}}$i\text{th}${{< /rawhtml >}} pixel value at time {{< rawhtml >}}$t${{< /rawhtml >}}. The average noisy pixel value, where the total pixel number is {{< rawhtml >}}$m${{< /rawhtml >}}, is:
{{< rawhtml >}}
$$
\bar{z_t} = \frac{1}{m}\sum_{i=1}^{m} z_{i,t} = (1-t)c + \frac{t}{m}\sum_{i=1}^{m}\epsilon_i
$$
{{< /rawhtml >}}
Therefore: 
{{< rawhtml >}}
$$
\bar{z_t} \sim \mathcal{N}((1-t)c\,, \frac{t^2}{m})
$$
{{< /rawhtml >}}

We can estimate the original pixel value {{< rawhtml >}}$c${{< /rawhtml >}} using the unbiased estimator {{< rawhtml >}}$\hat{c} = \frac{1}{1-t}\bar{z_t}${{< /rawhtml >}}. Then:
{{< rawhtml >}}
$$
\text{Var}(\hat{c}) = \frac{1}{(1-t)^2}\frac{t^2}{m}
$$
{{< /rawhtml >}}


Therefore, if we set the variance of the average pixel values of the two images to be equal:
{{< rawhtml >}}
$$
\frac{1}{(1-t_n)^2}\frac{t_n^2}{n} = \frac{1}{(1-t_m)^2}\frac{t_m^2}{m}
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
\frac{1}{(1-t_n)}\frac{t_n}{\sqrt{n}} = \frac{1}{(1-t_m)}\frac{t_m}{\sqrt{m}}
$$
{{< /rawhtml >}}
let {{< rawhtml >}}$S = \sqrt{\frac{m}{n}}${{< /rawhtml >}}:
{{< rawhtml >}}
$$
\frac{S \cdot t_n}{1-t_n} = \frac{t_m}{1-t_m}
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
S t_n (1 - t_m) = t_m (1 - t_n)
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
S t_n - S t_n t_m = t_m - t_n t_m
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
S t_n = t_m - t_n t_m + S t_n t_m
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
S t_n = t_m (1 + (S - 1)t_n)
$$
{{< /rawhtml >}}
{{< rawhtml >}}
$$
t_m = \frac{\sqrt{\frac{m}{n}} t_n}{1 + (\sqrt{\frac{m}{n}} - 1)t_n}
$$
{{< /rawhtml >}}

The last expression is the adjustment applied to the timestep for a new image {{< rawhtml >}}$\text{img}_m${{< /rawhtml >}}, ensuring its perceived noise matches what the model learned at the training resolution {{< rawhtml >}}$n${{< /rawhtml >}}.  

{{< rawhtml >}}<br>{{< /rawhtml >}}

If you believe something should be added or if you notice any mistakes in this post, please don’t hesitate to reach out to me at [ad dot vafaeian at gmail dot com]. I will address any issues promptly. :)

{{< rawhtml >}}<br>{{< /rawhtml >}}

## References

- Afshine Amidi and Shervine Amidi. "CME 296: Deep Generative Models." Stanford University. [https://cme296.stanford.edu/](https://cme296.stanford.edu/).

- Pascal Vincent. "A Connection Between Score Matching and Denoising Autoencoders." Technical Report 1356, Université de Montréal, 2010. [PDF](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)



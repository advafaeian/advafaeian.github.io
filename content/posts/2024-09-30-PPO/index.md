---
title: 'Proximal Policy Optimization' 
date: 2024-09-30T17:38:50+03:30
draft: false
tags: ["policy gradient", "actor-critic", "reinfrocement learning", "proximal policy optimization"]
description: "How does proximal policy optimization work?"
canonicalURL: "https://advafaeian.github.io/2024-09-30-PPO/"
cover:
    image: "cover.jpg" # image path/url
    alt: "Retro gaming" # alt text
    caption: "Photo by Lander Denys on Unsplash"
    relative: true
github: "https://github.com/advafaeian/proximal-policy-optimization"    
---Proximal Policy Optimization (PPO) is a model-free, on-policy reinforcement learning algorithm introduced by OpenAI in 2017. It aims to improve the stability and efficiency of policy gradient methods while maintaining simplicity. Its key advantages include good performance across a wide range of tasks, ease of implementation, and compatibility with both continuous and discrete action spaces. Since its introduction, PPO has become a standard baseline in reinforcement learning research and has been successfully applied to complex problems such as training AI agents to play video games and controlling robotic systems. It is also used in ChatGPT for Reinforcement Learning from Human Feedback (RLHF).

## Source
- #### [Github Repository](https://github.com/advafaeian/proximal-policy-optimization)
- #### [Notebook](https://github.com/advafaeian/proximal-policy-optimization/PPO.ipynb)

Here, we attempt to use PPO to train a neural network to land the [Gymnasium's 'Lunar Lander'](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment. I explain each step and function gradually as we move forward, starting with the importation of basic modules.


```python
import os  # For file operations related to video files
import random  # For seeding random number generation
import time  # To generate distinct filenames
import numpy as np
import tensorflow as tf
import keras
import gymnasium as gym

```

    2024-09-30 21:07:58.900408: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


## Configuring Global Variables and Training Parameters

Now, we set some necessary global variables.

We will let the environments run for a predefined number of steps to fill the replay buffer, and then use the data from this buffer to train the neural network.

In each episode, we execute 128 (`num_steps`) consecutive steps. With 4 (`num_envs`) separate environments, our batch size becomes 512 (`batch_size`). Additionally, gradient descent is performed in 4 (`num_mbatch`) separate mini-batches, each consisting of 128 (`mbatch_size`) non-consecutive steps. Each batch update is applied 4 (`update_epochs`) times. The training process will terminate after 2e6 (`total_timesteps`) steps.

Other variables are [set](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) according to the official PPO repository.


```python
seed = 42
num_envs = 4
num_steps = 128
batch_size = num_envs * num_steps
num_mbatch = 4
mbatch_size = int(batch_size // num_mbatch)
total_timesteps = 2e6
num_epochs = int(total_timesteps // batch_size)
update_epochs = 4

lr_rate = 2.5e-4
eps = 1e-5
gamma = .99
gae_lambda = .95
clip_coef = .2
ent_coef = .01
vf_coef = .5
max_grad_norm = .5
```

Setting seeds for consistent resutls.


```python
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)
```

## Generating Unique Run Names

A function that generates unique run names for TensorBoard logs and video file saving. This ensures that each run is distinct and easily identifiable.


```python
from time import strftime
def get_run_logdir(root_logdir="."):
    """
    Generates a unique run name based on the current system time.

    Args:
        root_logdir (str): The root parent directory for TensorBoard runs

    Returns:
        str: The full relative path to the run directory
    """
    return root_logdir + "/" + strftime("run_%Y_%m_%d_%H_%M_%S")
get_run_logdir()
```




    './run_2024_09_30_21_08_08'



## Setting Up Lunar Lander Environments with Video Recording

[Lunar Lander 2](https://gymnasium.farama.org/environments/box2d/lunar_lander/) has an 8-dimensional observation space, which includes the lander's x and y coordinates, linear velocities in the x and y directions, the angle, angular velocity, and two boolean indicators for whether each leg is in contact with the ground. The action space consists of 4 discrete actions: do nothing, fire the left orientation engine, fire the main engine, and fire the right orientation engine.

As mentioned earlier, we are using four environments that operate independently. The `gym.vector.SyncVectorEnv` function takes a list of functions, each of which returns another function that creates a Gymnasium environment. The `gym.wrappers.RecordVideo` wrapper enables direct video recording, with the parameter `episode_trigger=lambda x: x % 3 == 0` ensuring that video is recorded only once every three episodes.

A key feature is that the environments in `envs` automatically reset when an episode ends.


```python
def make_env(idx, run_dir):
    """
    Creates a 'LunarLander-v2' environment wrapped in video and statistics wrappers.

    Args:
        idx (int): The index of the current environment
        run_name (str): The name to be appended after 'videos_tf/' and used as the parent directory for videos

    Returns:
        function: A function that returns a single environment, to be passed to `gym.vector.SyncVectorEnv`
    """
    def thunk():
        env = gym.make("LunarLander-v2", render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0: # Only the first of four environments
            env = gym.wrappers.RecordVideo(env, run_dir, disable_logger=True, episode_trigger= lambda x: x % 3 == 0)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv(
        [
            make_env(i, get_run_logdir("videos"))
            for i in range(num_envs)
        ]
    )
```

## Actor-Critic Network Architecture and Initialization

The model follows an actor-critic pattern, where the actor head (policy function) determines the action to take, and the critic head (value function) estimates the value function of the current state. The value function of a state is the expected discounted sum of rewards from that state. Formally, it is expressed as {{< rawhtml >}}$V(s) = \mathbb{E} \left[ \sum \limits _{t=0}^{T-1} \gamma^t R(s_t, a_t) \right]${{< /rawhtml >}}. The input shape for this network is `(batch_size, number_of_observation_space_elements)`, with `batch_size` corresponding to `num_envs`, resulting in an input shape of `(4, 8)`. 

But why does the network also need to estimate the expected benefits of visiting a state? As we’ll explain later, understanding the value of our visits is essential for distinguishing a good policy from a bad one.

Orthogonal initialization is the primary method for weight initialization in the PPO repository and has [demonstrated superior results](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).


```python
def create_model(units, activation = "relu"):
    """
    Creates the actor-critic model.

    Args:
        units (list[int]): A list of integers specifying the number of units in each dense layer, where the length of the list determines the number of layers
        activation (str): The activation function used in the dense layers

    Returns:
        keras.Model: A model with an input shape of (None, 8) and two output heads: 
            - The value function head, which estimates the true value function
            - The policy function head, which determines the optimal action to take in the current state
    """
    units = list(units)
    initializer_1 = tf.keras.initializers.Orthogonal(gain=1)
    initializer_01 = tf.keras.initializers.Orthogonal(gain=.01)
    initializer_2s = tf.keras.initializers.Orthogonal(gain=tf.sqrt(2.))
    inputs = keras.layers.Input(shape=(np.array(envs.single_observation_space.shape).prod(),))
    x = keras.layers.Dense(units[0], activation=activation)(inputs)
    for num_units in units[-1:]:
        x = keras.layers.Dense(num_units, activation=activation, kernel_initializer=initializer_2s)(x)
    actor = keras.layers.Dense(envs.single_action_space.n, kernel_initializer=initializer_01, name="policy_function")(x)
    critic = keras.layers.Dense(1, kernel_initializer=initializer_1, name="value_function")(x)

    return keras.Model(inputs = inputs, outputs=[actor, critic])

model = create_model([1024, 1024], activation="relu")
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)      │      <span style="color: #00af00; text-decoration-color: #00af00">9,216</span> │ input_layer[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)      │  <span style="color: #00af00; text-decoration-color: #00af00">1,049,600</span> │ dense[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ policy_function     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)         │      <span style="color: #00af00; text-decoration-color: #00af00">4,100</span> │ dense_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ value_function      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │      <span style="color: #00af00; text-decoration-color: #00af00">1,025</span> │ dense_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │                   │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,063,941</span> (4.06 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,063,941</span> (4.06 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



This prints:

```Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 8)         │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (Dense)       │ (None, 1024)      │      9,216 │ input_layer[0][0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (Dense)     │ (None, 1024)      │  1,049,600 │ dense[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ policy_function     │ (None, 4)         │      4,100 │ dense_1[0][0]     │
│ (Dense)             │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ value_function      │ (None, 1)         │      1,025 │ dense_1[0][0]     │
│ (Dense)             │                   │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 1,063,941 (4.06 MB)
 Trainable params: 1,063,941 (4.06 MB)
 Non-trainable params: 0 (0.00 B)

## Handling Environment Steps and TensorFlow Integration

The `step` function takes actions for the four environments and returns the observations (current states), rewards, and whether each environment has terminated. Additionally, the function returns the episodic rewards and episode length for each environment that terminates. If any environment has not yet finished, the corresponding episodic rewards and episode length are set to zero.

The `envs.step` function does not return TensorFlow tensors, so it cannot be used directly in a function decorated with `@tf.function`. To use it in such functions, we must wrap it in a function that accepts tensors as parameters and also returns tensors. The `tf_step` function handles this.


```python
def step(action: np.ndarray):
    """
    Returns the state, reward, done flag, cumulative rewards in the episode, and episode length given an action, as numpy arrays.

    Args:
        action (np.int32): A numpy array of actions to be taken, shape: (num_envs,)

    Returns:
        state (np.float32): The observation after taking the action, shape: (num_envs, 8)
        reward (np.float32): The reward following the action, shape: (num_envs,)
        done (np.int32): A flag indicating whether the environment is terminated, shape: (num_envs,)
        r (np.float32): The cumulative rewards in the terminated episode, shape: ()
        l (np.int32): The length of the terminated episode, shape: ()
    """
    obs, reward, done, truncated, info = envs.step(action)

    ep_r = 0
    ep_l = 0
    for i in info:
        if i == "final_info":
            ep = info[i][info["_final_info"]][0]['episode']
            ep_r = ep["r"][0]
            ep_l = ep["l"][0]
    return (obs.astype(np.float32), 
            np.array(reward, np.float32), 
            np.array(done, np.int32),
            np.array(ep_r, np.float32),
            np.array(ep_l, np.int32))


def tf_step(action: tf.Tensor):
    """
    Wraps the step function in a TensorFlow-friendly function, taking and returning tensors.

    Args:
        action (tf.int32): The action to be taken, as a tf.Tensor.

    Returns:
        state (tf.float32): The observation after taking the action, shape: (4, 8).
        reward (tf.float32): The reward following the action, shape: (4,).
        done (tf.int32): A flag indicating whether the environment is terminated, shape: (4,).
        r (tf.float32): The cumulative rewards in the episode, shape: (,).
        l (tf.int32): The episode length, shape: (,).
    """
    return tf.numpy_function(step, [action], 
                            [tf.float32, tf.float32, tf.int32, tf.float32, tf.int32])
```

## Policy Gradient and Generalized Advantage Estimation

In policy gradient methods, including PPO, we aim to choose the parameters of our neural network to maximize the sum of rewards from our actions. More formally, given a trajectory {{< rawhtml >}}$\tau = (s_0, a_0, \dots, s_{T−1}, a_{T−1}, s_T)${{< /rawhtml >}}, we try to maximize {{< rawhtml >}}$f(\tau) = \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t)${{< /rawhtml >}}, which represents the discounted sum of rewards received along a path during our play.

But do we know which path we will take before starting the game? Of course not. Therefore, we must take the expectation over all possible trajectories and try to maximize {{< rawhtml >}}$\mathbb{E}_{\tau \sim P_\theta} \left[\sum_{t=0}^{T-1} \gamma^t R(s_t, a_t)\right] = \mathbb{E}_{\tau \sim P_\theta} \left[f(\tau)\right]${{< /rawhtml >}}.

To maximize this function through gradient ascent (or even descent), we need to calculate {{< rawhtml >}}$\nabla_\theta \mathbb{E}_{\tau \sim P_\theta} [f(\tau)]${{< /rawhtml >}}.

We can go further:

{{< rawhtml >}}
$$

\begin{align*}
    \nabla_\theta J(\theta) 
    &= \nabla_\theta \mathbb{E}_{\tau \sim P_\theta} [f(\tau)] \\
    &= \nabla_\theta \int P_\theta(\tau)f(\tau)\,d\tau \\ 
    &= \int \nabla_\theta(P_\theta(\tau) f(\tau))\,d\tau \qquad &&\text{\scriptsize(swap integration with gradient)} \\
    &= \int (\nabla_\theta P_\theta(\tau))f(\tau)\,d\tau \qquad &&\text{\scriptsize(since $f$ does not depend on $\theta$)}
\end{align*}

$$
{{< /rawhtml >}}

Since the probability of trajectories depends on {{< rawhtml >}}$\theta${{< /rawhtml >}} and includes transition probabilities to states following an action, which are determined through interaction with the environment and whose dynamics we do not know, we cannot directly evaluate {{< rawhtml >}}$\nabla_\theta P_\theta${{< /rawhtml >}}. However, by using elementary calculus and rearrangements, we can proceed as follows:

{{< rawhtml >}}
$$

\begin{align*}
\int (\nabla_\theta P_\theta(\tau))f(\tau)\,d\tau 
&= \int P_\theta(\tau)(\nabla_\theta \log P_\theta(\tau))f(\tau)\,d\tau \\
&= \mathbb{E}_{\tau \sim P_\theta} \left[(\nabla_\theta \log P_\theta(\tau))f(\tau)\right] \qquad &&\text{\scriptsize($\nabla_\theta \log P_\theta(\tau) = \frac{\nabla_\theta P_\theta(\tau)}{P_\theta(\tau)}$)}
\end{align*}

$$
{{< /rawhtml >}}

Considering: 
{{< rawhtml >}}
$$
P_\theta(\tau) = \mu(s_0) \pi_\theta(a_0 | s_0) P_{s_0 a_0}(s_1) \pi_\theta(a_1 | s_1) P_{s_1 a_1}(s_2) \cdots P_{s_{T-1} a_{T-1}}(s_T)
$$
{{< /rawhtml >}} 
Thus: {{< rawhtml >}}
$$
\nabla_\theta \log P_\theta(\tau) = \nabla_\theta \log \pi_\theta(a_0 | s_0) + \nabla_\theta \log \pi_\theta(a_1 | s_1) + \cdots + \nabla_\theta \log \pi_\theta(a_{T-1} | s_{T-1})
$$
{{< /rawhtml >}} 
We can express this as:
{{< rawhtml >}}
$$
\nabla_\theta \log P_\theta(\tau) = \nabla_\theta \log \pi_\theta(a_0 \mid s_0) + \nabla_\theta \log \pi_\theta(a_1 \mid s_1) + \cdots + \nabla_\theta \log \pi_\theta(a_{T-1} \mid s_{T-1})

$$
{{< /rawhtml >}}
This holds because the transition probabilities to the next state, given the current state and action, do not depend on {{< rawhtml >}}$\pi_\theta${{< /rawhtml >}}. Therefore, we can rewrite {{< rawhtml >}}$\nabla_\theta J(\theta)${{< /rawhtml >}} as:
{{< rawhtml >}}
$$
\begin{align*}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim P_\theta} \left[ f(\tau) \right] 
\\
&= \mathbb{E}_{\tau \sim P_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \right) \cdot f(\tau) \right] 
\\
&= \mathbb{E}_{\tau \sim P_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \right) \cdot \left( \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t) \right) \right]\\
&= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left( \sum_{j=0}^{T-1} \gamma^j R(s_j, a_j) \right) \right] \nonumber \\
&= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left( \sum_{j \geq t}^{T-1} \gamma^j R(s_j, a_j) \right) \right]
\end{align*}
$$
{{< /rawhtml >}}


The last equality follows from:

{{< rawhtml >}}
$$
\begin{align}
&\mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left( \sum_{0 \leq j < t} \gamma^j R(s_j, a_j) \right) \right]
\\ 
&= \mathbb{E} \left[ \mathbb{E} \left[\nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left( \sum_{0 \leq j < t} \gamma^j R(s_j, a_j) \right) \bigg| s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t \right] \right] 
\\
&= \mathbb{E} \left[ \mathbb{E} \left[\nabla_\theta \log \pi_\theta(a_t | s_t) \bigg| s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t \right] \cdot \left( \sum_{0 \leq j < t} \gamma^j R(s_j, a_j) \right) \right] 
\\
&= 0
\end{align}
$$
{{< /rawhtml >}}

Here’s a breakdown of the reasoning:

1. **Law of Total Expectation**: The equation in (2) follows from the law of total expectation (or "Adams's Law"), which states that {{< rawhtml >}}$\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]${{< /rawhtml >}}. This allows us to condition on the past trajectory {{< rawhtml >}}$(s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t)${{< /rawhtml >}} and take the expectation over the remaining trajectory.

2. **Separation of Known and Unknown Quantities**: In equation (3), we use the fact that {{< rawhtml >}}$\sum_{0 \leq j < t} \gamma^j R(s_j, a_j)${{< /rawhtml >}} is known given the trajectory up to time {{< rawhtml >}}$t${{< /rawhtml >}} (i.e., it depends only on {{< rawhtml >}}$(s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t)${{< /rawhtml >}}). Therefore, we can take it out of the inner expectation.

3. **Expectation of Gradient of Log PDF**: The equation (4) results from the fact that, given {{< rawhtml >}}$s_t${{< /rawhtml >}} and {{< rawhtml >}}$a_t${{< /rawhtml >}}, {{< rawhtml >}}$\pi_\theta(a_t \mid s_t)${{< /rawhtml >}} is still a valid probability density function (PDF). The expectation of the gradient of the log of a PDF is zero. This is because:
   
   {{< rawhtml >}}
$$

   \begin{align*}
   \mathbb{E} [\nabla \log f(x)] &= \int \nabla \log f(x) \, f(x) \, dx \\
   &= \int \frac{\nabla f(x)}{f(x)} \, f(x) \, dx \\
   &= \int \nabla f(x) \, dx \\
   &= \nabla \int f(x) \, dx \\
   &= \nabla 1 \\
   &= 0
   \end{align*}
   
$$
{{< /rawhtml >}}


### Adding a Baseline
So far, we have reached the equation:

{{< rawhtml >}}
$$

\nabla J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \left( \sum_{j \geq t}^{T-1} \gamma^j R(s_j, a_j) \right) \right]

$$
{{< /rawhtml >}}

To further refine our approach, we can add a baseline term {{< rawhtml >}}$-B(s_t)${{< /rawhtml >}} as follows:

{{< rawhtml >}}
$$

\begin{equation*} 
    \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta (a_t \mid s_t) \cdot \left( \sum_{j \geq t}^{T-1} \gamma^j R(s_j, a_j) - \gamma^t B(s_t) \right) \right]
\end{equation*}

$$
{{< /rawhtml >}}

This is still an equality because, similar to the previous steps, we can condition the inner expectation on {{< rawhtml >}}$\tau_{:t} = (s_0, a_0, s_1, a_1, \dots, s_t)${{< /rawhtml >}}, and thus {{< rawhtml >}}$-B(s_t)${{< /rawhtml >}} can be factored out of the inner expectation as it is known. Therefore, we get:

{{< rawhtml >}}
$$

\begin{align}
    &\sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta (a_t \mid s_t) \cdot \left( \sum_{j \geq t}^{T-1} \gamma^j R(s_j, a_j) - \gamma^t B(s_t) \right) \right] \\
    &= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta (a_t \mid s_t) \cdot \left( \sum_{j \geq t}^{T-1} \gamma^j R(s_j, a_j) \right) \right] \\ 
    &- \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta (a_t \mid s_t) \cdot \left( \gamma^t B(s_t) \right) \right]
\end{align}

$$
{{< /rawhtml >}}


The term 
{{< rawhtml >}}
$$

\mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \left( \gamma^t B(s_t) \right) \right]

$$
{{< /rawhtml >}}

is equal to {{< rawhtml >}}$0${{< /rawhtml >}} because:
{{< rawhtml >}}
$$

\begin{align*}
&\mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \left( \gamma^t B(s_t) \right) \right] \\
&= \mathbb{E} \left[ \mathbb{E} \left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \left( \gamma^t B(s_t) \right) \bigg| s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t \right] \right] \\
&= \mathbb{E} \left[ \mathbb{E} \left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \bigg| s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t \right] \cdot \left( \gamma^t B(s_t) \right) \right] \\
&= 0
\end{align*}

$$
{{< /rawhtml >}}

This result comes from the property that, given {{< rawhtml >}}$\nabla_\theta \log \pi_\theta(a_t \mid s_t)${{< /rawhtml >}} is independent of {{< rawhtml >}}$\gamma^t B(s_t)${{< /rawhtml >}}, the expectation of the product of {{< rawhtml >}}$\nabla_\theta \log \pi_\theta(a_t \mid s_t)${{< /rawhtml >}} and any term depending only on the state and action up to {{< rawhtml >}}$t${{< /rawhtml >}} is zero. This is due to the fact that {{< rawhtml >}}$\nabla_\theta \log \pi_\theta(a_t \mid s_t)${{< /rawhtml >}} is a gradient of a log probability, and thus has zero mean.

### Why Add a Baseline?

Adding a baseline, such as {{< rawhtml >}}$-B(s_t)${{< /rawhtml >}}, helps to reduce the variance of the policy gradient estimates. High variance in policy gradient updates can lead to unstable training. Large updates may cause the policy to diverge from optimality, resulting in poor performance and less effective exploration.

The variance of the gradient estimate can be reduced using a baseline because of the following property:

{{< rawhtml >}}
$$

\text{var}(A - B) = \text{var}(A) + \text{var}(B) - 2 \cdot \text{cov}(A, B)

$$
{{< /rawhtml >}}

By choosing a baseline {{< rawhtml >}}$B${{< /rawhtml >}} that is highly correlated with {{< rawhtml >}}$\sum_{j \geq t}^{T-1} \gamma^j R(s_j, a_j)${{< /rawhtml >}}, we reduce the covariance between {{< rawhtml >}}$\nabla_\theta \log \pi_\theta(a_t \mid s_t)${{< /rawhtml >}} and the return. In practice, the estimated value function of the current state, {{< rawhtml >}}$\hat{V}(s)${{< /rawhtml >}}, is often used as the baseline. If {{< rawhtml >}}$\hat{V}(s)${{< /rawhtml >}} is a good approximation of the expected return, it will be highly correlated with the total reward, reducing the variance of the gradient estimate and leading to more stable training.

### Substituting with $Q^{\pi_\theta}(s_t, a_t)$ 

First, a few notations:
{{< rawhtml >}}
$$

\begin{align*}
Q^{\pi_\theta}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[\sum_{j \geq t}^{T-1} \gamma^{j-t} R(s_j, a_j) \mid s_t, a_t \right] \\
\tau_{t:} &= (s_{t+1}, a_{t+1}, \ldots, s_{T-1}, a_{T-1}, s_T) \\
\tau_{:t} &= (s_0, a_0, \ldots, s_t, a_t)
\end{align*}

$$
{{< /rawhtml >}}


Thus, we can replace {{< rawhtml >}}$\sum_{j \geq t}^{T-1} \gamma^{j-t} R(s_j, a_j)${{< /rawhtml >}} with its expectation, {{< rawhtml >}}$Q^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}}. This is because:

{{< rawhtml >}}
$$

\begin{align*}
\nabla J(\theta) &= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \left( \sum_{j \geq t}^{T-1} \gamma^{j-t} R(s_j, a_j) \right) \right] \\
&= \sum_{t=0}^{T-1} \mathbb{E}_{\tau_{:t} \sim \pi_\theta} \left[ \mathbb{E}_{\tau_{t:} \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \left( \sum_{j \geq t}^{T-1} \gamma^{j-t} R(s_j, a_j) \right) \mid \tau_{:t} \right] \right] \qquad &&\text{\scriptsize(Adam's law)} \\
&= \sum_{t=0}^{T-1} \mathbb{E}_{\tau_{:t} \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \mathbb{E}_{\tau_{t:} \sim \pi_\theta} \left[ \left( \sum_{j \geq t}^{T-1} \gamma^{j-t} R(s_j, a_j) \right) \mid \tau_{:t} \right] \right] \qquad &&\text{\scriptsize(taking out the known quantities)} \\
&= \sum_{t=0}^{T-1} \mathbb{E}_{\tau_{:t} \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \mathbb{E}_{\tau_{t:} \sim \pi_\theta} \left[ \left( \sum_{j \geq t}^{T-1} \gamma^{j-t} R(s_j, a_j) \right) \mid s_t, a_t \right] \right] \\
&= \sum_{t=0}^{T-1} \mathbb{E}_{\tau_{:t} \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right] \qquad &&\text{\scriptsize(by the definition of $Q^{\pi_\theta}(s_t, a_t)$)}
\end{align*}

$$
{{< /rawhtml >}}



### Advantage Function

Following the **Adding a Baseline** section, we can subtract a baseline function that depends only on the state {{< rawhtml >}}$s${{< /rawhtml >}}, such as {{< rawhtml >}}$V^{\pi_\theta}(s_t) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{j \geq t}^{T-1} \gamma^{j-t} R(s_j, a_j) \mid s_t \right]${{< /rawhtml >}}. This yields the updated equation:

{{< rawhtml >}}
$$

\nabla J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{\tau_{:t} \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \left( Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t) \right) \right]

$$
{{< /rawhtml >}}

The term {{< rawhtml >}}$Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)${{< /rawhtml >}} is known as the **Advantage Function**, denoted by {{< rawhtml >}}$A^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}}:

{{< rawhtml >}}
$$

A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)

$$
{{< /rawhtml >}}

Thus, the gradient of the objective function {{< rawhtml >}}$\nabla J(\theta)${{< /rawhtml >}} can be expressed as:

{{< rawhtml >}}
$$

\nabla J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{\tau_{:t} \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right]

$$
{{< /rawhtml >}}

### Genralized Advantage Estimation

As we know, {{< rawhtml >}}$Q^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}} represents the expected sum of rewards after taking action {{< rawhtml >}}$a_t${{< /rawhtml >}} in state {{< rawhtml >}}$s_t${{< /rawhtml >}}. We can express it in terms of the immediate reward {{< rawhtml >}}$R(s_t, a_t)${{< /rawhtml >}} and the expected future rewards. We rewrite {{< rawhtml >}}$Q^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}} as follows:

{{< rawhtml >}}
$$

\begin{align*}
Q^{\pi_\theta}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[\sum_{j \geq t}^{T-1} \gamma^{j-t} R(s_j, a_j) \mid s_t, a_t \right] \\
&= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[R(s_t, a_t) + \sum_{j \geq t+1}^{T-1} \gamma^{j-t} R(s_j, a_j) \mid s_t, a_t \right] \\
&= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[R(s_t, a_t) \right] + \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[\sum_{j \geq t+1}^{T-1} \gamma^{j-(t+1)} \gamma R(s_j, a_j) \mid s_t, a_t \right] \\
&= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[R(s_t, a_t) \right] + \gamma \mathbb{E}_{\tau_{t+1:} \sim \pi_{\theta}} \left[\sum_{j \geq t+1}^{T-1} \gamma^{j-(t+1)} R(s_j, a_j) \mid s_{t+1} \right] \\
&= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[R(s_t, a_t) \right] + \gamma V^{\pi_\theta}(s_{t+1}) \\
&= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[R(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) \right] \qquad &&\text{\scriptsize(by Adam's Law)}
\end{align*}

$$
{{< /rawhtml >}}

Thus, we can rewrite the advantage function {{< rawhtml >}}$A^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}} as:

{{< rawhtml >}}
$$

A^{\pi_\theta}(s_t, a_t) = \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[R(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t) \right]

$$
{{< /rawhtml >}}

We can continue recursively and derive more general forms:

{{< rawhtml >}}
$$

\begin{align*}
    A^{\pi_\theta(1)}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[-V^{\pi_\theta}(s_t) + R(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) \right] \\
    A^{\pi_\theta(2)}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[-V^{\pi_\theta}(s_t) + R(s_t, a_t) + \gamma R(s_{t+1}, a_{t+1}) + \gamma^2 V^{\pi_\theta}(s_{t+2}) \right] \\
    A^{\pi_\theta(3)}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[-V^{\pi_\theta}(s_t) + R(s_t, a_t) + \gamma R(s_{t+1}, a_{t+1}) + \gamma^2 R(s_{t+2}, a_{t+2}) + \gamma^3 V^{\pi_\theta}(s_{t+3}) \right] \\
    A^{\pi_\theta(k)}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[-V^{\pi_\theta}(s_t) + R(s_t, a_t) + \gamma R(s_{t+1}, a_{t+1}) + \cdots + \gamma^{k-1} R(s_{t+k-1}, a_{t+k-1}) + \gamma^k V^{\pi_\theta}(s_{t+k}) \right]
\end{align*}

$$
{{< /rawhtml >}}

In these equations:

- {{< rawhtml >}}$A^{\pi_\theta(k)}(s_t, a_t)${{< /rawhtml >}} represents the advantage function considering rewards up to {{< rawhtml >}}$k${{< /rawhtml >}} time steps and the value function of the state at time {{< rawhtml >}}$t+k${{< /rawhtml >}}.
- As {{< rawhtml >}}$k${{< /rawhtml >}} increases, {{< rawhtml >}}$A^{\pi_\theta(k)}(s_t, a_t)${{< /rawhtml >}} incorporates more future rewards and the corresponding value functions.


In practice, we do not have access to the true advantage function {{< rawhtml >}}$A^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}} because it requires knowing {{< rawhtml >}}$Q^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}} and {{< rawhtml >}}$V^{\pi_\theta}(s_t)${{< /rawhtml >}}, neither of which is directly available. This is because we lack complete knowledge of the environment and can only collect a finite number of samples. Thus, we need to estimate {{< rawhtml >}}$A^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}}.

To estimate {{< rawhtml >}}$A^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}}, we use the following approach:

1. **Estimate {{< rawhtml >}}$V^{\pi_\theta}(s)${{< /rawhtml >}}**: Use a value function approximation {{< rawhtml >}}$\hat{V}(s)${{< /rawhtml >}}, which is typically learned through a separate model or neural network.

2. **Estimate {{< rawhtml >}}$A^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}}**: We can estimate {{< rawhtml >}}$A^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}} using the following equations derived from the advantage function formulations:

{{< rawhtml >}}
$$

\begin{align}
    \hat{A}^{(1)}_t & := \delta_t^V = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t) \\
    \hat{A}^{(2)}_t & := \delta_t^V + \gamma \delta_{t+1}^V = r_t + \gamma r_{t+1} + \gamma^2 \hat{V}(s_{t+2}) - \hat{V}(s_t) \\
    \hat{A}^{(3)}_t & := \delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 \hat{V}(s_{t+3}) - \hat{V}(s_t) \\
    \hat{A}^{(k)}_t & := \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} + \gamma^k \hat{V}(s_{t+k}) - \hat{V}(s_t)
\end{align}

$$
{{< /rawhtml >}}

Here:

- {{< rawhtml >}}$\delta_t^V = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)${{< /rawhtml >}} is the temporal difference error at time {{< rawhtml >}}$t${{< /rawhtml >}} and serves as an estimate of the advantage function {{< rawhtml >}}$A^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}} for {{< rawhtml >}}$k=1${{< /rawhtml >}}.
- For higher values of {{< rawhtml >}}$k${{< /rawhtml >}}, the estimates {{< rawhtml >}}$\hat{A}^{(k)}_t${{< /rawhtml >}} incorporate rewards over more future time steps and adjust for the value function at later states.



But which of the above equations (1) through (4) should we use? The key issue here is the bias-variance trade-off. Simpler estimates like {{< rawhtml >}}$\hat{A}^{(1)}_t${{< /rawhtml >}} (Equation (1)) generally have lower variance but higher bias, while more complex estimates like {{< rawhtml >}}$\hat{A}^{(k)}_t${{< /rawhtml >}} (Equation (4)) have lower bias but higher variance. 

To balance this trade-off, we can use a weighted average of these estimates. This approach is known as Generalized Advantage Estimation (GAE), which uses a parameter {{< rawhtml >}}$\lambda${{< /rawhtml >}} to control the trade-off between bias and variance. The GAE formulation combines different estimates of the advantage function into a single, more stable estimate.

The GAE estimate for the advantage function {{< rawhtml >}}$\hat{A}^{\text{GAE}(\gamma, \lambda)}_t${{< /rawhtml >}} is given by:

{{< rawhtml >}}
$$

\begin{align*}
\hat{A}^{\text{GAE}(\gamma, \lambda)}_t &:= (1-\lambda) \left(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots \right) \\
&= (1-\lambda) \left(\delta_t^V + \lambda(\delta_t^V + \gamma \delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V) + \cdots \right) \\
&= (1-\lambda) \left(\delta_t^V \left(1 + \lambda + \lambda^2 + \cdots \right) + \gamma \delta_{t+1}^V \left(\lambda + \lambda^2 + \lambda^3 + \cdots \right) + \gamma^2 \delta_{t+2}^V \left(\lambda^2 + \lambda^3 + \lambda^4 + \cdots \right) + \cdots \right) \\
&= (1-\lambda) \left(\delta_t^V \left(\frac{1}{1-\lambda}\right) + \gamma \delta_{t+1}^V \left(\frac{\lambda}{1-\lambda}\right) + \gamma^2 \delta_{t+2}^V \left(\frac{\lambda^2}{1-\lambda}\right) + \cdots \right) \\
&= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
\end{align*}

$$
{{< /rawhtml >}}

Here, {{< rawhtml >}}$\delta_{t+l}^V${{< /rawhtml >}} represents the temporal difference error at time {{< rawhtml >}}$t + l${{< /rawhtml >}}, and the series {{< rawhtml >}}$\sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V${{< /rawhtml >}} effectively blends different estimates of the advantage function by weighting them according to the parameter {{< rawhtml >}}$\lambda${{< /rawhtml >}}. 

This weighted combination allows for a balance between the low variance of simpler estimates and the low bias of more complex estimates, making the advantage function estimation more robust and effective in practice.



One important consideration is that when the current state is the terminal state, meaning the episode ends after this step, the value of the next state is zero. This results in the advantage estimate for the terminal step being simply:

{{< rawhtml >}}
$$

\hat{A}^{\text{GAE}(\gamma, \lambda)}_t = r_t - \hat{V}(s_t)

$$
{{< /rawhtml >}}

For non-terminal steps, the advantage is calculated using the GAE formula, which incorporates the temporal difference error {{< rawhtml >}}$\delta_t^V${{< /rawhtml >}} and a weighted sum of future {{< rawhtml >}}$\delta${{< /rawhtml >}} values. If the current time step is the last recorded one, the value for the next step must be estimated through bootstrapping using the critic neural network.

The function `calc_gae` is designed to return advantage estimates for each time step in a tensor. It operates iteratively from the last time step to the first, calculating the current {{< rawhtml >}}$\delta_t^V${{< /rawhtml >}} and adding it to {{< rawhtml >}}$\gamma \lambda \delta_{t+l}^V${{< /rawhtml >}}, where {{< rawhtml >}}$\delta_{t+l}^V${{< /rawhtml >}} is the previously computed advantage stored in the `advantages` tensor array.

Additionally, `calc_gae` returns the `returns` values, which are given by:

{{< rawhtml >}}
$$

\text{returns}_t = A(s_t, a_t) + V(s_t)

$$
{{< /rawhtml >}}

These `returns` values are crucial for calculating the loss function of the value network (critic) in Proximal Policy Optimization (PPO). This loss function optimizes the value function by minimizing the difference between the predicted value and the actual returns, ensuring that the value network provides accurate estimates for future rewards.





```python
@tf.function
def calc_gae (rewards, values, dones, gamma, gae_lambda, last_obs, last_done):
    """
    Calculates the Generalized Advantage Estimation (GAE) given data from the replay buffer.

    Args:
        rewards (float32): The rewards received at each timestep, shape: (num_steps, num_envs)
        values (float32): The value function estimates at each timestep, shape: (num_steps, num_envs)
        dones (int32): The done flags indicating episode termination, shape: (num_steps, num_envs)
        gamma (float32): The discount factor for future rewards
        gae_lambda (float32): The discount factor for the weighted average in the advantage calculation
        last_obs (float32): The state after the last action, shape: (num_envs, 8)
        last_done (int32): The done flags after the last action, shape: (num_envs,)

    Returns:
        returns (float32): The target values for the value function, shape: (num_envs, num_steps).
        advantages (float32): The Generalized Advantage Estimation for each timestep, shape: (num_envs, num_steps).
    """

    _, next_value = model(last_obs)
  
    advantages = tf.TensorArray(dtype=tf.float32, size=num_steps)
    lastgaelam = tf.zeros(last_done.shape)
    for t in tf.reverse(tf.range(num_steps), [0]):
        if t == num_steps - 1:
            nextnonterminal = tf.cast(1 - last_done, tf.float32)
            nextvalues = tf.squeeze(next_value)
        else:
            nextnonterminal = tf.cast(1 - dones[t + 1], tf.float32)
            nextvalues = values[t + 1]
        delta = (
            rewards[t]
            + gamma * nextvalues * nextnonterminal
            - values[t]
        )
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        lastgaelam.set_shape(last_done.shape)
        advantages = advantages.write(t, lastgaelam) 
    advantages = tf.squeeze(advantages.stack())
    returns = advantages + values
    return returns, advantages
```

## Computing the Loss Function

The total loss is computed as `pg_loss + vf_loss * vf_coef - ent_coef * entropy_loss`. However, to align with the paper's notation, the estimated objective function to be maximized (not minimized) is expressed as:

{{< rawhtml >}}
$$

\begin{align*}
L_t^{\text{CLIP+VF+S}(\theta)} &= \mathbb{E}_t\left[L_t^{\text{CLIP}}(\theta) - c_1L_t^{\text{VF}}(\theta) + c_2S\pi(\theta)\right] \\
&\approx \frac{1}{N}\sum_{n=1}^{N}\sum_{t=0}^{\infty}L_t^{\text{CLIP}}(\theta) - c_1L_t^{\text{VF}}(\theta) + c_2S\pi(\theta)
\end{align*}

$$
{{< /rawhtml >}}

where {{< rawhtml >}}$S${{< /rawhtml >}} denotes an entropy bonus, and:

{{< rawhtml >}}
$$

\begin{align*}
L^{\text{CLIP}}(\theta) &= \min\left[r_t(\theta)A_t, \text{clip}\left(r_t(\theta), 1 - \epsilon, 1 + \epsilon\right)A_t\right] \quad \text{with } r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_\text{old}(a_t|s_t)} \\
L_t^{\text{VF}}(\theta) &= \max\left[\left(V_t - V_{\text{targ}}\right)^2, \left(\text{clip}\left(V_t, V_{t-1} - \epsilon, V_{t-1} + \epsilon\right) - V_{\text{targ}}\right)^2\right] \quad \text{with } V_{\text{targ}} = A_t + V_t
\end{align*} 

$$
{{< /rawhtml >}}

Next, I will explain each component of this loss function in detail.






### Clipped Policy Gradient Loss, $L_t^{\text{CLIP}}(\theta)$ (`pg_loss`)
The last equation we reached in the previous section about the policy gradient was:

{{< rawhtml >}}
$$

\nabla J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta} (a_{t} \mid s_{t}) \cdot A^{\pi_\theta}(s_t, a_t) \right]

$$
{{< /rawhtml >}}

As previously mentioned, large updates can have detrimental effects on policy gradient algorithms. This happens because a bad update leads to a bad policy, which results in poor performance and, consequently, the collection of suboptimal samples. Poor samples, in turn, push the policy further away from the optimal state. This issue also leads to sample inefficiency, as the policy is constantly changing, and the expectation under the old policy {{< rawhtml >}}$\sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}} [\dots]${{< /rawhtml >}} differs from that under the new policy {{< rawhtml >}}$\sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \pi_{\theta_\text{new}}} [\dots]${{< /rawhtml >}}. 

To address this inefficiency, we can use **importance sampling**, a technique that allows us to convert the estimated expectation over one policy into an estimate over another. This is based on the following property:

{{< rawhtml >}}
$$

\mathbb{E}_{x\sim p}[f(x)] = \int f(x)p(x)dx = \int f(x)\frac{p(x)}{q(x)}q(x)dx = \mathbb{E}_{x\sim q}\left[f(x)\frac{p(x)}{q(x)}\right]

$$
{{< /rawhtml >}}

Applying this to our policy gradient:

{{< rawhtml >}}
$$

\begin{align*}
\nabla J(\theta) &= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta} (a_{t} \mid s_{t}) \cdot A^{\pi_\theta}(s_t, a_t) \right] \\
&=\sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}} \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} \nabla_{\theta}\log\pi_\theta(a_t \mid s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right] \\
&= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}} \left[ \frac{\nabla_{\theta}\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} \cdot A^{\pi_\theta}(s_t, a_t) \right] \qquad &&\text{\scriptsize(because $\nabla \log(x) = \frac{\nabla x}{x}$)}
\end{align*}

$$
{{< /rawhtml >}}

Therefore, after updating the parameters with one mini-batch, we can still use the remaining mini-batches (sampled with the previous parameters) to update the new parameters. This approach is known as the **surrogate loss function**.

There is a trade-off here. Although the expectations of the two forms are equal, their variances differ. The surrogate loss function could have a greater variance:

{{< rawhtml >}}
$$

\begin{align*}
\text{Var}_{x\sim p}[f(x)] &= \mathbb{E}_{x\sim p}[f(x)^2] - (\mathbb{E}_{x\sim p}[f(x)])^2, \\
\text{Var}_{x\sim q}[f(x)] &= \mathbb{E}_{x\sim q}\left[\left(f(x)\frac{p(x)}{q(x)}\right)^2\right] - \left(\mathbb{E}_{x\sim q}\left[f(x)\frac{p(x)}{q(x)}\right]\right)^2, \\
&= \mathbb{E}_{x\sim p}\left[f(x)^2\frac{p(x)}{q(x)}\right] - \left(\mathbb{E}_{x\sim p}\left[f(x)\right]\right)^2.
\end{align*}

$$
{{< /rawhtml >}}

Here, the surrogate function has an extra term, {{< rawhtml >}}$\frac{p(x)}{q(x)}${{< /rawhtml >}}. If {{< rawhtml >}}$p(x)${{< /rawhtml >}} is very different from {{< rawhtml >}}$q(x)${{< /rawhtml >}}, this can lead to high variance and, in fact, unstable updates. To alleviate this instability, we can limit {{< rawhtml >}}$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_\text{old}(a_t|s_t)}${{< /rawhtml >}}—the exact expression responsible for the extra variance—by {{< rawhtml >}}$1-\epsilon${{< /rawhtml >}} and {{< rawhtml >}}$1+\epsilon${{< /rawhtml >}}. This creates the final form of the clipped policy gradient loss:

{{< rawhtml >}}
$$

L^{\text{CLIP}}(\theta) = \min\left[r_t(\theta)A_t, \text{clip}\left(r_t(\theta), 1 - \epsilon, 1 + \epsilon\right)A_t\right], \text{ where } r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_\text{old}(a_t|s_t)}.

$$
{{< /rawhtml >}}

Since we are trying to minimize the loss function, we need to compute {{< rawhtml >}}$-L^{\text{CLIP}}(\theta)${{< /rawhtml >}} to be consistent with TensorFlow's automatic differentiation.

In this formulation, we don't explicitly see {{< rawhtml >}}$\nabla${{< /rawhtml >}} behind the {{< rawhtml >}}$\log${{< /rawhtml >}} because the entire {{< rawhtml >}}$L^{\text{CLIP}}(\theta)${{< /rawhtml >}} will be differentiated automatically. This approach aligns with the gradient equations discussed earlier, as it effectively sets {{< rawhtml >}}$\nabla_\theta \hat{A}(s_t, a_t) = 0${{< /rawhtml >}}. However, in practice, the actor and critic often share several layers, meaning the estimated advantage function {{< rawhtml >}}$\hat{A}^{\pi_\theta}(s_t, a_t)${{< /rawhtml >}}—constructed using values estimated by the critic network—depends on {{< rawhtml >}}$\theta${{< /rawhtml >}}. This introduces some bias but also accelerates training. Depending on the problem being solved, different network designs may be more appropriate.


### Value Function Loss, $c_1L_t^{\text{VF}}(\theta)$ (`vf_loss * vf_coef`)

Since the actor and critic networks share parameters, incorporating the value function loss into the overall loss function is crucial. PPO aims to bring the value function estimates closer to the target value function {{< rawhtml >}}$V_{targ} = A_t^{GAE} + V_t${{< /rawhtml >}}, using Temporal Difference (TD) learning with {{< rawhtml >}}$\lambda${{< /rawhtml >}}. 

Recall the Generalized Advantage Estimate (GAE) for a given time step {{< rawhtml >}}$t${{< /rawhtml >}}:

{{< rawhtml >}}
$$

\begin{align*}
    \hat{A}^{\text{GAE}(\gamma, \lambda)}_t &:= (1-\lambda) \left(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots \right) \\
    A^{\pi_\theta(1)}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[-V^{\pi_\theta}(s_t) + R(s_t, a_t) + \gamma V(s_{t+1}) \right] \\
    A^{\pi_\theta(2)}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[-V^{\pi_\theta}(s_t) + R(s_t, a_t) + \gamma R(s_{t+1}, a_{t+1}) + \gamma^2 V(s_{t+2}) \right]\\
    A^{\pi_\theta(3)}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[-V^{\pi_\theta}(s_t) + R(s_t, a_t) + \gamma R(s_{t+1}, a_{t+1}) + \gamma^2 R_{t+2} + \gamma^3 V(s_{t+3}) \right] \\
    A^{\pi_\theta(k)}(s_t, a_t) &= \mathbb{E}_{\tau_{t:} \sim \pi_{\theta}} \left[-V^{\pi_\theta}(s_t) + R(s_t, a_t) + \gamma R(s_{t+1}, a_{t+1}) + \cdots + \gamma^{k-1} R(s_{t+k-1}, a_{t+k-1}) + \gamma^k V(s_{t+k}) \right]
\end{align*}

$$
{{< /rawhtml >}}

The advantage function {{< rawhtml >}}$A_t^{GAE}${{< /rawhtml >}} represents a weighted average of the difference between the value function and more confident estimates of the value function, considering various degrees of confidence. The target value function {{< rawhtml >}}$V_{targ}${{< /rawhtml >}} is given by:

{{< rawhtml >}}
$$

V_{targ} = A_t^{GAE} + V_t

$$
{{< /rawhtml >}}

To stabilize updates and avoid instability, we clip the value function estimates. The loss function for the value function is thus defined as:

{{< rawhtml >}}
$$

L_t^{\text{VF}}(\theta) = \max\left[\left(V_t - V_{targ}\right)^2, \left(\text{clip}\left(V_t, V_{t-1} - \epsilon, V_{t-1} + \epsilon\right) - V_{targ}\right)^2\right] \text{ with } V_{targ} = A_t^{GAE} + V_t

$$
{{< /rawhtml >}}

In the original paper, this value is subtracted because the objective is to maximize the total loss. However, for consistency with automatic differentiation in frameworks like TensorFlow, where we typically minimize the loss, this value is added to the overall loss function.

### Entropy Loss, $c_2S\pi(\theta)$ (`ent_coef * entropy_loss`)

The entropy loss term addresses the exploration-exploitation dilemma in reinforcement learning. It encourages exploration by penalizing deviations of action probabilities from a uniform distribution, which helps prevent getting stuck in local minima. By promoting exploration, the algorithm can more effectively search the action space for better policies.

Entropy is a measure of randomness or uncertainty in a probability distribution. For a discrete probability distribution {{< rawhtml >}}$p(x_i)${{< /rawhtml >}} over {{< rawhtml >}}$n${{< /rawhtml >}} possible outcomes, entropy is defined as:

{{< rawhtml >}}
$$

H(X) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)

$$
{{< /rawhtml >}}

The entropy is maximized for a discrete uniform distribution. For a uniform distribution {{< rawhtml >}}$U = \{u_1, u_2, \dots, u_n\}${{< /rawhtml >}}, the entropy is:

{{< rawhtml >}}
$$

H(U) = \sum_{i=1}^{n} \frac{1}{n} \log\left(\frac{1}{n}\right) = \log(n)

$$
{{< /rawhtml >}}

Now, consider a random variable with probabilities given by {{< rawhtml >}}$X = \{\frac{1}{p_1}, \frac{1}{p_2}, \dots, \frac{1}{p_n}\}${{< /rawhtml >}}. For this random variable, the entropy is:

{{< rawhtml >}}
$$

H(X) = \mathbb{E}[\log(X)] \stackrel{\text{Jensen's inequality}}{\leq} \log(\mathbb{E}[X]) = \log(n) = H(U)

$$
{{< /rawhtml >}}

This shows that the entropy for any other distribution is less than or equal to that of the uniform distribution. 

To encourage exploration, we aim to increase the entropy, which is achieved by subtracting the entropy loss in the overall loss function. This approach helps to ensure that the policy does not become overly deterministic and remains exploratory.

Finally, we perform all calculations in logarithmic form where possible to enhance numerical stability. We also use the `@tf.function` decorator to create TensorFlow static graphs, which improves computation speed. However, in reinforcement learning, certain processes—such as interacting with the environment and filling the replay buffer—cannot be easily parallelized. For instance, you must wait for the completion of one step before proceeding to the next. Consequently, while GPUs can accelerate computations, their impact on speed is less dramatic in reinforcement learning compared to other domains, such as CNNs for image processing.




```python

@tf.function
def compute_loss(mb_obses, mb_actions, mb_log_probs, mb_advantages, mb_values, mb_returns):
    """
    Computes the overall loss function to be minimized given the replay buffer and the computed GAE for mini-batches.

    Args:
        mb_obses (float32): Mini-batch states, shape: (mbatch_size, 8)
        mb_actions (int32): Mini-batch actions that were taken, shape: (mbatch_size,)
        mb_log_probs (float32): Mini-batch log probabilities of the actions that were taken, shape: (mbatch_size,)
        mb_advantages (float32): Mini-batch calculated GAE, shape: (mbatch_size,)
        mb_values (float32): Mini-batch current value estimates, shape: (mbatch_size,)
        mb_returns (float32): Mini-batch targets for the value function, shape: (mbatch_size,)

    Returns:
        overall_loss (float32): Overall loss for the mini-batch, shape: ()
    """


    logits, curr_values = model(mb_obses)
    curr_log_probs  = tf.nn.log_softmax(logits)
    curr_log_aprobs = tf.reduce_sum(curr_log_probs*tf.one_hot(mb_actions, envs.single_action_space.n),1)
    ratio = tf.exp(curr_log_aprobs - mb_log_probs) #r_t(theta)
    mb_advantages = (mb_advantages - tf.reduce_mean(mb_advantages)) / (tf.math.reduce_std(mb_advantages) + 1e-8) #scaling the advantages
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * tf.clip_by_value(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

    curr_values = tf.squeeze(curr_values)
    vpredclipped = mb_values + tf.clip_by_value(curr_values - mb_values, - clip_coef, clip_coef)
    vf_losses1 = tf.square(curr_values - mb_returns)
    vf_losses2 = tf.square(vpredclipped - mb_returns)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))


    entropy = tf.reduce_sum(tf.exp(tf.math.log(-curr_log_probs + 1e-8) + curr_log_probs), axis=1)
    entropy_loss = tf.reduce_mean(entropy)

    return pg_loss + vf_loss * vf_coef - ent_coef * entropy_loss 


```

## Training the Network with Mini-Batches

This function trains the network using data from the replay buffer. The batch size of 512 is divided into 4 non-consecutive mini-batches, with each batch being used for training 4 times.


```python
@tf.function
def train_batch(obses, actions, log_probs, dones, values, returns, advantages):
    """
    Trains the network using data from the replay buffer.

    Args:
        obses (float32): The batch of states, shape: (num_steps, num_envs, 8)
        actions (int32): The actions taken in the replay buffer, shape: (num_steps, num_envs)
        log_probs (float32): The log probabilities of the actions taken in the replay buffer, shape: (num_steps, num_envs)
        dones (float32): The done flags for each timestep in the batch, shape: (num_steps, num_envs)
        values (float32): The estimated values for each state in the batch, shape: (num_steps, num_envs)
        returns (float32): The target values of the value function in the batch, shape: (num_steps, num_envs)
        advantages (float32): The calculated Generalized Advantage Estimates (GAE) for the batch, shape: (num_steps, num_envs)

    Returns:
        None
    """

    #flattening (num_envs, num_steps) to (batch_size):
    obses = tf.reshape(obses, (-1,) + envs.single_observation_space.shape)
    actions = tf.reshape(actions, (-1,))
    log_probs = tf.reshape(log_probs, (-1,))
    dones = tf.reshape(dones, (-1,))
    values = tf.reshape(values, (-1,))
    returns = tf.reshape(returns, (-1,))
    advantages = tf.reshape(advantages, (-1,))
    
    for update in tf.range(update_epochs):
        batch_idx = tf.random.shuffle(tf.range(batch_size))

        for mb in tf.range(0, batch_size, mbatch_size):
            idx = batch_idx[mb:mb+mbatch_size]
            mb_obses = tf.gather(obses, idx)
            mb_actions = tf.gather(actions, idx)
            mb_log_probs = tf.gather(log_probs, idx)
            mb_dones = tf.gather(dones, idx)
            mb_values = tf.gather(values, idx)
            mb_returns = tf.gather(returns, idx)
            mb_advantages = tf.gather(advantages, idx)

            with tf.GradientTape() as tape:
                loss = compute_loss(mb_obses, mb_actions, mb_log_probs, mb_advantages, mb_values, mb_returns)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

## Play the Game and Train Network

This function plays the game for a predefined number of steps and passes the replay buffer to the `train_batch` function for training the network.


```python
@tf.function
def run_batch(initial_obs):
    """Fills the replay buffer starting from the initial state for each environment.

        Args:
            initial_obs (float32): The initial state for each environment, shape: (num_envs, 8)

        Returns:
            ep_rs (float32): The episodic return for the terminated episodes, shape: (batch_size,)
            ep_ls (float32): The episode length for the terminated episodes, shape: (batch_size,)
    """

    obses = tf.TensorArray(tf.float32, size=num_steps)
    log_probs = tf.TensorArray(tf.float32, size=num_steps)
    rewards = tf.TensorArray(tf.float32, size=num_steps)
    dones = tf.TensorArray(tf.int32, size=num_steps)
    values = tf.TensorArray(tf.float32, size=num_steps)
    actions = tf.TensorArray(tf.int32, size=num_steps)
    ep_rs = tf.TensorArray(tf.float32, size=num_steps)
    ep_ls = tf.TensorArray(tf.int32, size=num_steps)
    obs = initial_obs
    done = tf.zeros((num_envs,), dtype=tf.int32)
    for t in tf.range(num_steps):
        obses = obses.write(t, obs)
        dones = dones.write(t, done)
        logit, value = model(obs)
        action = tf.squeeze(tf.random.categorical(logit, 1, dtype=tf.int32))
        log_prob = tf.reduce_sum(tf.nn.log_softmax(logit) * tf.one_hot(action, logit.shape[-1]), axis=1)
        obs, reward, done, ep_r, ep_l = tf_step(action)
        ep_rs = ep_rs.write(t, ep_r)
        ep_ls = ep_ls.write(t, ep_l)
        obs.set_shape(initial_obs.shape)
        done.set_shape((num_envs,))
        log_probs = log_probs.write(t, log_prob)
        rewards = rewards.write(t, reward)
        actions = actions.write(t, action)
        values = values.write(t, value)

    obses = obses.stack()
    log_probs = log_probs.stack()
    rewards = rewards.stack()
    dones = dones.stack()
    values = tf.squeeze(values.stack())
    actions = actions.stack()
    ep_rs = ep_rs.stack()
    ep_ls = ep_ls.stack()
    returns, advantages = calc_gae(rewards, values, dones, gamma, gae_lambda, obs, done)
    train_batch(obses, actions, log_probs, dones, values, returns, advantages)
    return ep_rs, ep_ls

```

## Training Loop with Learning Rate Annealing

This loop handles the training process and incorporates learning rate annealing. For each epoch, it outputs several metrics, including mean rewards per episode, mean episode length, steps per second, and the global step count. The model weights are saved at the end.


```python
optimizer = keras.optimizers.Adam(global_clipnorm=max_grad_norm, epsilon=eps)
initial_obs = tf.constant(envs.reset(seed = seed)[0], dtype=tf.float32)
global_steps = 0
start = time.time()
writer = tf.summary.create_file_writer(get_run_logdir("logs"))
for epoch in range(num_epochs):
    frac = 1.0 - epoch / num_epochs
    lrnow = frac * lr_rate
    optimizer.learning_rate = lrnow
    ep_rs, ep_ls = run_batch(initial_obs)
    rewards = tf.reduce_mean(ep_rs[ep_rs != 0])
    length = tf.reduce_mean(ep_ls[ep_ls != 0])
    global_steps += batch_size
    if not tf.math.is_nan(rewards):
        with writer.as_default():
                tf.summary.scalar("rewards", rewards, step=global_steps)
                tf.summary.scalar("length", length, step=global_steps)
        print(f"rewards:{int(rewards)}, length:{length}, SPS:{int(global_steps/(time.time() - start))}, step:{global_steps}")
model.save_weights('./m.weights.h5')
```

Output:
```
rewards:-201, length:95, SPS:109, step:512
rewards:-202, length:88, SPS:209, step:1024
rewards:-262, length:121, SPS:299, step:1536
rewards:-233, length:105, SPS:383, step:2048
rewards:-206, length:91, SPS:459, step:2560
rewards:-167, length:82, SPS:531, step:3072
rewards:-278, length:97, SPS:596, step:3584
....
rewards:274, length:186, SPS:1576, step:1959936
rewards:294, length:200, SPS:1576, step:1960448
rewards:266, length:205, SPS:1576, step:1960960
rewards:203, length:152, SPS:1576, step:1961472
rewards:208, length:162, SPS:1577, step:1961984
rewards:284, length:175, SPS:1577, step:1962496
rewards:276, length:173, SPS:1577, step:1963008
```
Summary:

!['rewards and lenght summary'](./images/run.jpg)

## Rendering an Episode and Saving as a MP4

This function renders an episode from a given environment using a provided model while using `gym.wrappers.RecordVideo` to save the episode in `.mp4` format. The environment is stepped through until the episode ends or the maximum number of steps is reached. The model weights are being loaded at the start, and the environment interacts with the model's actions until the episode concludes.


```python


model.load_weights('model2M.weights.h5')
render_env = gym.wrappers.RecordVideo(gym.make("LunarLander-v2", render_mode='rgb_array'), "videos", name_prefix="lunar_lander", disable_logger=True)
state, info = render_env.reset()
state = tf.constant(state, dtype=tf.float32)

for i in range(1, 1001):
  state = tf.expand_dims(state, 0)
  action_probs, _ = model(state)
  action = np.argmax(np.squeeze(action_probs))

  state, reward, done, truncated, info = render_env.step(action)
  
  if done:
    break
```

<video src="./videos/lunar_lander-episode-0.mp4" width="500" controls autoplay loop></video>

## References

- Richard S. Sutton and Andrew G. Barto, "Reinforcement Learning: An Introduction". 2nd edition. Bradford Books. MIT Press. (2018)

- OpenAI docs. "Extra Material, Proof for Using Q-Function in Policy Gradient Formula." OpenAI Spinning Up. https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof2.html

- Huang, et al., "The 37 Implementation Details of Proximal Policy Optimization", ICLR Blog Track, 2022.

- Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).

- Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

- Ruifan Yu. "CS885 Lecture 15b: Proximal Policy Optimization." Pascal Poupart's YouTube channel. https://youtu.be/wM-Sh-0GbR4?si=_1cX52IfNyx14Iuu

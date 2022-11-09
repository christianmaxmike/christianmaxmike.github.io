---
title: "Reinforcement Learning - Fundamentals"
topic: rl
collection: rl
permalink: /mindnotes/ml-fundamentals
---


<img src="logo_cmmf.png"
     alt="Markdown Monster icon"
     style="float: right" />
# MindNote - Machine Learning - Reinforcement Learning

**Author: Christian M.M. Frey**  
**E-Mail: <christianmaxmike@gmail.com>**

---

## Reinforcement Learning - Fundamentals
---


**Level**: beginner<br>
**pre-knowledge**: none<br>

---

### What it is about?

In this series, i'd like to recap some basif definition in the scope of Reinforcement Learning.

### Terminologies

---
What is a **Markov Decision Process (MDP)**?

An MDP is a base model used all over the world of Reinforcement Learning. Basically, it defines the envrironemnt in terms of transitioning from one state to another by taking specific actions and their rewards by taking these actions. 
Therefore, we can define a tuple $(S,A,T,R)$, where $S$ denotes the environment's states, $A$ is the action space an agent can take, $T$ denotes the probabilitiy of transitioning from one state to another by taking a specific action, and $R$ denotes a reward function define for all transitions. 

As MDP form one of the crucial definitions in RL, I'll give here a formal definition. We define a MDP as follows:

>  Definition **Markov Decision Process**:<br>
A Markov Decision Process is a tuple $(S,A,T,R)$, in which $S$ is a finite set of states, $A$ a finite set of actions, $T$ a transition function  defined as $T \colon S \times A \times S \rightarrow [0, 1]$ and $R$ a reward function defined as $R \colon S \times A \times S \rightarrow \mathbb{R}$.


---
What is a **policy**?

Policy describes a mapping of an action to every possible state in the environment. In RL, we are interested in finding an optimal policy, i.e., the policy which maximizes the long term reward. This implies, that for each sufficient large environment we can have multiple policies, i.e., a set of actions at each state. However, there is probably only one policy which is optimal w.r.t an optimal criterion. 

---
What are **optimality criteria**?

The goal of learning in an MDP is to optimize the acquisition of rewards. As an agent behaves sequentially in an environment, we are interested in including also possible future rewards. Now, different optimality criteria can be defined depending on how to incroporate and interpreting future steps. Let's have a look at three different criteria:

- _Finite horizon_: <br> 
$\mathbb{E}\left[\sum_{i=0}^{h} r_t \right]$ <br>
In this model, the agent optimizes the reward over a horizon of length $h$

- _Infitine horizon / discounted horizon_ <br>
$\mathbb{E}\left[\sum_{i=0}^{\infty} \gamma^t r_t \right]$ <br> 
As the name implies, we take the full horizon of an agent into account. However, future rewards are discounted according to how far away in time they will be received. A discount factor $\gamma$ with $0 \leq \gamma \lt 1$ regulates the influence of future rewards. 

- _Average reward_ <br>
$\lim_{h \rightarrow \infty} \mathbb{E}\left[\frac{1}{h}\sum_{i=0}^{h} r_t \right]$ <br> 
This model maximizes the long-run average rewards. The connection to the infinitie horizon model is trivial if $h$ approaches $\infty$ with a discount factor of $1$. A problem arises that this model cannot distinguish between rewards resulting from short-runs and long-runs 


More optimiality criteria have been defined in the literature 


---
What is a **value function**?

A value function now links the chosen optimal criterion to policies. Generally, a value function is used to estimate how good it is for the agent to be in a specific state, respectively, how good it is to perform a certain action in a specific state. The _goodness_ is expressed w.r.t. the optimality criterion. For example, the value of a state $s$ under policy $\pi$, denoted by $V^\pi(s)$ can be expressed as follows using the discounted horizon optimality criterion: <br>
$V^\pi(s)=\mathbb{E}_\pi \left\{\sum_{k=0}^{\infty} \gamma^k r_{t+k} | s_t = s\right\}$


--- 
What is the **q-value**? 

Similiar to the example given for the value function, we can express the state-action value function $Q:S\times A \rightarrow \mathbb{R}$ for a state $s$ under policy $\pi$, performing action $a$ as follows using the discounted horizon optimality criterion: <br>
$Q^\pi(s,a)=\mathbb{E}_\pi \left\{\sum_{k=0}^{\infty} \gamma^k r_{t+k} | s_t = s, a_t = a\right\}$

---
What is the **Bellman equation**?

The Bellman equation states the recursive property of value functions. For example, let's have a look at the value function w.r.t. the discounted optimality criterion: <br>
$V^\pi(s) = \mathbb{E}_\pi\left\{r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots | s_t = s \right\}$<br>
$= \mathbb{E}_\pi \left\{ r_t + \gamma V^\pi(s_{t+1)}| s_t = s \right\}$<br>
$\sum_{s'}T(s,\pi(s), s')\left(R(s,a,s') + \gamma V^\pi(s') \right)$

In words, it denotes the state's expected value considering the reward when transitioning from state $s$ to $s'$ and the next states values weighted by the transitioning probability denoted by $T(\cdot)$.

# End of this MindNote


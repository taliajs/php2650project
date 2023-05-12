# Applying Bayesian Inference into Neural Networks

Deep learning is a machine learning technique used for learning neural
networks, and for processing data in a more “human” way. Deep Learning
Neural Networks are a powerful tool that can be used to approximate
solutions to many functions, even if they are not closed form. However,
a major problem with neural networks is overfitting, which is when the
learning algorithm does such a good job of tuning the parameters on the
training set that it performs badly on new data \[1\]. There are many
methods that can be used to reduce overfitting (like early stopping or
dropout), but the method we are focusing on is **applying Bayesian
inference into the neural network**.

## Bayesian

The Bayesian paradigm is based on 2 simple ideas: 1) Probability is a
measure of belief in the occurrence of events, rather than the limit in
the frequency of occurrence (when the number of samples goes to
infinity), which is assumed in the frequentist paradigm. 2) Prior
beliefs influence posterior probabilities, also known as Bayes theorem
\[2\].

A saying that is often used to describe Bayesian is “it’s foolish to
believe something with 100% probability”. A well known example of this
is swans. For the longest time, people used to think there were only
white swans (that all swans were white). But, that statement was proven
wrong when it was discovered that black swans exist.

The idea of Bayesian inference is to take into account everything. This
means to have probabilities, no matter how small, on all possibilities
(all possible outcomes), so we can update our prior belief/knowledge
when new information is obtained. In the context of a model, leaving
everything with probability allows the model to take into account future
information and update it, rather than making a new model.

## Bayesian Neural Networks (BNNs)

Bayesian inference allows us to learn a probability distribution over
possible neural networks \[1\], which can not only reduce overfitting,
but inform us on how much uncertainty the model has.

Bayesian neural networks are neural networks that are trained using a
Bayesian approach \[2\], meaning that the neural network has a prior on
its weights \[3\]. BNNs are a promising paradigm that allows the
generalization of applying deep learning in areas where a system is not
allowed to fail \[2\]. These networks are about modeling uncertainty in
parameters \[2\]. By modeling uncertainty, BNNs provides better
prediction accuracy under the same model, and provides better
uncertainty estimation for a predictive distribution \[3\]. BNNs are
useful in active learning (human or algorithm labels new points from an
unlabeled dataset) and online learning (where a model is retrained as
new data becomes available) \[2\].

Normally, a neural networks aims to use the training data \[**INSERT
D_train** as a mathematical formula with subscript\] to update the
weight parameters
<img src = "https://render.githubusercontent.com/render/math?math=w"> so
as to maximize a loss function *L(w)*. A neural network through the
Bayesian approach aims to estimate the posterior distribution (**INSERT
EQUATION HERE**), the distribution of the weight parameters given the
training data the model is fitted with \[1\].

There are several ways to estimate this posterior distribution - using
**exact inference**, approximating the posterior through **sampling
methods/techniques**, or using **variational inference** to approximate
the posterior distribution. Our project will explore these three
methods, and how these methods can be implemented.

## Exact Inference

This method takes a traditional Bayesian approach to estimate the
posterior, where the posterior is computed using Bayes Rule. First, we
need to identify the prior distribution of the weights *p(w)* as well as
the likelihood function for our training data *p(D\|w)*. The prior
represents our current belief on what possible values the weight
parameters could be, while the likelihood function for the training data
is required to estimate the probabilities for each observation of the
training data to occur. Through the use of Bayes Rule, the posterior
distribution can be calculated as follows:

<center>

<img src = "img/bayes-posterior.png" width = "400"/>

</center>

After estimating the posterior, the predictive distribution \[**INSERT
EQUATION HERE**\] can be calculated to estimate the output values given
any new input:

<center>

<img src = "img/predictive-distribution.png" width = "325" />

</center>

This predictive distribution can be interpreted as an infinite ensemble
of weights, where each possible network contributes to the overall
prediction after being weighted by the posterior likelihood \[1\].

Exact inference is very easy to express, but a downside is that it
becomes computationally expensive the more parameters the neural network
has. This is due to integration being required over all of the weight
parameters, which for the most part can only be approximated
numerically. Thus, exact inference is really only feasible for small
neural networks.

## Sampling Methods

A sampling method used to exactly sample the posterior is **Markov Chain
Monte Carlo (MCMC)**. MCMC algorithms are considered the best tool for
sampling from the exact posterior, and tend to be better for high
dimensionality problems \[1\].

The assumption of sample independence is sacrificed by using a Markov
Chain to produce a sequence of dependent samples \[3\], with a proposal
distribution over choices in the next sample in the sequence conditioned
on the previous choice \[**INSERT EQUATION HERE!!**\] \[1\].

The idea behind MCMC is to construct a Markov Chain (a sequence of
random samples Si, which probabilistically depends only on the previous
sample Si-1, such that Si are distributed following a desired
distribution \[2\]. Most MCMC algorithms require an initial burn in time
before the Markov chain converges to the desired distribution \[1\].

The most relevant MCMC method for Bayesian neural networks is the
**Metropolis-Hastings** algorithm. This algorithm uses the proposal
distribution mentioned earlier to generate a set of samples that
asymptotically are distributed according to *p(w\|D)* \[1\]. The Markov
Chain is used to generate candidate samples, and then stochastically
accept these samples with probability a, expressed as the acceptance
rate:

<center>

<img src = "img/acceptance-rate.png" height = "50" />

</center>

This accept rate is equivalent to the ratio of posterior probabilities
under the proposed and most recent samples \[**INSERT EQUATION HERE**\]
\[1\].

## Approximate Inference/Variational Inference

Variational inference is another method for learning the approximation
of a posterior \[2\]. Variational inference generates a proposal
posterior and compares that posterior to the observed or actual
posterior from the data. The idea of variational inference is to set up
a approximate posterior distribution \[**EQUATION**\] to model the
posterior *p(w\|D)*, where where the distribution is parameterized by a
set of parameters \[PHI - INSERT SYMBOL!\] \[1,2\]. Then, a loss
function is used to update the posterior parameters until the loss is
minimized.

A common loss function used is the **Kullback-Liebler divergence**
(KL-divergence), which is a similarity measure for two distributions
(measures the difference between two probability distributions):

<center>

<img src = "img/kl-divergence-formula2.png" width = "400" />

</center>

In the equation above, the log represents the two distributions. The
numerator represents the posterior proposal, and the denominator, which
represents the observed posterior, stays constant. At each epoch, you
fit a new posterior and compare it to the observed posterior; this would
be the loss function. Additionally, at every new epoch, new weights are
sampled, and backpropagation and gradient descent is done on the
parameters for each posterior distribution for each weight and node.

**SOMETHING ABOUT EVIDENCE LOWER BOUND (ELBO??)**

### Bayes by Backpropagation

-   source 4

## Application

We will implement variational inference using Kullback-Liebler/Bayes by
Backprop and MCMC.

First, we will implement a Bayesian neural network using variational
inference for a sine function:

***EXPLANATION!!!!***

<figure>
<img src = "img/sine-plot.png" width = "500" /> alt=“Plot showing…” /\>
<figcaption aria-hidden="true">
<em>Plot showing…</em>
</figcaption>
<figure>

**TEST**

<u>Concrete Data </u>

This dataset was found on Kaggle, an online platform that has publicly
avilable datasets for use. This dataset contains information on 8
different components/features used to make concrete for 1,030 concrete
samples, with the outcome of interest being concrete compressive
strength. All the variables are quantitative.

Description of the Dataset (from Kaggle):

> Concrete is the most used material for construction in the world!
> There are some components that should be combined to make the
> concrete. These components can affect the compressive strength of the
> concrete. To obtain the real compressive strength of concrete (target
> labels in the dataset), an engineer needs to break the cylinder
> samples under the compression-testing machine. The failure load is
> divided by the cylinder’s cross-section to obtain the compressive
> strength. Engineers use different kinds of concretes for different
> building purposes. For example, the strength of concrete used for
> residential buildings should not be lower than 2500 psi (17.2 MPa).
> Concrete is a material with high strength in compression, but low
> strength in tension. That is why engineers use reinforced concrete
> (usually with steel rebars) to build structures.

<table>
<colgroup>
<col style="width: 36%" />
<col style="width: 63%" />
</colgroup>
<thead>
<tr class="header">
<th>
Variable
</th>
<th>
Description
</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>
cement
</td>
<td>
Cement (kg in a m3 mixture)
</td>
</tr>
<tr class="even">
<td>
slag
</td>
<td>
Blast Furnace Stag (kg in a m3 mixture)
</td>
</tr>
<tr class="odd">
<td>
flyash
</td>
<td>
Fly Ash Stag (kg in a m3 mixture)
</td>
</tr>
<tr class="even">
<td>
water
</td>
<td>
Water (kg in a m3 mixture)
</td>
</tr>
<tr class="odd">
<td>
superplasticizer
</td>
<td>
Superplasticizer component (kg in a m3 mixture)
</td>
</tr>
<tr class="even">
<td>
coarseaggregate
</td>
<td>
Coarse Aggregate (kg in a m3 mixture)
</td>
</tr>
<tr class="odd">
<td>
fineaggregate
</td>
<td>
amount of Fine Aggregate (kg in a m3 mixture)
</td>
</tr>
<tr class="even">
<td>
age
</td>
<td>
Age of the concrete sample (in days); ranges from 1-365 days
</td>
</tr>
<tr class="odd">
<td>
csMPa
</td>
<td>
concrete compressive strength in MPa (output variable)
</td>
</tr>
</tbody>
</table>

We will implement variational and sampling methods on this concrete
linear regression dataset.

### Variational Inference

***EXPLANATION***

<figure>
<img src = "img/variational-data.png" width = "500" />
<figure>

### MCMC Sampling

***EXPLANATION***

***MENTION THE ALGORITHM USED***

<figure>
<img src = "img/mcmc-data.png" width = "500" />
<figure>

## References

1.  Yu, et al. Bayesian Neural Networks.
    <https://www.cs.toronto.edu/~duvenaud/distill_bayes_net/public/>

2.  Jospin et al (2022). Hands-on Bayesian Neural Networks – A Tutorial
    for Neural Deep Learning Methods. Machine Learning.IEEE
    Computational Intelligence Magazine 17(2): 29-48

3.  Bayesian Deep Learning \[slides\]:
    <https://alinlab.kaist.ac.kr/resource/Lec8_Bayesian_DL.pdf>

4.  Blundell, et al. Weight Uncertainty in Neural Networks.
    <https://arxiv.org/abs/1505.05424>

(MCMC algorithm that is used in our application) Chandra, et
al. Bayesian Neural Networks via MCMC: a Python-based tutorial.
<https://arxiv.org/pdf/2304.02595v1.pdf>

Concrete Data Reference - I-Cheng Yeh, “Modeling of strength of high
performance concrete using artificial neural networks,” Cement and
Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).

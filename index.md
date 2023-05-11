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

## Bayesian Neural Networks (BNNs)

Bayesian inference allows us to learn a probability distribution over
possible neural networks \[1, which can not only reduce overfitting, but
inform us on how much uncertainty the model has.

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

Normally, a neural networks aims to use the training data D_train to
update the weight parameters
<img scr=https://render.githubusercontent.com/render/math?math=w"> so as
to maximize a loss function L. A neural network through the Bayesian
approach aims to estimate the posterior distribution (**INSERT EQUATION
HERE**), the distribution of the weight parameters given the training
data the model is fitted with \[1\].

There are several ways to estimate this posterior distribution - using
**exact inference**, approximating the posterior through **sampling
methods/techniques**, or using **variational inference** to approximate
the posterior distribution. Our project will explore these three
methods, and how these methods can be implemented.

<img src="https://render.githubusercontent.com/render/math?math={\L = -\sum_{j}[T_{j}ln(O_{j})] + \frac{\lambda W_{ij}^{2}}{2} \rightarrow \text{one-hot} \rightarrow -ln(O_{c}) + \frac{\lambda W_{ij}^{2}}{2}}">

<https://render.githubusercontent.com/render/math?math=%7Bln(O_%7Bc%7D>)}

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

## Exact Inference

## Sampling Methods

## Approximate Inference/Variational Inference

## References

1.  Yu, et al. Bayesian Neural Networks.
    <https://www.cs.toronto.edu/~duvenaud/distill_bayes_net/public/>

2.  Jospin et al (2022). Hands-on Bayesian Neural Networks – A Tutorial
    for Neural Deep Learning Methods. Machine Learning.IEEE
    Computational Intelligence Magazine 17(2): 29-48

3.  Bayesian Deep Learning \[slides\]:
    <https://alinlab.kaist.ac.kr/resource/Lec8_Bayesian_DL.pdf>

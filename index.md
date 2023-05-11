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
beliefs influence posterior probabilities, also known as Bayes theorem.
\[2\].

## Bayesian Neural Networks (BNNs)

This sentence uses delimiters to show math inline: $\sqrt{3x-1}+(1+x)^2$

$$\left( \sum\_{k=1}^n a_k b_k \right)^2 \leq \left( \sum\_{k=1}^n a_k^2 \right) \left( \sum\_{k=1}^n b_k^2 \right)$$

Display math:

``` math
e^{i\pi} + 1 = 0
```

and inline math $\`a^2 + b^2 = c^2\`$.

## References

1.  Yu, et al. Bayesian Neural Networks.
    <https://www.cs.toronto.edu/~duvenaud/distill_bayes_net/public/>

2.  Jospin et al (2022). Hands-on Bayesian Neural Networks – A Tutorial
    for Neural Deep Learning Methods. Machine Learning.IEEE
    Computational Intelligence Magazine 17(2): 29-48

3.  Bayesian Deep Learning \[slides\]:
    <https://alinlab.kaist.ac.kr/resource/Lec8_Bayesian_DL.pdf>

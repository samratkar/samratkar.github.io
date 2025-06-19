
### What is expectation of a random variable?

1. Say we roll a die. The possible outcomes are 1, 2, 3, 4, 5, and 6.
2. Probability of each outcome is 1/6.
3. Therefore if we would like to find the contribution of each number in the die to the expected outcome, we would multiply each number by its probability:
   $E[X] = 1 * \frac{1}{6} + 2 * \frac{1}{6} + 3 * \frac{1}{6} + 4 * \frac{1}{6} + 5 * \frac{1}{6} + 6 * \frac{1}{6} = \frac{21}{6} = 3.5$
4. The expected value of the die roll is 3.5, which is the average outcome if we rolled the die many times.
5. Generalizing - 
   $$E_x[f(x)] = \int xf(x) \, dx$$
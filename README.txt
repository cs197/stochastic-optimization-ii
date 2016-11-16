Main References for this Assignment
===================================

Stochastic Optimization
-----------------------

** http://www.jhuapl.edu/spsa/PDF-SPSA/Handbook04_StochasticOptimization.pdf (particularly Section 6.3.2 and 6.3.3).

The Spall reference is excellent for high-dimensional minimization problems. In the code comments you will
see exactly which of Spall's equations I have implemented and which remain to be implemented.

Collaborative Filtering
-----------------------

** https://www.coursera.org/learn/machine-learning/lecture/f26nH/collaborative-filtering-algorithm (Lecture 98)
** https://web.stanford.edu/~lmackey/papers/cf_slides-pml09.pdf (only if you prefer slides to lectures)

An alternative to the Simultaneous Perturbation method for Stochastic Optimization,
is to use the Gradient Descent algorithm described toward the end of Ng's Lecture 98.
However, Spall's method should be much more performant in high dimensions.

Remember, the number of dimensions is:

( the number of movies + the number of users ) * the number of features

This is a very high-dimensional optimization.

Other Notes
-----------

I am at present unclear how lambda is meant to be chosen. For now, I just hard-coded it to 1.0.

I also have little feeling for how what Spall calls the gain parameters (alpha_k and c_k) are meant to be chosen,
although Spall gives some suggestions.

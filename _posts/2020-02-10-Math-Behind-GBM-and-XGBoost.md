Many of you ML enthusiasts out there might have used boosting algorithms to get the best predictions (most of the time) on your data. In this blog post, I want to demystify how these algorithms work and how are they different from others. But why would anyone want to do the tough job of looking under the hood? This post is for all the curious minds out there who want to learn and innovate new techniques to tackle unprecedented problems. Let’s get started!

To solve any supervised Machine Learning problem, given the dataset $\{(x\_i, y\_i)\}\_{i=1,\ldots,n}$ where $x$ are the features and $y$ is the target, we try to restore the function $y = f(x)$ by approximately estimating $\hat{f}(x)$ while measuring how good the mapping is using a loss function $L(y,f)$ and then take mean over the data to get the final cost, i.e., 
$\hat{f}(x) = \underset{f(x)}{\arg\min}\mathbb{E}\_{x,y}[L(y,f(x))]$

The only problem that remains is to find the $\hat{f}(x)$. Since there are infinite possibilities/combinations to create a function, the functional space is infinite-dimensional. Hence, to find a function we need to limit our search space by restricting our function to a specific structure, $f(x,\theta), \theta \in \mathbb{R}^n$. Remember your linear regression equation? There we only consider linear combination of features multiplied with the parameter, $\hat{f}(x) = \theta^Tx$, we are limiting the search space to find parameters $\theta$, this is same as that. The optimization problem has now become,
$\hat{\theta} = \underset{\theta}{\arg\min} \mathbb {E}\_{x,y}[L(y,f(x,\theta))]$, so we only need to search over $\theta$. 
We can find this by updating $\theta$  in an iterative fashion using our favorite gradient descent algorithm and come up with our estimate $\hat{\theta}$ after $T$ iterations, $\hat{\theta} = \sum\_{i = 1}^T \hat{\theta\_i}$. To start we initialize $\hat{\theta} = \hat{\theta\_0}$ and at each iteration we calculate the gradient of loss function, i.e. $-\left[\frac{\partial L(y, f(x, \theta))}{\partial \theta}\right]\_{\theta = \hat{\theta} }$ and call this $\hat{\theta\_t}$ for the $t^{th}$  step. We add this $\hat{\theta\_t}$ to our current estimate to get the new estimate, $\hat{\theta} \leftarrow \hat{\theta} + \hat{\theta\_t} = \sum\_{i = 0}^t \hat{\theta\_i}$ . We repeat this till convergence, when the gradient of the loss function is close to 0. Finally, we have the $\hat{f}(x) = f(x, \hat{\theta})$.
Let's take a pause and understand why we had to go through all this. Here, in the gradient descent algorithm we did an iterative search over the parameters. Why can't we search in the same way over functions? We can start with a randomly guessed function and keep adding new functions to get better estimate like we did with parameters. We will start with our initial estimate $\hat{f}\_0$  and then reach to our final function after $T$ iterations, $\hat{f}(x) = \sum\_{i=0}^T\hat{f\_i}(x)$.
Same as earlier, we will restrict our functions to a family, $\hat{f}(x) = g(x,\theta)$. We will also search for an optimal coefficient $\rho$ for each function we want to add. In  $t^{th}$ iteration, the problem becomes,

$\hat{f}(x) = \sum\_{i = 0}^{t-1} \hat{f\_i}(x)$
$(\rho\_t,\theta\_t) = \underset{\rho,\theta}{\arg\min}\mathbb {E}\_{x,y}[L(y,\hat{f}(x) + \rho \cdot g(x, \theta))]$
$\hat{f\_t}(x) = \rho\_t \cdot g(x, \theta\_t)$
Now, we will try to solve this using gradient descent. But how? We can calculate gradient of loss with respect to the function instead of a parameter. Suppose we are using squared error $L = (y-f)^2$, the gradient of the loss w.r.t. will be $[\frac{\partial L(y, f)}{\partial f}]\_{f=\hat{f}} = 2*(y-\hat{f})$, which is the $residual$. So the new function that needs to be added to our previous estimate should be $-residual$. This makes sense, right? We are adding a new function to the previous estimate to correct wherever it had made errors. So now in  $t^{th}$ iteration, the problem becomes,
$\hat{f}(x) = \sum\_{i = 0}^{t-1}\hat{f\_i}(x)$
$r\_{it} = -\left[\frac{\partial L(y\_i, f(x\_i))}{\partial f(x\_i)}\right]\_{f(x)=\hat{f}(x)}, \quad {for}\ i=1,\ldots,n$
$\theta\_t = \underset{\theta}{\arg\min}\sum\_{i = 1}^{n} (r\_{it} - g(x\_i, \theta))^2,$
$\rho\_t = \underset{\rho}{\arg\min}\sum\_{i = 1}^{n} L(y\_i, \hat{f}(x\_i) + \rho \cdot g(x\_i, \theta\_t))$

We can solve these above equations to find $\hat{f}$ in an iterative manner as shown below:
1. Initialize the function estimate with a constant value$\hat{f}(x) = \hat{f}\_0, \hat{f}\_0 = \gamma, \gamma \in \mathbb{R}, \hat{f}\_0 = \underset{\gamma}{\arg\min}\sum\_{i = 1}^{n} L(y\_i, \gamma)$
2. For each iteration $t = 1, \dots, T$:
	i. Calculate pseudo-residuals $r\_t$:
	 $r\_{it} = -\left[\frac{\partial L(y\_i, f(x\_i))}{\partial f(x\_i)}\right]\_{f(x)=\hat{f}(x)}, \quad{for }\ i=1,\ldots,n$
	ii. Add a new function $g\_t(x)$ as regression on pseudo-residuals $\{ (x\_i, r\_{it})\}\_{i=1, \ldots,n}$

	iii. Find optimal coefficient $\large \rho\_t$ at $g\_t(x)$ regarding initial loss function
	$\rho\_t = \underset{\rho}{\arg\min}\sum\_{i = 1}^{n} L(y\_i, \hat{f}(x\_i) + \rho \cdot g(x\_i, \theta))$
	iv. Update current approximation $\hat{f}(x)$ where $\hat{f\_t}(x) = \rho\_t \cdot g\_t(x)$

      $\hat{f}(x)\leftarrow\hat{f}(x)+\hat{f\_t}(x) = \sum\_{i = 0}^{t}\hat{f\_i}(x)$
3. Compose final GBM model $\large \hat{f}(x)$
$\hat{f}(x) = \sum\_{i = 0}^T\hat{f\_i}(x)$
This is the Gradient Boosting Machines algorithm.

Xgboost:
cntd.....




References:
[https://mlcourse.ai/articles/topic10-boosting/](https://mlcourse.ai/articles/topic10-boosting/) 

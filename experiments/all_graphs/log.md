# Experiment Log

**3 May 2016**

I first tried all graphs, no minibatches, 1,000 iters. MSE = 0.17, not bad.

Architecture: Conv -> FP -> LinReg

However, gradient boost & random forest give MSE = 0.07 and 0.09 respectively. ConvNet not beating it yet. 


------

**5 May 2016**

I decided to try all graphs again, no minibatches, 10,000 iters. Let's see how it goes.

------

**6 May 2016**

I cancelled the 10,000 iters job, and went with 5,000 iters instead.

I'm observing that overfitting may be taking place. 

Also suspect that the architecture that I'm using right (Conv -> FP -> LinReg) may not have enough learning capacity. Might want to try an alternative: Conv -> Conv -> FP -> LinReg the next time round.

------

** 7 May 2016**

Looking at the training loss, I am quite confident that the best thing to try next is a Conv->Conv->FP->LinReg architecture.

Best to do 1000 iters first, to see if it can beat a single conv layer first, and then do the two conv layer architecture.

-------

** 10 May 2016 **

Whoa! Today we reached MSE = 0.08x! That's pretty awesome, quite close to the performance of gradient boosting.

However, it took a long time to get there. Still curious, can a 2-conv layer architecture do learning faster?

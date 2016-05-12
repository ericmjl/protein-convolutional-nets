# Experiment Log

**4 May 2016**

In this directory, I try doing the following experiment:

- Conv->Conv->FP->LinReg
- 1000, 2000, 5000, 10000 iters
- open up 200 graphs (out of all of them); minibatch of 100, 20 test 80 train.

Things I want to do on the next experiment:

- Instead of opening only 200 graphs, open up all 3000+ graphs.
- Do minibatch on 500, still 20% test and 80% train.

*6:13 PM:* Submitted the jobs on Rous.

-----

11 May 2016

5:26 PM

After abandoning some other tests on other directories, I am finding that minibatch with minibatch size = 1 works best for optimization - in terms of speed.

However, I'm also seeing that it needs on the order of much more than 10,000 steps to finish optimizing. 

Previous test on whole dataset needed ~'000s of steps over 5 days to finish optimizing. I'm guessing that ~10^5 to 10^6 steps are needed for optimization to work with single data point.


------

12 May 2016

Batch size of 1 doesn't work well for optimization. Let me see if doing optimization in batches of 10, over 1000 steps, works better.

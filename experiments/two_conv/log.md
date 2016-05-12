10 May 2016

Today I will try doing a two convolutional layer network, and run it with 1000 iterations to see how it performs.

Apparently, two conv architecture does not progress to actual learning. I'm not sure why. Converting back to one_conv first...

After trying the one-conv architecture, it trains with no issues.

Okay, I think there's something wrong with doing two convolutional layers...

5:18 PM
I have added some print statements to debug what's going on. Let's see after 15 minutes...

7:14 PM
I had peeked at this earlier, but here's the official statement. It gets stuck on the 2nd convolutional layer. I'm not sure why.

Let me try digging in again...

9:36 PM
Okay, after digging through it, I found that two conv takes a ton of time. What if I instead do a fully connected layer on top of the FP instead of a 2nd conv?

10:19 PM
Finally did this thing where I brought in the old train.py file from another directory, which had two convolutional layers inside it, and used it to do training. It is working now.

10:56 PM
Okay, I found out that the old `train.py` file was doing fine because it was doing minibatches and not training well at all.

I brought back the old `train_all_graphs.py` from another directory, and used it again. Going to let it run overnight, see where it's at tomorrow.

--------

11 May 2016

7:17 AM
I inspected the training log - it wasn't training again. Alrighty. Something is wrong, but I'm not sure where. Yet, two convolutions works for synthetic data... Anyways, I deleted the job. Deleting the `train_all_graphs.py` file.
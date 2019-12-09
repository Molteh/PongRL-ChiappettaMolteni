# PongRL-ChiappettaMolteni

We started by following two main paths:

- Start from the main resource provided in the instructions for the project (2) and try to adapt
to our environment the relative REINFORCE algorithm in (4) and its improvement with an
Actor Critic approach in (5)

- Use two of the most advanced techniques studied during the course, DQN and Actor Critic,
and adapt the code produced by ourselves during the weekly assignments to this case. For
DQN, the theoretical reference was provided in (6), while for the Actor Critic we referred
again to (2)

The adaptation of the REINFORCE algorithm proposed in (4) didn't prove to be effective, having
scored no more than 8% against SimpleAi after a training process of 60000 episodes. For this reason,
we decided to move to the Actor Critic solution proposed in (5) as an improvement of the previous
one. But also in this case the solution performed in a similar way.

Having studied more intensely the drawbacks of Policy Gradient algorithms in terms of how bad
moves can ruin the performance of training (7), we decided to try implementations of TRPO (Trust
Region Policy Optimization) (8), and towards the end of the project, of PPO (Proximal Policy
Optimization). In both cases, the adaptation proved to learn very slowly and we decided to move
on. We also tried to adapt an implementation of ACER (Actor Critic with Experience Replay)
provided in (9), but the results were similar to the previous two. In all these cases, we believe that
performance should have been improving from algorithm to algorithm, but our lack of practical
experience and the differences between our environment and the one offered by Gym (3) have pre-
vented us to come up with the right learning process. Most of the sources we referred to propose
very detailed and efficient implementations, but it has also been complicated to understand what
to modify in order to make them work in our case.

What proved to perform more effective was the implementation of the algorithms learnt during
the assignments. For DQN, results started improving, but the algorithm was taking a lot of time
to learn and never went above a. 22% victory rate.

The implementation of Actor Critic based on the 5th assignment of the course was the one
performing better, after being trained using as input the state of the game, modeled by the positions
of the ball and the two paddles. At that point, we realized we could use supervised learning and
try to predict these data from the raw image made of pixel data. The implementation of this step
with a Neural Network proved to be successful, and predictions quickly reached satisfying results
with an average error of 1-2- pixels, thus only slightly disturbing the behaviour of the Actor Critic
algorithm.

After having trained the agent against SimpleAi, this implementation has guaranteed a per-
formance that, from game to game, ranged between 35% to 55% victory rate against SimpleAi.
This was the final agent and model submitted for the evaluation. We also tried to train the agent
against himself or against the agent named SomeAgent provided by the course instructors, but in
both cases, our agent did not learn a behavior that could be effective against SimpleAi.

The last approach that we tried to implement was an implementation of the Experience Replay
buffer in our Actor Critic agent, using a sample of recently played episodes (and not just one) for
each policy gradient update. However, in the little time remained, it did not learn a useful policy
and we did not have time to tune it perfectly.

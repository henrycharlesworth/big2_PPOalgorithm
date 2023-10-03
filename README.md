# Big 2 Self-Play Reinforcement Learning AI
Big 2 is a 4 player game of imperfect information with quite a complicated action space (being able to choose to play singles, pairs, three of a kinds, two pairs, straights, flushes, full houses etc from an initial starting hand of 13 cards). The aim of the game is to is to be the first player to play all of your cards but to play the game well requires formulating a long term plan, thinking about what your opponents plans are and knowing when to play a hand and when to save a hand for later. This is my implementation of training an AI to learn the game purely via self-play deep reinforcement learning using the "Proximal Policy Optimization" algorithm. The results have been surprisingly good - my friends and I play this game A LOT every time we go on holiday and it has got to the point where it convincingly beats all of us over a decent amount of games.  

If you run generateGUI.py you can play with the AI and also see the values it assigns to each state as well as the probability of choosing each option. I've also made a web app using Django so that you can play against the trained networks in a more proper setting <a href="https://big2-rl-4ba753215e7b.herokuapp.com/game/">here</a> (it may take a while to load). <a href="https://github.com/henrycharlesworth/big2_PPOalgorithm/blob/master/rules.md">Here</a> are the rules of the game.

<a href="https://big2-rl-4ba753215e7b.herokuapp.com/game/"><img src="https://henrycharlesworth.com/fileStorage/big2aiscreenshot.png" /></a>

I wrote up the details of how I trained the network and added it to arXiv <a href="https://arxiv.org/abs/1808.10442">here!</a>

## Update (October 2023)
Apologies, the Heroku app was down for a long time (lots of emails sent to an old account warning me, but I missed them). I have deployed a new app <a href="https://big2-rl-4ba753215e7b.herokuapp.com/game">here.</a>

Also as I may not keep that app running indefinitely, I have released the code so that you can run the game locally and play against the trained agents <a href="https://github.com/henrycharlesworth/big2_server/">here.</a>


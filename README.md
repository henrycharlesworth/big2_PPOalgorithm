# Big 2 Self-Play Reinforcement Learning AI
Big 2 is a 4 player game of imperfect information with quite a complicated action space (being able to choose to play singles, pairs, three of a kinds, two pairs, straights, flushes, full houses etc from an initial starting hand of 13 cards). This is my implementation of training an AI to play the game purely via self-play using the "Proximal Policy Optimization" algorithm and I was pleasantly surprised by how well it worked! My friends and I play this game A LOT every time we go on holiday and it has got to the point where it convincingly beats all of us over a decent amount of games.  

If you run generateGUI.py you can play with the AI and also see the values it assigns to each state as well as the probability of choosing each option. I've also made a web app using Django so that you can play against the trained networks in a more proper setting <a href="https://big2-ai.herokuapp.com/game/">here</a> (it may take a while to load). <a href="http://henrycharlesworth.com/singlePlayerBig2/rules.html">Here</a> are the rules of the game.

<a href="https://big2-ai.herokuapp.com/game/"><img src="https://henrycharlesworth.com/fileStorage/big2aiscreenshot.png" /></a>

Read more about the details of what I actually did to train the network <a href="https://www.henrycharlesworth.com/blog">here</a>!

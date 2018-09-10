# baselines installation guide for Windows

OpenAI's <a href="https://github.com/openai/baselines">baselines</a> does not provide an official support for Windows. 
Simply running `pip install baselines` will result in
```diff
-Failed building wheel for atari-py
```
However, atari-py is the only part of gym causing problems. Luckily, you can easily install a <a href="https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299">precompiled atari-py</a>. Follow these steps:
1.	`pip install baselines` to install all requirements apart from gym. This will result in the above error, but just proceed
2.	`pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py`
3.	`pip install baselines`

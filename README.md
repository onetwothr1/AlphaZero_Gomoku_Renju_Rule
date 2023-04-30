## AlphaZero Gomoku

Implementation of AlphaZero algorithm on Gomoku (also called Omok or Gobang) with Renju rule. I trained a model that plays on 9 * 9 board with 3000 self-play games in 5 days.

References:
- [An implementation of the AlphaZero algorithm for Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
- [Python implementation of Renju rule(written in Korean)](https://blog.naver.com/dnpc7848/221506783416)

### Example Games Between AIs
- Each move with 10 MCTS rollouts:
![rollout10]()

- Each move with 200 MCTS rollouts:
![rollout200]()

### Play
```
python play.py  
```

You can also play on Google Colab [![Opein In Colab]()]()

### Files
|File|Description|
|------|----|
|game.py|Defines the game board and the game flow.|
|player.py|Defines player.|
|renju_rule.py|Defines renju rule.|
|alphazero_net.py|Defiens Policy and Value network of AlphaZero model.|
|alphazero_agent.py|Defines an agent that utilizes a combination of policy and value networks along with MCTS.|
|encoder.py|Encodes a board into feature channels.|
|experience.py|Defines the datatype of self-play experiences and data augmentation.|
|self_play.py|Generates AI self-play experiences.|
|train.py|Trains a model with self-play experiences.|
|compare_performance.py|Competes two models.|
|bot_v_bot.py|Play a game between AIs.|
|play.py|Play a game between human and AI.|


**Tips for training:**
1. It is good to start with a 6 * 6 board and 4 in a row. For this case, we may obtain a reasonably good model within 500~1000 self-play games in about 2 hours.
2. For the case of 8 * 8 board and 5 in a row, it may need 2000~3000 self-play games to get a good model, and it may take about 2 days on a single PC.s
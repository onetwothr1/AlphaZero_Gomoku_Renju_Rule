## AlphaZero Gomoku with Renju Rule

Implementation of AlphaZero algorithm on Gomoku (also called Omok or Gobang) with Renju rule, which has the rule of forbidden moves to limit black's advantage. I trained a model that plays on 9 * 9 board with 3000 self-play games over 5 days.
<br><br>
References:
- [Deep Learning and the Game of Go](https://github.com/maxpumperla/deep_learning_and_the_game_of_go)
- [An implementation of the AlphaZero algorithm for Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
- [Python implementation of Renju rule (written in Korean)](https://blog.naver.com/dnpc7848/221506783416)
<br><br>

## Example Games Between AIs
- Each move with 10 MCTS rollouts:
<img src="https://raw.githubusercontent.com/onetwothr1/GomokuAI/main/play_10rollout.gif" width="70%" height="70%">

- Each move with 200 MCTS rollouts:
<img src="https://raw.githubusercontent.com/onetwothr1/GomokuAI/main/play_200rollout.gif" width="70%" height="70%">

## Play
Run following script from the root directory:
```
python play.py  
```

You can also play on Google Colab. [![Opein In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onetwothr1/GomokuAI/blob/main/play.ipynb)
<br><br>
## Files
|File|Description|
|------|----|
|game.py|Defines the game board and the game flow.|
|player.py|Defines player (black and white).|
|renju_rule.py|Defines Renju rule.|
|alphazero_net.py|Defiens policy and value network for the AlphaZero model.|
|alphazero_agent.py|Defines an agent that utilizes a combination of policy and value networks along with MCTS.|
|encoder.py|Encodes a game board into feature channels.|
|experience.py|Defines the data structure of self-play experiences and data augmentation.|
|self_play.py|Generates AI self-play experiences.|
|train.py|Trains a model using self-play experiences.|
|compare_performance.py|Simulates games between two agents to determine the superior one.|
|bot_v_bot.py|Runs a game between AIs.|
|play.py|Runs a game between human and AI.|

<br><br>
**Challenges in training:**
<br>

 After 2000 self-play games, the agent showed proficiency in offense but struggled with defense. To address this, I conducted an additional 1000 self-play games and implemented two actions.

1. Encouraging Defensive Moves: If the agent identified a move leading to the opponent's victory, it was prompted to try that move more, facilitating understanding of defensive strategies.

2. Adjusting Move Selection Probability: I recorded predicted losses for each move and subtracted this value from the corresponding move's total visit count. This adjustment emphasized proper defensive moves during training while reducing the probabilities of less effective moves.

Although these actions significantly improved the agent's defense, occasionally, an imbalance between offense and defense occurred. To solve this, I extended the first action also to winning scenarios. These modifications quicky improved the agent's offensive-defensive balance without extensive self-play games and training times.

# ai-nba-predict: Predicting the NBA Champion

## Project Members

| Name | Organization | Email |
| --- | --- | --- |
| Youssef Ben Khelil | Paris Institute of Digital Technology | <youssef.ben-khelil@eleve.isep.fr> |
| Leo Bertazzoli | Zurich University of Applied Sciences | <leobe@gmx.ch> |
| Chang Won Jung | Hanyang University | <richardj0916@hanyang.ac.kr> |

## I. Introduction

The National Basketball Assosciation (NBA) is a North American basketball league made up of 30 teams. Every year from October to April, the teams compete in a regular season tournament of 82 games each, in order to determine their seeding in the playoffs. The top 8 teams in each conference are placed in the playoff bracket, where the eventual winner is crowned the NBA Champion.

Because we follow the NBA it would be great if we could predict the champions of the future seasons. On the one hand its just for our interest. On the other hand, if our prediction succeeds, we could start betting on the prediction. The final goal is to predict the champion of the next season by analyzing the past 40 years and find patterns, with the help of AI, with which we can predict future champions.

## II. Dataset

Like many sports leagues, the NBA tracks and records various statistics from all of their games. These statistics range from simple statistics such as win percentage, to advanced statistics such as effective field goal percentage. [Baksetball Reference](https://www.basketball-reference.com/) provides a massive collection of NBA statistics from team stats to individual player statistics.

The dataset that we used for our project was provided by [JK-Future-Github](https://github.com/JK-Future-GitHub/NBA_Champion/tree/main), who used a web crawler to collect data from Basketball Reference. He also added additional features such as Top_3_Conference, which describes whether the team finished within the top 3 in their respective conference. We removed seasons before 1980, as that was the year when the 3 point line was introduced, and the 2023 season, as the data collected from that season was from an incomplete season.

## III. Methodology

TODO

## IV. Evaluation and Analysis

TODO

## V. Related Work

ML Libraries/Tools:

- [XGBoost (Gradient Boost/Random Forest)](https://xgboost.readthedocs.io/en/stable/index.html)
- [SHAP (ML Model Analysis)](https://shap.readthedocs.io/en/latest/index.html)

## VI. Conclusion

TODO

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

First, we divided the dataset into training and test data. Instead of randomly dividing the dataset, we decided to randomly select two seasons from each decade. The reasoning behind this is because the NBA game has evolved with time, causing changes in playstyles which altered which skillsets were considered valuable to winning (e.g. shifting of paint dominating centers to increased value of three point shooting).

The target feature is ```Champion_Win_Share```. This is defined to be the total wins that a team gets in the playoffs divided by the max number of wins that a team can get (16 wins). A team with a greater number of playoff wins will have a higher ```Champion_Win_Share```, and be considered to be more successful. The championship team will have a value of 1.

TODO: Use XGBoost's boosted trees and random forest models and train them with the training dataset. Feed the testing dataset to the models and check if they can accurately predict which NBA team won the championship.

## IV. Evaluation and Analysis

TODO: Use SHAP library to analyze and explain the results of our machine learning models, and compare the results between each model.

## V. Related Work

ML Libraries/Tools:

- [XGBoost (Gradient Boost/Random Forest)](https://xgboost.readthedocs.io/en/stable/index.html)
- [SHAP (ML Model Analysis)](https://shap.readthedocs.io/en/latest/index.html)

## VI. Conclusion

TODO

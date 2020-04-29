***** THE PERISHABLE INVENTORY PROBLEM *****

A retailer (the agent) has an inventory of a certain perishable good. 
Every morning, the retailer decides on what order to place for that specific, perishable good. 
Since zero lead time is assumed, this order is added to the inventory the same morning. 
During the day, the retailer faces a random demand which lies between known boundaries. 
At the end of the day, the retailer incurs three different costs. 
A first cost is the cost of lost sales which arises when demand could not be met by the inventory. 
Secondly, the cost of perished items is calculated as the amount of perished items multiplied by a certain cost. 
Lastly, the holding cost is calculated which is equal to the number of units still in inventory multiplied by a certain cost. 
The goal of the retailer is to place orders in such a way that the average cost per period is minimized.


Train environment: random demand between known boundaries; to train (D)RL-agents
Test environment: seeded random demand to evaluate policies
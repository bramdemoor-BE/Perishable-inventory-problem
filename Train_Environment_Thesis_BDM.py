'''
Environment in which the neural network trains.
The specific problem is the period-review replenishment model of a single, perishable lifetime.
The lifetime is fixed and known.
The demand is i.i.d. between known barriers.
The agent is bounded in his actions by a maximum order.
Costs include lost sales, holding costs, costs of perished items.
Note: a fixed ordering cost is included for further research. In this thesis, it was set at 0.
The agent has to place actions in such a way that the average cost per period is minimized.
'''
#import dependencies
import numpy as np 
from random import randint
import random


class Perishable_Inventory_Train (object):

	def __init__ (self, lifetime, poss_orders, demand_range, lost_sales_cost, holding_cost, perish_cost, fixed_order_cost, time):

		self.lifetime = lifetime
		self.poss_orders = poss_orders
		self.demand_range = demand_range
		self.lost_sales_cost = lost_sales_cost
		self.holding_cost = holding_cost
		self.perish_cost = perish_cost
		self.fixed_order_cost = fixed_order_cost
		self.time = time
		self.current_time = 0
		self.action = 0
		self.demand = 0
		self.reward = 0
		
		#create starting state
		self.agentPosition = []
		for i in range(self.lifetime):
			self.agentPosition.append(0)

		#create action space
		self.possibleActions = []
		for i in range(self.poss_orders + 1):
			self.possibleActions.append(i)


	#take an action in the environment
	def step (self, action):

		self.action = action

		#update inventory with order
		resultingState = self.agentPosition
		resultingState[0] = resultingState[0] + action

		#update inventory with random demand
		demand = randint(0, self.demand_range)
		self.demand = demand
		inv_for_calc = resultingState.copy()
		for i in range(len(resultingState)):
			if demand > 0:
				resultingState[-i - 1] = max(inv_for_calc[-i - 1] - demand, 0)
				demand = max(demand - inv_for_calc[-i -1], 0)

		#update inventory with time
		inv_for_calc = resultingState.copy()
		for i in range(len(resultingState)):
			if i == 0:
				resultingState[i] = 0
			else:
				resultingState[i] = inv_for_calc[i - 1]

		#calculate lost sales cost
		cost_lost_sales = 0
		if demand > 0:
			lost_sales = demand
		else:
			lost_sales = 0
		cost_lost_sales = lost_sales * self.lost_sales_cost

		#calculate holding cost
		cost_holding = 0
		for i in range(len(resultingState)):
			cost_holding = cost_holding + (resultingState[i] * self.holding_cost)

		#calculate cost of perished goods
		cost_perish = 0
		perished = inv_for_calc[-1]
		cost_perish = perished * self.perish_cost

		#calculate total reward (= total cost)
		reward = -cost_lost_sales - cost_holding - cost_perish
		if self.action > 0 :
			reward -= self.fixed_order_cost
		self.reward = reward

		#add one timestep
		self.current_time += 1

		return self.agentPosition, self.reward, \
				self.isFinished(self.current_time), None


	#reset the environment: starting position and again warmup periods
	def reset(self):
		
		#reset agent's inventory to all zeros
		self.agentPosition = []
		for i in range(self.lifetime):
			self.agentPosition.append(0)

		#reset the current time to zero (start the game all over)
		self.current_time = 0

		return self.agentPosition, self.current_time


	#show the period, order, demand, resulting inventory and reward
	def render(self):
		print('------------------------------')
		print('***** Period ' + str(self.current_time) + ' *****')
		print('Order placed: ' + str(self.action))
		print('Demand encountered: ' + str(self.demand))
		print('Inventory after period ' + str(self.current_time) + ': ' + str(self.agentPosition))
		print('Reward: ' + str(self.reward))
		print('------------------------------')


	#check whether the environment is finished
	def isFinished(self, current_time):
		return current_time == self.time


	#take a random action in the environment
	def randomAction (self):
		return random.choice(self.possibleActions)





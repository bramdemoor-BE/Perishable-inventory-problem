'''
Environment to compare different ordering policies to each other.
The specific problem is the period-review replenishment model of a single, perishable lifetime.
The lifetime is fixed and known.
The demand is i.i.d. between known barriers. To compare different policies, the demand follows the same random path when initialized. (seeded)
The agent is bounded in his actions by a maximum order.
Costs include lost sales, holding costs, costs of perished items.
Note: a fixed ordering cost is included for further research. In this thesis, it was set at 0.
The agent has to place actions in such a way that the average cost per period is minimized.
'''
import numpy as np 
import random


class Perishable_Inventory_Test (object):

	def __init__ (self, lifetime, poss_orders, demand_range, lost_sales_cost, holding_cost, perish_cost, fixed_order_cost, time, warmup):

		self.lifetime = lifetime
		self.poss_orders = poss_orders
		self.demand_range = demand_range
		self.lost_sales_cost = lost_sales_cost
		self.holding_cost = holding_cost
		self.perish_cost = perish_cost
		self.fixed_order_cost = fixed_order_cost
		self.time = time
		self.current_time = 0
		self.episode = 0
		self.period_demand = 0
		self.warmup_periods = warmup

		#create starting state
		self.agentPosition = []
		for i in range(self.lifetime):
			self.agentPosition.append(0)

		#create actionspace
		self.possibleActions = []
		for i in range(self.poss_orders + 1):
			self.possibleActions.append(i)

		#create same demand everytime the environment is created
		self.demand = []
		random.seed(3)
		for i in range(self.time * 52):
			self.demand.append(random.randint(0, self.demand_range))


	#take an action in the environment
	def step (self, action):

		self.action = action

		#update inventory with order
		resultingState = self.agentPosition
		resultingState[0] = resultingState[0] + action

		#update inventory with random demand
		self.period_demand = self.demand[self.current_time + (self.time * self.episode)]
		demand = self.period_demand
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

		#calculate total reward (costs), keeping in mind that in the warmup periods, no costs are incurred
		reward = -cost_lost_sales - cost_holding - cost_perish
		if self.action > 0:
			reward -= self.fixed_order_cost
		if self.current_time < self.warmup_periods:
			self.reward = 0
		else:
			self.reward = reward

		#add one timestep
		self.current_time += 1

		return self.agentPosition, self.reward, \
				self.isFinished(), None


	#reset the environment: starting position and again warmup periods
	def reset(self):
		
		#reset agent's inventory to all zeros
		self.agentPosition = []
		for i in range(self.lifetime):
			self.agentPosition.append(0)

		#reset the current time to zero (start the game all over)
		self.current_time = 0

		self.episode += 1

		return self.agentPosition, self.current_time


	#fully reset the environment. Let the random, seeded demand start again from the beginning
	def full_reset(self):

		self.episode = 0


	#show the period, order, demand, resulting inventory and reward
	def render(self):
		print('------------------------------')
		if self.current_time <= self.warmup_periods:
			print('***** Warmup period ' + str(self.current_time) + ' *****')
		else:
			print('*** Period ' + str(self.current_time) + ' ***')
		print('Order placed: ' + str(self.action))
		print('Demand encountered: ' + str(self.period_demand))
		if self.current_time <= self.warmup_periods:
			print('Inventory after warmup period ' + str(self.current_time) + ': ' + str(self.agentPosition))
		else:
			print('Inventory after period ' + str(self.current_time) + ': ' + str(self.agentPosition))
		print('Reward: ' + str(self.reward))


	#check whether the environment is finished
	def isFinished(self):
		return self.current_time == self.time


	#take a random action
	def randomAction (self):
		return random.choice(self.possibleActions)





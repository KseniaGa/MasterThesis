import numpy as np
import random
from random import randrange, uniform
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import scipy 
# calculating euclidean distance between vectors
from scipy.spatial.distance import euclidean
import sklearn
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_distances


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        
        #world.n_episode = 0
        
        # set any world properties first
        #???
        world.dim_c = 0
        #world.dim_c = 0
        num_agents = 7
        
        #AGENT SIZE MATTERS 

        world.num_agents = num_agents
        num_adversaries = 3
        num_landmarks = 1
        
        world.agent_win = False 
        world.adv_win = False
        
        world.ag_wins = 0
        world.adv_wins = 0
        
        world.n_steps = 0 
        
        #baseline = False 
        
        #world.adv_swarm_pos = []
        
 
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.win = False
            #agent.silent = False
            
            #agent.max_speed = 5
            #agent.accel = False
            agent.adversary = True if i < num_adversaries else False
            #agent.silent = True if agent.adversary else False 
            agent.silent = True
            
            agent.size = 0.12
            
            #if num_agents > 6:
            #For over 7?? 
                #OG
             #   agent.size = 0.12
            
            #agent.size = 0.11
            #else:
            #    agent.size = 0.14
                #OG
                
                
                
                
        # add landmarks
        
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.17
            #landmark.size = 0.16
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        
        #baseline = False 
        
        if world.agent_win: 
            world.ag_wins +=1 
        elif world.adv_win:
            world.adv_wins +=1
            
        
        
        world.agent_win = False 
        world.adv_win = False
        
        #world.n_episode += 1
        
        world.n_steps = 0
        
        

        world.all_pos = np.zeros(shape=(0,2))
        
        #world.ag_wins = 0
        #world.adv_wins = 0
        
        #test
        for i, agent in enumerate(world.agents):
            if agent.adversary == False:
                world.agents[i].color = np.array([0.35, 0.35, 0.85])
                
            else: 
                world.agents[i].color = np.array([0.85, 0.35, 0.35])
        
        #world.agents[0].color = np.array([0.85, 0.35, 0.35])
        #for i in range(1, world.num_agents):
        #    world.agents[i].color = np.array([0.35, 0.35, 0.85])
        
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
            
        # set goal landmark
        #goal = np.random.choice(world.landmarks)
        goal = world.landmarks[0]
        #goal_fake = world.landmarks[1]
        goal.color = np.array([0.15, 0.65, 0.15])
        
        for agent in world.agents:
            agent.goal_a = goal
            #agent.goal_b = goal_fake
        # set random initial states

        for agent in world.agents:
            agent.win = False
            
            
        goal.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p) 
        distance = np.random.uniform(+0.5, +0.5, world.dim_p)
  
        #world.adv_swarm_pos = []

        
        for agent in world.agents:
            #adv_pos = []

            if agent.adversary:  
                #agent.state.p_pos = random.choice([np.random.uniform(-0.8, -0.6, world.dim_p), np.random.uniform(0.8, 0.6, world.dim_p)])
                #if baseline == False:
                agent.state.p_pos = np.random.uniform(-0.8, -0.6, world.dim_p)
                #else: 
                    #agent.state.p_pos = np.random.uniform(-0.7, -0.5, world.dim_p)
                    #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
  
            else: 
                #if baseline == False:
                
                #agent.state.p_pos = goal.state.p_pos
                agent.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)
                
                #baselein test!
                #else:
                    #agent.state.p_pos = np.random.uniform(+0.7, +0.9, world.dim_p)
                    #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                
            world.all_pos = np.vstack((world.all_pos, agent.state.p_pos))
            
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        
        
        
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.zeros(world.dim_p)
        

            
            

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


# Agents are rewarded based on minimum agent distance to each landmark
    def reward(self, agent, world):
    
        baseline = False 

        rand_rew = uniform(-100, 100)
        boundary_reward = -20 if self.outside_boundary(agent) else 10
        #boundary_reward = boundary_reward if agent.adversary else 0
        #boundary_reward = -20 if self.outside_boundary(agent) else 10
        agent_win_rew = 50 if world.agent_win else 0 
        adv_win_rew = 50 if world.adv_win else 0 
        swarm_reward = self.swarm_reward(agent, world)
        #swarm_reward = self.swarm_reward(agent, world) if agent.adversary else 0
        
        
        main_reward = self.adversary_reward(agent, world)+adv_win_rew if agent.adversary else self.agent_reward(agent, world)+agent_win_rew
        #main_reward = self.adversary_reward(agent, world)+adv_win_rew if agent.adversary else 0
        
        
        #return swarm_reward+boundary_reward+main_reward
        return swarm_reward + main_reward + boundary_reward
        #return boundary_reward+main_reward
        
        #main_reward = self.adversary_reward(agent, world)+adv_win_rew if agent.adversary else self.agent_reward(agent, world)
        #return boundary_reward+main_reward
        


        
    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False
            
            
    def outside_boundary_landmark(self, landmark):
        if landmark.state.p_pos[0] > 1 or landmark.state.p_pos[0] < -1 or landmark.state.p_pos[1] > 1 or landmark.state.p_pos[1] < -1:
            return True
        else:
            return False
        
            
 
     
            
            
    def swarm_reward(self, agent, world):
        
        
        #other_pos = []
        
        #SUM OF REP REW FOR ALL ADV AGENTS 
        
        rep_rew_sum = []
        attr_rew_sum = []

        for other in world.agents:
            attr_thresh = 6*agent.size
            rep_thresh = 3*agent.size
             
            if other is agent: continue
            
            rep_rew = 0
            attr_rew = 0 
            #IS THAT OK? 
            dist = euclidean(agent.state.p_pos,other.state.p_pos)
            
  
            
            #BAD AGENTS SHOULDN'T COLLIDE WITH EACH OTHER OR GOOD AGENTS 
            #if agent.adversary:
            if euclidean(agent.state.p_pos,other.state.p_pos) <  rep_thresh:
                rep_rew -= 3
                rep_rew_sum.append(rep_rew)
            
            #GOOD AGENTS SHOULDN'T COLLIDE WITH EACH OTHER, BUT THEY CAN PUSH BAD AGENTS 
            #if (not agent.adversary) and (not other.adversary):
            #    if euclidean(agent.state.p_pos,other.state.p_pos) <  rep_thresh:
            #        rep_rew -= 3
            #       rep_rew_sum.append(rep_rew)  
                        
            
            #BAD AGENTS STAY WITHIN A CERTAIN THRESHHOLD 
            #if (other.adversary) and (agent.adversary):
            if (euclidean(agent.state.p_pos,other.state.p_pos) <  attr_thresh) and (euclidean(agent.state.p_pos,other.state.p_pos) >=  rep_thresh): 
                attr_rew  += 3
                attr_rew_sum.append(attr_rew)
                
            

        
        return sum(rep_rew_sum)+sum(attr_rew_sum)

        

            
        
        

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True
        targetProximityRew = True
        prox = 0
        

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
        
            #HOW FAR THE ADVERSARY AGENTS FROM THE REAL TARGET ? Pos reward 
            #TRAIN WITHOUT ADV POS KNOWLEDGE
            adv_rew1 = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            
            #??? HOW CLOSE THEY ARE TO THE Real TARGET ? Pos reward ???
            #adv_rew2 = -min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            if targetProximityRew: 
                if euclidean(agent.state.p_pos,agent.goal_a.state.p_pos) < 4*agent.size:
                    prox += 5
           
                
        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])


        return pos_rew+prox+adv_rew1
 
        
        

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        adversary_agents = self.adversaries(world) 
        targetProximityRew = True
        
        prox = 0 
        
        if shaped_reward:  
        # distance-based reward
            #adv_rew1 = -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            adv_rew1 = -min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            
            if targetProximityRew: 
                if euclidean(agent.state.p_pos,agent.goal_a.state.p_pos) < 6*agent.size:
                    prox += 5
                    
            
            #HOW FAR THE ADVERSARY AGENTS FROM THE REAL TARGET ? Neg reward SO IF FAR - NEG ? 
            #adv_rew_test = -sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            
            
            
            return adv_rew1+prox
            

    def info(self, agent, world):
        adv_wins = world.adv_wins
        ag_wins = world.ag_wins
        
        win = agent.win
        n_steps = world.n_steps
        #swarm_reward = self.swarm_reward(agent, world)
        #all_pos = world.all_pos
        
        #return {"agent wins": ag_wins, "adversary wins": adv_wins}
        adversary_agents = self.adversaries(world) 
        good_agents = self.good_agents(world)
        
        #Att_Sum_Rew = sum([self.reward(agent, world) for a in adversary_agents])
        #Def_Sum_Rew = sum([self.reward(agent, world) for a in good_agents])

        #adv_rew1 = -min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            
        #return agent.state.p_pos
        #return win
        
        #def reward(self, agent, world): 
        
         
        #SELF? 
        #B_Agent_Rew = self.reward(agent, world) if agent.adversary else 0 
        #G_Agent_Rew = self.reward(agent, world) if not agent.adversary else 0 

        #TIED TO THE PLOTS 
        return ag_wins, adv_wins
        #return Def_Sum_Rew, Att_Sum_Rew
        
        #return swarm_reward
        #return other_pos
        #return [agent.goal_a.state.p_pos - agent.state.p_pos]
        
        

        
        
    


    def done(self, agent, world): 
        done1 = False
        #WIN STATES 

        n_steps = world.n_steps

        #already_won = True if alrwon() else False
        already_won = False
       


        if agent.adversary: 
            #if euclidean(agent.state.p_pos,agent.goal_a.state.p_pos) <  2*agent.size :
            if euclidean(agent.state.p_pos,agent.goal_a.state.p_pos) <  2*agent.size :
                #adversary_reward(agent, world) += 10 
                world.adv_win = True
                agent.win = True
                #world.adv_wins += 1
                done1 = True
                
            
        else: 
            #if (n_steps>30) and (euclidean(agent.state.p_pos, agent.goal_a.state.p_pos) <  6*agent.goal_a.size):
            if (n_steps>30) and (euclidean(agent.state.p_pos, agent.goal_a.state.p_pos) <  6*agent.goal_a.size):
            #if (n_steps>45) and (euclidean(agent.state.p_pos, agent.goal_a.state.p_pos) <  6*agent.goal_a.size):
            #if (n_steps>95) and (euclidean(agent.state.p_pos, agent.goal_a.state.p_pos) <  6*agent.goal_a.size):


                #already_won = False 
                world.agent_win = True
                agent.win = True
                            
                done1 = True
                #world.ag_wins += 1
    

        return done1
        


    def observation(self, agent, world):
    
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
            
            
        goal_color = np.zeros(world.dim_color)
        
        if agent.goal_a is not None:
            goal_color = agent.goal_a.color 
            
        # communication of all other agents
        other_pos = []
        other_pos_adv = []
        
        comm = []
        
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            #comm.append(other.state.c)
            
        #for other in world.agents: 
            #if other is agent or (other.state.c is None): continue
            #comm.append(other.state.c)

        if agent.adversary:
            #CURRENT 
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
            

            
            #return np.concatenate(other_pos)
            #return np.concatenate(entity_pos + other_pos)
            
            #return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
            #return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos+[goal_color])


        #DEFENDER 
        else:
            #CURRENT 
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos]+ entity_pos + other_pos )
            

            

            
            #return np.concatenate([goal_color])
            #return np.concatenate(entity_pos + other_pos)
            


            

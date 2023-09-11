import numpy as np
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
        # set any world properties first
        world.dim_c = 1
        num_agents = 4
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = 2
        
        world.agent_win = False 
        world.adv_win = False
        
        world.ag_wins = 0
        world.adv_wins = 0
        
        world.n_steps = 0

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            #agent.silent = False
            agent.adversary = True if i < num_adversaries else False
            agent.silent = True if agent.adversary else False 
            #agent.silent = True
            
            agent.size = 0.12
            #agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.14
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        
        if world.agent_win: 
            world.ag_wins +=1 
        elif world.adv_win:
            world.adv_wins +=1
        
        
        world.agent_win = False 
        world.adv_win = False
        
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
        goal_fake = world.landmarks[1]
        goal.color = np.array([0.15, 0.65, 0.15])
        
        for agent in world.agents:
            agent.goal_a = goal
            agent.goal_b = goal_fake
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
            #agent.state.p_pos = np.random.uniform(-0.7, -0.5, world.dim_p)
            world.all_pos = np.vstack((world.all_pos, agent.state.p_pos))
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
            
            
        goal.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p) 
        distance = np.random.uniform(+0.5, +0.5, world.dim_p)
        goal_fake.state.p_pos = self.get_l2_pos(goal, goal_fake, distance)
        
        
        
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.zeros(world.dim_p)
        
 
    def get_l2_pos(self, landmark1, landmark2, d):
        
        #distance = np.random.uniform(+0.4, +0.8, world.dim_p)
        
        landmark2.state.p_pos = landmark1.state.p_pos+d 
        
        while self.outside_boundary_landmark(landmark2): 
            landmark2.state.p_pos = landmark1.state.p_pos+np.random.uniform(+0.5, +0.5, 2)
            
        else: 
            return landmark2.state.p_pos
            
            
            

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

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark

        
        #boundary_reward = -0.5 if self.outside_boundary(agent) else 0
        boundary_reward = -20 if self.outside_boundary(agent) else 10
        agent_win_rew = 50 if world.agent_win else 0 
        adv_win_rew = 50 if world.adv_win else 0 
        swarm_reward = self.swarm_reward(agent, world)
        
        main_reward = self.adversary_reward(agent, world)+adv_win_rew if agent.adversary else self.agent_reward(agent, world)+agent_win_rew

        return main_reward+swarm_reward+boundary_reward
        #return main_reward+boundary_reward


        
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
    
        rep_rew_sum = []
        attr_rew_sum = []

        for other in world.agents:
            attr_thresh = 6*agent.size
            rep_thresh = 3*agent.size
             
            if other is agent: continue
            
            rep_rew = 0
            attr_rew = 0 

            dist = euclidean(agent.state.p_pos,other.state.p_pos)
            
  
            
            # AGENTS SHOULDN'T COLLIDE WITH EACH OTHER  
            #if agent.adversary:
            if dist <  rep_thresh:
                rep_rew -= 3
                rep_rew_sum.append(rep_rew)

           
            #AGENTS STAY WITHIN A CERTAIN THRESHHOLD 
            #if (other.adversary) and (agent.adversary):
            if (dist <  attr_thresh) and (dist >=  rep_thresh): 
                attr_rew  += 3
                attr_rew_sum.append(attr_rew)
                
            

        
        return sum(rep_rew_sum)+sum(attr_rew_sum)
        


            
        
        #return swarm_reward

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
        
            #HOW FAR THE ADVERSARY AGENTS FROM THE REAL TARGET ? Pos reward 
            #adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            
            #HOW CLOSE THEY ARE TO THE FAKE TARGET ? Pos reward 
            adv_rew2 = -min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_b.state.p_pos))) for a in adversary_agents])
                    
        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_b.state.p_pos))) for a in good_agents])

        
        #return adv_rew+adv_rew2
        return adv_rew2+pos_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        adversary_agents = self.adversaries(world) 
        if shaped_reward:  
        # distance-based reward
            #adv_rew1 = -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            adv_rew1 = -min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
            
            return adv_rew1 

            
            
    def info(self, agent, world):
        adv_wins = world.adv_wins
        ag_wins = world.ag_wins
        swarm_reward = self.swarm_reward(agent, world)
        #all_pos = world.all_pos
        
        #return {"agent wins": ag_wins, "adversary wins": adv_wins}
        #return comm
        
        #comm = []
        
        #for other in world.agents: 
        #    if other is agent or (other.state.c is None): continue
        #    comm.append(other.state.c)

        return ag_wins, adv_wins


    def done(self, agent, world): 
        done1 = False

        if agent.adversary: 
            if euclidean(agent.state.p_pos,agent.goal_a.state.p_pos) <  2*agent.goal_a.size :
 
                world.adv_win = True
                #world.adv_wins += 1
                done1 = True
                
            elif euclidean(agent.state.p_pos, agent.goal_b.state.p_pos) <  2*agent.goal_b.size :
                world.agent_win = True 
                #world.ag_wins += 1
                done1 = True
  


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
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color 
            
        # communication of all other agents
        other_pos = []
        comm = []
        
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            comm.append(other.state.c)
            
        for other in world.agents: 
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)

        if not agent.adversary:
            #return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
            #return np.concatenate([goal_color]+ entity_pos + other_pos)
            return np.concatenate(entity_pos + other_pos + [goal_color])
            #return np.concatenate([goal_color])

        else:
            #return np.concatenate(entity_pos + other_pos+comm)
            return np.concatenate(entity_pos + other_pos + comm)
            #return np.concatenate(comm)
            

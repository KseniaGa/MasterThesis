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
        num_agents = 3
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = 2
        
        world.agent_win = False 
        world.adv_win = False
        
        world.ag_wins = 0
        world.adv_wins = 0
        
        
        
     
        
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            #agent.silent = False
            agent.adversary = True if i < num_adversaries else False
            agent.silent = True if agent.adversary else False 
            #agent.silent = True
            
            agent.size = 0.09
            #agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.09
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        
        world.agent_win = False 
        world.adv_win = False
        
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
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
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
        boundary_reward = -0.5 if self.outside_boundary(agent) else 0
        agent_win_rew = 50 if world.agent_win else 0 
        adv_win_rew = 50 if world.adv_win else 0 
        swarm_reward = self.swarm_reward(agent, world)
        
        
        
        main_reward = self.adversary_reward(agent, world)+adv_win_rew if agent.adversary else self.agent_reward(agent, world)+agent_win_rew
        #return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        #return main_reward+swarm_reward+boundary_reward
        return main_reward+boundary_reward


        
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
        
        current_states = dict(enumerate(world.all_pos))
        agent_states = np.array(list(current_states.values()))
        dist = DistanceMetric.get_metric('chebyshev')
        dist_1 = dist.pairwise(agent_states)
        #19/17? is the observation space of an agent ? 
        if agent.adversary:
            dist_2 = 8 - dist_1
        else: 
            dist_2 = 11 - dist_1
        
        attraction_thresh = 0.3
        repulsion_thresh = 0.1
        
        reps_1 = (dist_1 < repulsion_thresh)
        reps_2 = (dist_2 < repulsion_thresh)
        
        attr_1 = (dist_1 < attraction_thresh) == (dist_1 >= repulsion_thresh)
        attr_2 =  (dist_2 < attraction_thresh) == (dist_2 >= repulsion_thresh)
        rew_attr = attr_1.sum() + attr_2.sum()

        
        rew_rep = -(reps_1.sum() - world.num_agents + reps_2.sum())
        
        
        # Return reward according to curriculum objective
        # Normalize by the number of agents (twice - symmetry)
        #Global reward
        reward_global_repulsion =  rew_rep / (world.num_agents*(world.num_agents - 1))
        reward_global_attraction =  rew_attr / (world.num_agents*(world.num_agents - 1))
        #reward["global"]["alignment"] = reward_type["alignment"] * rew_unalign / (self.num_agents*(self.num_agents - 1))
        reward_global_sum = reward_global_repulsion + reward_global_attraction
        
        for agent_id in range(world.num_agents):
            rew_rep_i = -(reps_1[agent_id, :].sum() + reps_2[agent_id, :].sum() - 1)
            rew_attr_i = attr_1[agent_id, :].sum() + attr_2[agent_id, :].sum()

            reward_ag_rep=  rew_rep_i/ (world.num_agents*(world.num_agents - 1))
            reward_ag_attr =  rew_attr_i/ (world.num_agents*(world.num_agents - 1))
            #reward[agent_id]["alignment"] = reward_type["alignment"] * rew_unalign_i/ (self.num_agents*(self.num_agents - 1))
            reward_sum = 100*(reward_ag_rep+reward_ag_attr)
            
            
        
        return reward_sum

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
            
            
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            pos_rew = 0 
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 1
                elif np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) > 0.7:
                    pos_rew += 1
                
        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_b.state.p_pos))) for a in good_agents])
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < 2 * agent.goal_a.size:
                pos_rew += 5
            pos_rew -= min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        
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
            
        else:  # proximity-based reward (binary)
            adv_rew2 = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew2 += 2
            return adv_rew2
            
            
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


   
        #return ag_wins, adv_wins
        return ag_wins, adv_wins


    def done(self, agent, world): 
        done1 = False
        #WIN STATES 
        #adv_win = False
        #agent_win = False 

        if agent.adversary: 
            if euclidean(agent.state.p_pos,agent.goal_a.state.p_pos) <  2*agent.goal_a.size :
                #adversary_reward(agent, world) += 10 
                world.adv_win = True
                world.adv_wins += 1
                done1 = True
                
            elif euclidean(agent.state.p_pos, agent.goal_b.state.p_pos) <  2*agent.goal_b.size :
                world.agent_win = True 
                world.ag_wins += 1
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
            

import numpy as np
import argparse
from copy import deepcopy
import torch
from torch.optim import Adam
import torch.nn.functional as F

from external_agents.DDPGs.model import (Actor_Reurrent, Critic)
from external_agents.DDPGs.memory import EpisodicMemory
from external_agents.DDPGs.random_process import OrnsteinUhlenbeckProcess
from external_agents.DDPGs.util import *

criterion = nn.MSELoss()        
USE_CUDA = torch.cuda.is_available()


class Agent(object):
    def __init__(self, nb_states, nb_actions, n_predicates, n_entities, n_responses, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions

        # Create Actor and Critic Network
        self.actor = Actor_Reurrent(self.nb_states, self.nb_actions, init_w=args.init_w)
        self.actor_target = Actor_Reurrent(self.nb_states, self.nb_actions, init_w=args.init_w)

        self.critic = Critic(self.nb_states, self.nb_actions, args.init_w)
        self.critic_target = Critic(self.nb_states, self.nb_actions, args.init_w)

        hard_update(self.actor_target, self.actor)   # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        # Create replay buffer
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.trajectory_length = args.trajectory_length
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.is_training = True
        # 
        if USE_CUDA: self.cuda()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        return action

    def select_action(self, state, decay_epsilon=True):
        action, _ = self.actor(to_tensor(np.array([state])))
        action = to_numpy(action).squeeze(0)
        action += self.is_training * max(self.epsilon, 0)*self.random_process.sample()   # change 0 in max if necessary

        action = np.clip(action, -1., 1.)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        return action

    def reset_lstm_hidden_state(self, done=True):
        self.actor.reset_lstm_hidden_state(done)

    def reset(self, s_t):
        self.random_process.reset_states()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def load_weights(self, output):
        if output is None: return False

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

        return True

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )


class RDPG(object):
    def __init__(self, args, nb_states, nb_actions, n_predicates, n_entities, n_responses):
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions

        self.agent = Agent(nb_states, nb_actions, args)
        self.memory = EpisodicMemory(capacity=args.rmsize, max_episode_length=args.trajectory_length, window_length=args.window_length)

        self.critic_optim  = Adam(self.agent.critic.parameters(), lr=args.rate)
        self.actor_optim  = Adam(self.agent.actor.parameters(), lr=args.prate)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.trajectory_length = args.trajectory_length
        self.max_episode_length = args.max_episode_length
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.warmup = args.warmup
        self.validate_steps = args.validate_steps

        # 
        self.epsilon = 1.0
        self.is_training = True
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action

        # 
        if USE_CUDA: self.cuda()

    def eval(self):
        self.agent.eval()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1
 
    def random_action(self):
        return self.agent.random_action()

    def select_action(self, state, decay_epsilon=True):
        return self.agent.select_action(state, decay_epsilon=True)

    def reset(self, s_t):
        self.agent.reset()
        self.s_t = s_t

    def train(self, num_iterations, checkpoint_path, debug):
        self.agent.is_training = True
        step = episode = episode_steps = trajectory_steps = 0
        episode_reward = 0.
        state0 = None
        while step < num_iterations:
            episode_steps = 0
            while episode_steps < self.max_episode_length:
                # reset if it is the start of episode
                if state0 is None:
                    state0 = deepcopy(self.env.reset())
                    self.agent.reset()

                # agent pick action ...
                if step <= self.warmup:
                    action = self.agent.random_action()
                else:
                    action = self.agent.select_action(state0)

                # env response with next_observation, reward, terminate_info
                state, reward, done, info = self.env.step(action)
                state = deepcopy(state)

                self.env.render()

                # agent observe and update policy
                self.memory.append(state0, action, reward, done)

                # update 
                step += 1
                episode_steps += 1
                trajectory_steps += 1
                episode_reward += reward
                state0 = deepcopy(state)

                if trajectory_steps >= self.trajectory_length: 
                    self.agent.reset_lstm_hidden_state(done=False)          # Attention here
                    trajectory_steps = 0
                    if step > self.warmup:
                        self.update_policy()

                # [optional] save intermideate model
                if step % int(num_iterations/3) == 0:
                    self.agent.save_model(checkpoint_path)

                if done: # end of episode
                    if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

                    # reset
                    state0 = None
                    episode_reward = 0.
                    episode += 1
                    self.agent.reset_lstm_hidden_state(done=True)
                    break

            # [optional] evaluate
            # if self.evaluate is not None and self.validate_steps > 0 and step % self.validate_steps == 0:
            #     policy = lambda x: self.agent.select_action(x, decay_epsilon=False)
            #     validate_reward = self.evaluate(self.env, policy, debug=False, visualize=False)
            #     if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

#            if step >= args.warmup and episode > args.bsize:
#                # Update weights
#                agent.update_policy()


    def update_policy(self):
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        if len(experiences) == 0: # not enough samples
            return

        policy_loss_total = 0
        value_loss_total = 0
        for t in range(len(experiences) - 1): # iterate over episodes
            target_cx = Variable(torch.zeros(self.batch_size, 50)).type(FLOAT)
            target_hx = Variable(torch.zeros(self.batch_size, 50)).type(FLOAT)

            cx = Variable(torch.zeros(self.batch_size, 50)).type(FLOAT)
            hx = Variable(torch.zeros(self.batch_size, 50)).type(FLOAT)

            # we first get the data out of the sampled experience
            state0 = np.stack((trajectory.state0 for trajectory in experiences[t]))
            # action = np.expand_dims(np.stack((trajectory.action for trajectory in experiences[t])), axis=1)
            action = np.stack((trajectory.action for trajectory in experiences[t]))
            reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
            # reward = np.stack((trajectory.reward for trajectory in experiences[t]))
            state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))

            # compute target q values
            target_action, (target_hx, target_cx) = self.agent.actor_target(to_tensor(state1, volatile=True), (target_hx, target_cx))
            next_q_value = self.agent.critic_target([
                to_tensor(state1, volatile=True),
                target_action
            ])
            next_q_value.volatile=False
            target_q = to_tensor(reward) + self.discount*next_q_value

            # Critic update
            current_q = self.agent.critic([to_tensor(state0), to_tensor(action)])

            # value_loss = criterion(q_batch, target_q_batch)
            value_loss = F.smooth_l1_loss(current_q, target_q)
            value_loss /= len(experiences) # divide by trajectory length
            value_loss_total += value_loss

            # Actor update
            print("size compare", to_tensor(state0).shape, cx.shape)
            exit()
            
            action, (hx, cx) = self.agent.actor(to_tensor(state0), (hx, cx))
            policy_loss = -self.agent.critic([
                to_tensor(state0),
                action
            ])

            policy_loss /= len(experiences)    # divide by trajectory length
            policy_loss_total += policy_loss.mean()

            # update per trajectory
            self.agent.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

            self.agent.actor.zero_grad()
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()

        # Target update
        soft_update(self.agent.actor_target, self.agent.actor, self.tau)
        soft_update(self.agent.critic_target, self.agent.critic, self.tau)

        # update only once
#        policy_loss_total /= self.batch_size # divide by number of trajectories
#        value_loss_total /= self.batch_size # divide by number of trajectories
#
#        self.agent.critic.zero_grad()
#        value_loss_total.backward()
#        self.critic_optim.step()
#
#        self.agent.actor.zero_grad()
#        policy_loss_total.backward()
#        self.actor_optim.step()

    def load_weights(self, output):
        self.agent.load_weights(output)

    def save_model(self, output):
        self.agent.save_model(output)

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
import os
import sys
import subprocess
from ray.rllib import train

from random import shuffle
import torch.optim as optim
import torch.nn as nn

from azureml.core import Run

from collections import OrderedDict
from azureml.core import Model
from datetime import datetime
import torch
from files import wrappers
from files import dqn_model
from files.rollout_actor import RolloutActor
import time
import os
import sys
from collections import deque
from random import shuffle

import numpy as np



import logging


log = logging.getLogger(__name__)

BATCH_SIZE = 200
BUFFER_SIZE = 20000
TRAIN_START_SIZE = BATCH_SIZE*5+1
LEARNING_RATE = 1e-3
VALIDATION_STEPS = 1
TEST_STEPS = 1000

MEAN_REWARD_BOUND = 19

GAMMA = 0.99
# BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 5
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 200
EPSILON_START = 1.0
EPSILON_FINAL = 0.01



trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
done_rewards=[]
model_name ='pong_atari'

DEFAULT_RAY_ADDRESS = 'localhost:6379'
import  ray

def getCheckpointFile(iteration):
    return 'checkpoint_' + str(iteration) + '.pth.tar'

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_v
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)

def sample(trainExamples, batch_size):
    indices = np.random.choice(len(trainExamples), batch_size,
                               replace=False)
    states, actions, rewards, dones, next_states = \
        zip(*[trainExamples[idx] for idx in indices])
    return np.array(states), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(next_states)



if __name__ == "__main__":

    # Parse arguments
    run = Run.get_context()
    ws = run.get_context().experiment.workspace
    try:
        amlmodel = Model(name=model_name,workspace=ws)
        os.makedirs("model", exist_ok=True)
        amlmodel.download(target_dir="model", exist_ok=True)
        print("model connected successfully")
        print("content of  model folder: ", os.listdir("model"))
    except Exception as e:
        print("model does not exist", e)

    train_parser = train.create_parser()
    ray_args = train_parser.parse_args()
    num_workers=ray_args.config["num_workers"]
    env_name = ray_args.env
    # device=args.config["head_gpu"]
#     replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("detected device is ", device)
    env = wrappers.make_env(env_name)

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)



    if ray_args.ray_address is None:
        ray_args.ray_address = DEFAULT_RAY_ADDRESS

    ray.init(address=ray_args.ray_address)
    # Start simulations on actors
    try:
        net.load_state_dict(torch.load("model/"+os.listdir("model")[0]))
        print("model loaded successfully")

    except Exception as e:
        print("cannot load from disk", e)



    step_idx = 0
    best_idx = 0
    version_advancement=0
    master_speed_time=0
    replay_buffer_comp_acc=0
    init=True

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    iterationTrainExamples = deque([], maxlen=REPLAY_SIZE)
    agents = [RolloutActor.remote(i, env_name) for i in range(num_workers)]
    epsilon =EPSILON_START
    while True:
        state_dict = net.state_dict()
        state_dict = OrderedDict({k: v.to('cpu') for k, v in state_dict.items()})
        replay_buffer_list = [agent.executeEpisode.remote(state_dict, 3, epsilon) for agent in agents]
        # examples of the iteration


        # replay_buffer_output = ray.get(replay_buffer_list)

        done_id, replay_buffer_list = ray.wait(replay_buffer_list)

        # done_count=+1
        # print(ray.get(done_id)[0])
        replay_buffer_output= ray.get(done_id)

        for output in replay_buffer_output:
            iterationTrainExamples += output[0]
            done_rewards.append(output[1])
            replay_buffer_list.extend([agents[output[2]].executeEpisode.remote(state_dict, 1, epsilon)])


        # shuffle examples before training

        # print(trainExamples[3])
        if len(iterationTrainExamples)<=TRAIN_START_SIZE:
            continue
        print("size of examples ", len(iterationTrainExamples))
        shuffle(iterationTrainExamples)
        # print(trainExamples[0])
        best_m_reward = None
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = sum(done_rewards)/len(done_rewards)
        total_rewards.append(reward)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
        m_reward = np.mean(total_rewards[-100:])
        print("%d: done %d games, reward %.3f, "
              "eps %.2f, speed %.2f f/s" % (
            frame_idx, len(total_rewards), m_reward, epsilon,
            speed
        ))
        run.log("Frame", frame_idx)
        run.log("reward_100", m_reward)
        run.log("reward", reward)

        if best_m_reward is None or best_m_reward < m_reward:
            torch.save(net.state_dict(), ray_args.env +
                       "-best_%.0f.dat" % m_reward)
            if best_m_reward is not None:
                print("Best reward updated %.3f -> %.3f" % (
                    best_m_reward, m_reward))
            best_m_reward = m_reward
        if m_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break



        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            # state_dict = tgt_net.state_dict()
            # state_dict = OrderedDict({k: v.to('cpu') for k, v in state_dict.items()})
            # replay_buffer_list = [agent.executeEpisode.remote(state_dict, 3, epsilon) for agent in agents]
        for _ in range(len(iterationTrainExamples)//5):
            optimizer.zero_grad()
            batch = sample( iterationTrainExamples,BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()




        # training new network, keeping a copy of the old one
        #
        #         try:
        #             run.upload_file(name=destination_name, path_or_stream=file_name)
        #         except Exception as e:
        #             print("error uploading, the file may already exists", e)
        #
        #         run.register_model(model_name, destination_name, tags={"test_negam":
        #                                                                         sum(result_test_play) / len(result_test_play)})






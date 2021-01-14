import numpy as np
import ray
from . import wrappers
from . import dqn_model
import torch


TRAIN_START_SIZE = 20
STEPS_BEFORE_T = 10



@ray.remote
class RolloutActor(object):
    """Actor object to start running simulation on workers.
        Gradient computation is also executed on this object."""
    def __init__(self, actor_id, env_name):
        # starts simulation environment, policy, and thread.
        # Thread will continuously interact with the simulation environment
        self.id = actor_id

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = wrappers.make_env(env_name)
        self.net = dqn_model.DQN(self.env.observation_space.shape,
                        self.env.action_space.n)


    def executeEpisode(self,state_dict,num_episode,epsilon=0.0):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        if state_dict:
            #             print("load new net weight")
            self.net.load_state_dict(state_dict)
        trainExamples = []
        state = self.env.reset()
        episodeStep = 0
        done_reward = []
        for _ in range(num_episode):
            is_done=False
            total_reward = 0.0
            while(not is_done):
                if np.random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    state_a = np.array([state], copy=False)
                    state_v = torch.tensor(state_a)
                    q_vals_v = self.net(state_v)
                    _, act_v = torch.max(q_vals_v, dim=1)
                    action = int(act_v.item())

                # do step in the environment
                new_state, reward, is_done, _ = self.env.step(action)
                trainExamples.append((state, action, reward, is_done,new_state))
                total_reward += reward
                state = new_state
                if is_done:
                    done_reward.append(total_reward)
                    state = self.env.reset()
        return trainExamples,sum(done_reward)/len(done_reward),self.id

        # while True:
        #     episodeStep += 1
        #     canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
        #     temp = int(episodeStep < 2)
        #
        #     pi = mcts.getActionProb(canonicalBoard, temp=temp)
        #     sym = self.game.getSymmetries(canonicalBoard, pi)
        #     for b, p in sym:
        #         trainExamples.append([b, curPlayer, p, None])
        #
        #     action = np.random.choice(len(pi), p=pi)
        #     board, curPlayer = self.game.getNextState(board, curPlayer, action)
        #
        #     r = self.game.getGameEnded(board, curPlayer)
        #
        #     if r != 0:
        #         return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples], self.id



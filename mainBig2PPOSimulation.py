#main big2PPOSimulation class

import numpy as np
from PPONetwork import PPONetwork, PPOModel
from big2Game import vectorizedBig2Games
import tensorflow as tf
import joblib
import copy


#taken directly from baselines implementation - reshape minibatch in preparation for training.
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])



class big2PPOSimulation(object):
    
    def __init__(self, sess, *, inpDim = 412, nGames = 8, nSteps = 20, nMiniBatches = 4, nOptEpochs = 5, lam = 0.95, gamma = 0.995, ent_coef = 0.01, vf_coef = 0.5, max_grad_norm = 0.5, minLearningRate = 0.000001, learningRate, clipRange, saveEvery = 500):
        
        #network/model for training
        self.trainingNetwork = PPONetwork(sess, inpDim, 1695, "trainNet")
        self.trainingModel = PPOModel(sess, self.trainingNetwork, inpDim, 1695, ent_coef, vf_coef, max_grad_norm)
        
        #player networks which choose decisions - allowing for later on experimenting with playing against older versions of the network (so decisions they make are not trained on).
        self.playerNetworks = {}
        
        #for now each player uses the same (up to date) network to make it's decisions.
        self.playerNetworks[1] = self.playerNetworks[2] = self.playerNetworks[3] = self.playerNetworks[4] = self.trainingNetwork
        self.trainOnPlayer = [True, True, True, True]
        
        tf.global_variables_initializer().run(session=sess)
        
        #environment
        self.vectorizedGame = vectorizedBig2Games(nGames)
        
        #params
        self.nGames = nGames
        self.inpDim = inpDim
        self.nSteps = nSteps
        self.nMiniBatches = nMiniBatches
        self.nOptEpochs = nOptEpochs
        self.lam = lam
        self.gamma = gamma
        self.learningRate = learningRate
        self.minLearningRate = minLearningRate
        self.clipRange = clipRange
        self.saveEvery = saveEvery
        
        self.rewardNormalization = 5.0 #divide rewards by this number (so reward ranges from -1.0 to 3.0)
        
        #test networks - keep network saved periodically and run test games against current network
        self.testNetworks = {}
        
        # final 4 observations need to be carried over (for value estimation and propagating rewards back)
        self.prevObs = []
        self.prevGos = []
        self.prevAvailAcs = []
        self.prevRewards = []
        self.prevActions = []
        self.prevValues = []
        self.prevDones = []
        self.prevNeglogpacs = []
        
        #episode/training information
        self.totTrainingSteps = 0
        self.epInfos = []
        self.gamesDone = 0
        self.losses = []
        
    def run(self):
        #run vectorized games for nSteps and generate mini batch to train on.
        mb_obs, mb_pGos, mb_actions, mb_values, mb_neglogpacs, mb_rewards, mb_dones, mb_availAcs = [], [], [], [], [], [], [], []
        for i in range(len(self.prevObs)):
            mb_obs.append(self.prevObs[i])
            mb_pGos.append(self.prevGos[i])
            mb_actions.append(self.prevActions[i])
            mb_values.append(self.prevValues[i])
            mb_neglogpacs.append(self.prevNeglogpacs[i])
            mb_rewards.append(self.prevRewards[i])
            mb_dones.append(self.prevDones[i])
            mb_availAcs.append(self.prevAvailAcs[i])
        if len(self.prevObs) == 4:
            endLength = self.nSteps
        else:
            endLength = self.nSteps-4
        for _ in range(self.nSteps):
            currGos, currStates, currAvailAcs = self.vectorizedGame.getCurrStates()
            currStates = np.squeeze(currStates)
            currAvailAcs = np.squeeze(currAvailAcs)
            currGos = np.squeeze(currGos)
            actions, values, neglogpacs = self.trainingNetwork.step(currStates, currAvailAcs)
            rewards, dones, infos = self.vectorizedGame.step(actions)
            mb_obs.append(currStates.copy())
            mb_pGos.append(currGos)
            mb_availAcs.append(currAvailAcs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(list(dones))
            #now back assign rewards if state is terminal
            toAppendRewards = np.zeros((self.nGames,))
            mb_rewards.append(toAppendRewards)
            for i in range(self.nGames):
                if dones[i] == True:
                    reward = rewards[i]
                    mb_rewards[-1][i] = reward[mb_pGos[-1][i]-1] / self.rewardNormalization
                    mb_rewards[-2][i] = reward[mb_pGos[-2][i]-1] / self.rewardNormalization
                    mb_rewards[-3][i] = reward[mb_pGos[-3][i]-1] / self.rewardNormalization
                    mb_rewards[-4][i] = reward[mb_pGos[-4][i]-1] / self.rewardNormalization
                    mb_dones[-2][i] = True
                    mb_dones[-3][i] = True
                    mb_dones[-4][i] = True
                    self.epInfos.append(infos[i])
                    self.gamesDone += 1
                    print("Game %d finished. Lasted %d turns" % (self.gamesDone, infos[i]['numTurns']))
        self.prevObs = mb_obs[endLength:]
        self.prevGos = mb_pGos[endLength:]
        self.prevRewards = mb_rewards[endLength:]
        self.prevActions = mb_actions[endLength:]
        self.prevValues = mb_values[endLength:]
        self.prevDones = mb_dones[endLength:]
        self.prevNeglogpacs = mb_neglogpacs[endLength:]
        self.prevAvailAcs = mb_availAcs[endLength:]
        mb_obs = np.asarray(mb_obs, dtype=np.float32)[:endLength]
        mb_availAcs = np.asarray(mb_availAcs, dtype=np.float32)[:endLength]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)[:endLength]
        mb_actions = np.asarray(mb_actions, dtype=np.float32)[:endLength]
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)[:endLength]
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        #discount/bootstrap value function with generalized advantage estimation:
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        for k in range(4):
            lastgaelam = 0
            for t in reversed(range(k, endLength, 4)):
                nextNonTerminal = 1.0 - mb_dones[t]
                nextValues = mb_values[t+4]
                delta = mb_rewards[t] + self.gamma * nextValues * nextNonTerminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextNonTerminal * lastgaelam
        
        mb_values = mb_values[:endLength]
        #mb_dones = mb_dones[:endLength]
        mb_returns = mb_advs + mb_values
        
        return map(sf01, (mb_obs, mb_availAcs, mb_returns, mb_actions, mb_values, mb_neglogpacs))
        
    def train(self, nTotalSteps):

        nUpdates = nTotalSteps // (self.nGames * self.nSteps)
        
        for update in range(nUpdates):
            
            alpha = 1.0 - update/nUpdates
            lrnow = self.learningRate * alpha
            if lrnow < self.minLearningRate:
                lrnow = self.minLearningRate
            cliprangenow = self.clipRange * alpha
            
            states, availAcs, returns, actions, values, neglogpacs = self.run()
            
            batchSize = states.shape[0]
            self.totTrainingSteps += batchSize
            
            nTrainingBatch = batchSize // self.nMiniBatches
            
            currParams = self.trainingNetwork.getParams()
            
            mb_lossvals = []
            inds = np.arange(batchSize)
            for _ in range(self.nOptEpochs):
                np.random.shuffle(inds)
                for start in range(0, batchSize, nTrainingBatch):
                    end = start + nTrainingBatch
                    mb_inds = inds[start:end]
                    mb_lossvals.append(self.trainingModel.train(lrnow, cliprangenow, states[mb_inds], availAcs[mb_inds], returns[mb_inds], actions[mb_inds], values[mb_inds], neglogpacs[mb_inds]))
            lossvals = np.mean(mb_lossvals, axis=0)
            self.losses.append(lossvals)
            
            newParams = self.trainingNetwork.getParams()
            needToReset = 0
            for param in newParams:
                if np.sum(np.isnan(param)) > 0:
                    needToReset = 1
                    
            if needToReset == 1:
                self.trainingNetwork.loadParams(currParams)
            
            if update % self.saveEvery == 0:
                name = "modelParameters" + str(update)
                self.trainingNetwork.saveParams(name)
                joblib.dump(self.losses,"losses.pkl")
                joblib.dump(self.epInfos, "epInfos.pkl")

    
if __name__ == "__main__":
    import time
    
    with tf.Session() as sess:
        mainSim = big2PPOSimulation(sess, nGames=64, nSteps=20, learningRate = 0.00025, clipRange = 0.2)
        start = time.time()
        mainSim.train(1000000000)
        end = time.time()
        print("Time Taken: %f" % (end-start))
        
        
        
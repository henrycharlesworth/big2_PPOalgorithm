import tensorflow as tf
import numpy as np
from baselines.a2c.utils import fc
import joblib

class PPONetwork(object):
    
    def __init__(self, sess, obs_dim, act_dim, name):
        #action is continuous in R^act_dim.
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = name
        
        with tf.variable_scope(name):
            X = tf.placeholder(tf.float32, [None, obs_dim], name="input")
            activation = tf.nn.relu
            h1 = activation(fc(X,'fc1', nh=512, init_scale=np.sqrt(2)))
            h2 = activation(fc(h1,'fc2', nh=256, init_scale=np.sqrt(2)))
            mean = fc(h2,'mean', act_dim, init_scale=np.sqrt(2), init_bias = 6.5)
            logstd = fc(h2,'logstd', act_dim, init_scale=np.sqrt(2), init_bias= 1.15)
            std = tf.exp(logstd)
            #value function - share layer h1
            h3 = activation(fc(h1,'fc3', nh=256, init_scale=np.sqrt(2)))
            vf = fc(h3, 'vf', 1)[:,0]
            
        def sample():
            return mean + std * tf.random_normal(tf.shape(mean))
        
        def neglogpacfunc(x):
            return 0.5*tf.reduce_sum(tf.square((x-mean) / std), axis=-1) + 0.5*np.log(2.0*np.pi)*tf.to_float(tf.shape(x)[-1]) + tf.reduce_sum(logstd, axis=-1)
        
        a0 = sample()
        neglogpac = neglogpacfunc(a0)
        
        def step(obs):
            a, v, neglogp = sess.run([a0, vf, neglogpac], {X:obs})
            return a, v, neglogp
            
        def value(obs):
            return sess.run(vf, {X:obs})
        
        self.vf = vf
        self.neglogpac = neglogpacfunc
        self.X = X
        self.mean = mean
        self.logstd = logstd
        self.std = std
        self.step = step
        self.value = value
        
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        def getParams():
            return sess.run(self.params)
        
        self.getParams = getParams
        
        def loadParams(paramsToLoad):
            restores = []
            for p, loadedP in zip(self.params, paramsToLoad):
                restores.append(p.assign(loadedP))
            sess.run(restores)
            
        self.loadParams = loadParams
        
        def saveParams(path):
            modelParams = sess.run(self.params)
            joblib.dump(modelParams, path)
            
        self.saveParams = saveParams
     
        
        
class PPOModel(object):
    
    def __init__(self, sess, network, inpDim, actDim, ent_coef, vf_coef, max_grad_norm):
        
        self.network = network
        
        #placeholder variables
        ACTIONS = tf.placeholder(tf.float32, [None, actDim], name='actionsPlaceholder')
        ADVANTAGES = tf.placeholder(tf.float32, [None], name='advantagesPlaceholder')
        RETURNS = tf.placeholder(tf.float32, [None], name='returnsPlaceholder')
        OLD_NEG_LOG_PROB_ACTIONS = tf.placeholder(tf.float32,[None], name='oldNegLogProbActionsPlaceholder')
        OLD_VAL_PRED = tf.placeholder(tf.float32,[None], name='oldValPlaceholder')
        LEARNING_RATE = tf.placeholder(tf.float32,[], name='LRplaceholder')
        CLIP_RANGE = tf.placeholder(tf.float32,[], name='cliprangePlaceholder')
        
        neglogpac = network.neglogpac(ACTIONS)
        entropy = tf.reduce_sum(network.logstd + 0.5*np.log(2*np.pi*np.e), axis=-1)
        
        #define loss functions
        #entropy loss
        entropyLoss = tf.reduce_mean(entropy)
        #value loss
        v_pred = network.vf
        v_pred_clipped = OLD_VAL_PRED + tf.clip_by_value(v_pred - OLD_VAL_PRED, -CLIP_RANGE, CLIP_RANGE)
        vf_losses1 = tf.square(v_pred - RETURNS)
        vf_losses2 = tf.square(v_pred_clipped - RETURNS)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        #policy gradient loss
        prob_ratio = tf.exp(OLD_NEG_LOG_PROB_ACTIONS - neglogpac)
        pg_losses1 = -ADVANTAGES * prob_ratio
        pg_losses2 = -ADVANTAGES * tf.clip_by_value(prob_ratio, 1.0-CLIP_RANGE, 1.0+CLIP_RANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))
        #total loss
        loss = pg_loss + vf_coef*vf_loss - ent_coef*entropyLoss
        
        params = network.params
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        
        def train(lr, cliprange, observations, returns, actions, values, neglogpacs):
            advs = returns - values
            advs = (advs-advs.mean()) / (advs.std() + 1e-8)
            inputMap = {network.X: observations, ACTIONS: actions, ADVANTAGES: advs, RETURNS: returns,
                        OLD_VAL_PRED: values, OLD_NEG_LOG_PROB_ACTIONS: neglogpacs, LEARNING_RATE: lr, CLIP_RANGE: cliprange}
            return sess.run([pg_loss, vf_loss, entropyLoss, _train], inputMap)[:-1]
        
        self.train = train
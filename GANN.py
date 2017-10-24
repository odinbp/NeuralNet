import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
import getData as GWG
import pandas as pd

# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py

class Gann():

    def __init__(self, dims, cman, initWeightRange, hiddenActFunct, mapBatchSize, costFunct, lrate=.1,showint=None,mbs=10,vint=None,softmax=False):
        self.learning_rate = lrate
        self.layer_sizes = dims # Sizes of each layer of neurons
        self.show_interval = showint # Frequency of showing grabbed variables
        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        self.mapLayer_grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.mapLayer_grabvar_figures = [] # One matplotlib figure for each grabvar
        self.dendrogram_grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.dendrogram_grabvar_figures = [] # One matplotlib figure for each grabvar
        self.mapBatchSize = mapBatchSize
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.softmax_outputs = softmax
        self.modules = []
        self.build(initWeightRange, hiddenActFunct, costFunct)

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())

    def add_mapLayer_grabvar(self,module_index,type='out'):
        self.mapLayer_grabvars.append(self.modules[module_index].getvar(type))
        self.mapLayer_grabvar_figures.append(PLT.figure())

    def add_dendrogram_grabvar(self,module_index,type='out'):
        self.dendrogram_grabvars.append(self.modules[module_index].getvar(type))
        self.dendrogram_grabvar_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self,module): self.modules.append(module)

    def build(self, initWeightRange, hiddenActFunct, costFunct):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input; insize = num_inputs
        # Build all of the modules
        for i,outsize in enumerate(self.layer_sizes[1:]):
            gmod = Gannmodule(self,i,invar,insize,outsize, initWeightRange, hiddenActFunct)
            invar = gmod.output; insize = gmod.outsize
        self.output = gmod.output # Output of last module is output of whole network
        if self.softmax_outputs: self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning(costFunct)

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self, costFunct = 'MSE'):
        if costFunct == 'MSE':
            self.error = tf.reduce_mean(tf.square(self.target - self.output),name='MSE')
        else: 
            self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.target,name='CE'))
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error,name='Backprop')

    def do_training(self,sess,cases,epochs=100,continued=False):
        if not(continued): self.error_history = []
        for i in range(epochs):
            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)
            for cstart in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases,cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                k,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                         feed_dict=feeder,step=step,show_interval=self.show_interval)
                error += grabvals[0]
            self.error_history.append((step, error/nmb))
            self.consider_validation_testing(step,sess, mbs)
        self.global_training_step += epochs
        TFT.plot_training_history(self.error_history,self.validation_history, xtitle="Epoch",ytitle="Error",
                                  title="",fig=not(continued))

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    def do_testing(self,sess,cases,msg='Testing',bestk=1):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        if msg == 'Total Training':
            self.trainingScore = (100*testres/len(cases))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    def do_mapLayer_mapping(self,sess,cases,msg='Testing'):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.predictor
        testres, grabvals, _ = self.run_one_step(self.test_func, self.mapLayer_grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        return testres, grabvals  # self.error uses MSE, so this is a per-case value when bestk=None

    def do_dendrogram_mapping(self,sess,cases,msg='Testing'):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.predictor
        testres, grabvals, _ = self.run_one_step(self.test_func, self.dendrogram_grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        return testres, grabvals # self.error uses MSE, so this is a per-case value when bestk=None


    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session,self.caseman.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess,bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing',bestk=bestk)

    def mapping_session(self,sess,bestk=None):
        cases = self.caseman.get_training_cases()[:self.mapBatchSize]
        if len(cases) > 0:
            test, mapLayerVals = self.do_mapLayer_mapping(sess,cases,msg='Mapping')
            test, dendrogramVals = self.do_dendrogram_mapping(sess,cases,msg='Mapping')
        if(len(mapLayerVals)>0):
            self.display_mapvars(mapLayerVals, self.mapLayer_grabvars, mode = 'h', cases = cases)
        if(len(dendrogramVals)>0):
            self.display_mapvars(dendrogramVals, self.dendrogram_grabvars, mode = 'd', cases = cases)

    def consider_validation_testing(self,epoch,sess, mbs):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess,cases,msg='Validation Testing', bestk=1)
                self.validation_history.append((epoch,error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess,bestk=None):
        res = self.do_testing(sess,self.caseman.get_training_cases(),msg='Total Training',bestk=bestk)

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars];
        msg = "Grabbed Variables at Step " + str(step)
        #print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            #if names: print("   " + names[i] + " = ", end="\n")
            if names[i].split(':')[0][-1]=="s":
                v = np.array([v])
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix
                TFT.hinton_plot(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                fig_index += 1
                #TFT.display_matrix(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))   For matrix plot
                #fig_index += 1
            #else:
                #print(v, end="\n\n")

    def display_mapvars(self, grabbed_vals, grabbed_vars, mode = 'h', cases = None):
        names = [x.name for x in grabbed_vars];
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if mode == 'h': # If v is a matrix
                TFT.hinton_plot(v,fig=self.mapLayer_grabvar_figures[fig_index],title= names[i])
                fig_index += 1
            elif mode == 'd': # If v is a matrix
                labels = []
                for j in cases:
                    labels.append(''.join(str(e) for e in j[0]))
                grabbed_vals = grabbed_vals[0].tolist()
                TFT.dendrogram(grabbed_vals, labels)
        
        
        

    def run(self,epochs=100,sess=None,continued=False,bestk=1):
        PLT.ion()
        self.training_session(epochs,sess=sess,continued=continued)
        self.test_on_trains(sess=self.current_session,bestk=bestk)
        self.testing_session(sess=self.current_session,bestk=bestk)
        if self.mapBatchSize != 0:
            self.mapping_session(sess=self.current_session)
        self.close_current_session(view=False)
        PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self,epochs=100,bestk=1):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True,bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self,view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self,ann,index,invariable,insize,outsize, initWeightRange, hiddenActFunct):
        self.ann = ann
        self.insize=insize  # Number of neurons feeding into this module
        self.outsize=outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-"+str(self.index)
        self.build(initWeightRange, hiddenActFunct)

    def build(self, initWeightRange, hiddenActFunct):
        mona = self.name; n = self.outsize
        self.weights = tf.Variable(np.random.uniform(initWeightRange[0], initWeightRange[1], size=(self.insize,n)),
                                   name=mona+'-wgt',trainable=True) # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(initWeightRange[0], initWeightRange[1], size=n),
                                  name=mona+'-bias', trainable=True)  # First bias vector
        self.output = getattr(tf.nn, hiddenActFunct)(tf.matmul(self.input,self.weights)+self.biases,name=mona+'-out')
        self.ann.add_module(self)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)

# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self,cfunc,vfrac=0,tfrac=0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases




#   ****  MAIN functions ****
def run_network(dataSource='yeast.txt', hidden = [40, 15], lrate = 0.5, epochs = 100, vfrac=0.1, tfrac=0.1, mbs = 118, sm = True, initWeightRange = (-0.3, 0.3), hiddenActFunct = 'relu', costFunct = 'MSE', displayWeights = [], displayBiases = [], mapLayers = [], mapDendrograms = [], mapBatchSize = 0, bestk = 1, caseFraction = 1, noClases = 11, vint=100, showint=10000):
    case_generator = (lambda: GWG.getData(dataSource, caseFraction,noClases))
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    mbs = mbs if mbs else len(cman.training_cases)

    inputSize = len(cman.cases[0][0])
    outputSize = len(cman.cases[0][1])
    dimensions = []
    dimensions.append(inputSize)
    for element in hidden:
        dimensions.append(element)
    dimensions.append(outputSize)
    ann = Gann(dims=dimensions,cman=cman,lrate=lrate,showint=showint,mbs=mbs,vint=vint,softmax=sm, initWeightRange = initWeightRange, hiddenActFunct=hiddenActFunct, costFunct = costFunct, mapBatchSize = mapBatchSize)

    for i in displayWeights:
        ann.add_grabvar(i,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    
    for i in displayBiases:
        ann.add_grabvar(i,'bias') # Add a grabvar (to be displayed in its own matplotlib window).

    for i in mapLayers:
        ann.add_mapLayer_grabvar(i,'out') # Add a grabvar (to be map tested and visualized in its own matplotlib window).

    for i in mapDendrograms:
        ann.add_dendrogram_grabvar(i,'out') # Add a grabvar (to be map tested and made dendrogram from in its own matplotlib window).
    
    ann.run(epochs, bestk = bestk)
    #ann.runmore(epochs*2, bestk = bestk)

    return ann.trainingScore

run_network(dataSource='yeast.txt', hidden = [200], lrate = 0.5, epochs = 5000, vfrac=0.1, tfrac=0.1, mbs = 118, sm = True, initWeightRange = (-0.3, 0.3), hiddenActFunct = 'tanh', costFunct = 'CE', displayWeights = [], displayBiases = [], mapLayers = [], mapDendrograms = [], mapBatchSize = 0, bestk = 1, caseFraction = 1, noClases = 11)






'''
def run_config(file='config.xlsx',linenr=2):
    df = pd.read_excel(file, sheetname='Sheet1')
    print("Column headings:")
    print(df.columns)
    strings = ['dataSource','lrate','vfrac','tfrac','sm','hiddenActFunct', 'costFunct', 'bestk', 'caseFraction']
    ints = ['noClases', 'epochs', 'mbs','vint', 'mapBatchSize', 'vint', 'showint']
    lists = ['hidden','initWeightRange', 'displayWeights','displayBiases', 'mapLayers', 'mapDendrograms']
    args = {}   #{'data_s':'glass.txt','dims':[9,9,8]}

    for s in strings:
        if pd.notnull(df[s][linenr-2]):
            args[s] = df[s][linenr-2]
        # Below is just a hack
        if 'bestk' in args:
            if args['bestk'] == 'None':
                args['bestk'] = None
            else:
                args['bestk'] = 1
    for i in ints:
        if pd.notnull(df[i][linenr-2]):
            args[i] = int(df[i][linenr-2])
    for l in lists:
        if pd.notnull(df[l][linenr-2]):
            if ';' in df[l][linenr-2]:
                args[l] = list(map(int,df[l][linenr-2].split(';'))) # :TODO crashes if list not contains ','
            else:
                args[l] = int(df[l][linenr-2])

    return run_network(**args)

f = open('result.txt', 'w')
for i in range(2, 102):
    f.write(str(i)+ "  :    " + str(run_config(linenr = i))+'\n')
f.close()

'''


require 'optim'
require 'dplm'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()

cmd:text('Options:')
cmd:option('--learningRate', 1e-2, 'learning rate at t=0')
cmd:option('--minLR', 1e-5, 'minimal learning rate')
cmd:option('--maxWait', 0, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.5, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--gradClip', 5, 'max magnitude of individual grad params')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--evalSize', 100, 'size of context used for evaluation (more means more memory). With --bidirectional, specifies number of steps between each bwd rnn forget() (more means longer bwd recursions)')
cmd:option('--cuda', -1, '> 0 means use CUDA with specified device id')
cmd:option('--maxEpoch', 10, 'maximum number of epochs to run')
cmd:option('--maxTries', 4, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--uniform', 8e-2, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--id',dp.uniqueID(),'name of experiment, defaults to dp.uniqueID() generator')
cmd:option('--log',dp.SAVE_DIR,'path of log directory')

--[[ recurrent layer ]]--
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--bidirectional', false, 'use a Bidirectional RNN/LSTM (nn.BiSequencer instead of nn.Sequencer)')
cmd:option('--rho', 64, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', '{200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs are stacked')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', 0.1, 'apply dropout after lookup and before softmax')

--[[ data ]]--
cmd:option('--trainEpochSize', 400000, 'number of train examples seen between each epoch')
--cmd:option('--validEpochSize', 24000, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--dataPath', '', 'path to data directory')
cmd:option('--splitFractions', '{1,0,0}',  'fractions of dataset used for train/valid/test')
cmd:option('--useText', false,  'use textset instead of sentenceset')
cmd:option('--maxSize', 1000,  'when using sentence level training, this is where sentences are cut off.')

--[[ Model file ]]--
cmd:option('--modelFile','','path to load and/or write final model to/from')
cmd:option('--overwrite',false,'overwrite existing model with new one')


cmd:text()
opt = cmd:parse(arg or {})
print("Starting experiment: " .. opt.id)
opt.hiddenSize = dp.returnString(opt.hiddenSize)
opt.splitFractions = dp.returnString(opt.splitFractions)
if opt.splitFractions[1] == 1 then
  opt.trainOnly = true
  print "Training only, because split fraction for training set to 1!"
end
if not opt.silent then
  table.print(opt)
end

if opt.bidirectional and not opt.silent then
  print("Warning : the Perplexity of a bidirectional RNN/LSTM isn't "..
      "necessarily mathematically valid as it uses P(x_t|x_{/neq t}) "..
      "instead of P(x_t|x_{<t}), which is used for unidirectional RNN/LSTMs. "..
      "You can however still use predictions to measure pseudo-likelihood.")
end

if opt.xpPath ~= '' then
  -- check that saved model exists
  assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
end

--[[Saved experiment]]--
if opt.xpPath ~= '' then
  if opt.cuda > 0 then
    require 'cunnx'
    cutorch.setDevice(opt.cuda)
  end
  xp = torch.load(opt.xpPath)
  if opt.cuda then
    xp:cuda()
  else
    xp:float()
  end
  xp:run(ds)
  os.exit()
end

--[[Model]]--
local charlm
-- language model
if not paths.filep(opt.modelFile) or opt.overwrite then
    local lm = nn.Sequential()

    local inputSize = opt.hiddenSize[1]
    if type(inputSize) == "table" then
      inputSize = inputSize[1]
    end
    for i,hiddenSize in ipairs(opt.hiddenSize) do
      local is_lstm = opt.lstm
      if type(hiddenSize) == "table" then
        is_lstm = hiddenSize[2] ~= "rnn"
        hiddenSize = hiddenSize[1]
      end

      if i~= 1 and not is_lstm then
        lm:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
      end

      -- recurrent layer
      local rnn
      if is_lstm then
        -- Long Short Term Memory
        rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
      else
        -- simple recurrent neural network
        rnn = nn.Recurrent(
          hiddenSize, -- first step will use nn.Add
          nn.Identity(), -- for efficiency (see above input layer)
          nn.Linear(hiddenSize, hiddenSize), -- feedback layer (recurrence)
          nn.Tanh(), -- transfer function
          99999 -- maximum number of time-steps per sequence
        )
        if opt.zeroFirst then
          -- this is equivalent to forwarding a zero vector through the feedback layer
          rnn.startModule:share(rnn.feedbackModule, 'bias')
        end
        rnn = nn.Sequencer(rnn)
      end

      lm:add(rnn)

      inputSize = hiddenSize
    end

    if opt.bidirectional then
      -- initialize BRNN with fwd, bwd RNN/LSTMs
      local bwd = lm:clone()
      bwd:reset()
      bwd:remember('neither')
      local brnn = nn.BiSequencerLM(lm, bwd)

      lm = nn.Sequential()
      lm:add(brnn)

      inputSize = inputSize*2
    end

    if opt.dropout > 0 then -- dropout it applied at end of recurrence
        lm:add(nn.Sequencer(nn.Dropout(opt.dropout)))
    end

    -- input layer (i.e. word embedding space)
    lm:insert(nn.SplitTable(1,2), 1) -- tensor to table of tensors

    if opt.dropout > 0 then
      lm:insert(nn.Dropout(opt.dropout), 1)
    end
    local lookup
    if type(opt.hiddenSize[1]) == "table" then
      lookup = nn.LookupTable(ds:vocabularySize(), opt.hiddenSize[1][1])
    else
      lookup = nn.LookupTable(ds:vocabularySize(), opt.hiddenSize[1])
    end
    lookup.maxOutNorm = -1 -- disable maxParamNorm on the lookup table
    lm:insert(lookup, 1)

    -- output layer
    if #ds:vocabulary() > 50000 then
      print("Warning: you are using full LogSoftMax for last layer, which "..
          "is really slow (800,000 x outputEmbeddingSize multiply adds "..
          "per example. Try --softmaxtree instead.")
    end
    local softmax = nn.Sequential()
    softmax:add(nn.Linear(inputSize, ds:vocabularySize()))
    softmax:add(nn.LogSoftMax())

    lm:add(nn.Sequencer(softmax))

    local num_params = 0
    if opt.uniform > 0 then
      for k,param in pairs(lm:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
        num_params = num_params + param:nElement()
      end
    end

    print("Number of parameters: " .. num_params)

    -- we should always remember last state until manually calling forget, even for sentence sampler
    lm:remember('both')

    ds = dplm.SplitCharSource{
      context_size=1, --opt.bidirectional and opt.rho+1 or opt.rho,
      recurrent=true, bidirectional=opt.bidirectional,
      name='rnnlm', data_path = opt.dataPath, split_fractions = opt.splitFractions,
      sentence= not opt.useText, context_size=opt.bidirectional and opt.rho+1 or opt.rho
    }

    charlm = dplm.CharLM(lm,ds.vocab)
else
    charlm = torch.load(opt.modelFile)
    ds = dplm.SplitCharSource{
      context_size=1, --opt.bidirectional and opt.rho+1 or opt.rho,
      recurrent=true, bidirectional=opt.bidirectional, vocab = charlm.vocab,
      name='rnnlm', data_path = opt.dataPath, split_fractions = opt.splitFractions,
      sentence= not opt.useText, context_size=opt.bidirectional and opt.rho+1 or opt.rho
    }
end

local lm = charlm.model





--[[Propagators]]--
ad = dp.ThresholdedAdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}

optim_state = {learningRate = opt.learningRate, beta1 = 0 }

local params, grad_params -- initialized later, after it is clear which device is used (cpu, cuda, ...)

local training_sampler = opt.useText and dp.TextSampler{epoch_size = opt.trainEpochSize, batch_size = opt.batchSize}
    or dplm.LargeSentenceSampler{epoch_size = opt.trainEpochSize, batch_size = opt.batchSize, max_size=opt.maxSize,
      context_size = opt.rho}

train = dp.Optimizer{
  loss = nn.ModuleCriterion(
    nn.SequencerCriterion(nn.ClassNLLCriterion()),
    nn.Identity(),
    opt.cuda and nn.Sequencer(nn.Convert()) or nn.Identity()
  ),
  epoch_callback = function(model, report) -- called every epoch
    if report.epoch > 0 then
      if ad.decay ~= 1 then
        optim_state.learningRate = optim_state.learningRate*ad.decay
        optim_state.learningRate = math.max(opt.minLR, optim_state.learningRate)
      else
        if opt.modelFile ~= '' then
          torch.save(opt.modelFile,charlm)
          print("CharLM saved to " .. opt.modelFile)
        end
      end
      if not opt.silent then
        print("learningRate", optim_state.learningRate)
        if opt.meanNorm then
          print("mean gradParam/param norm", opt.meanNorm)
        end
      end
    end
  end,
  callback = function(model, report) -- called every batch
    grad_params:clamp(-opt.gradClip,opt.gradClip)
    optim.adam(function(x) return report.loss, grad_params end, params, optim_state)
    local norm = grad_params:norm() -- / params:norm()
    opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
    --model:maxParamNorm(opt.maxOutNorm) -- affects params
    grad_params:zero() -- affects gradParams
    if training_sampler._end_of_batch then model:forget() end
  end,
  feedback = dp.Perplexity(),
  sampler = training_sampler,
  progress = opt.progress
}

if not opt.trainOnly then
  local valid_sampler = opt.useText and dp.TextSampler{epoch_size = -1, batch_size = opt.batchSize}
      or dplm.LargeSentenceSampler{epoch_size = -1, batch_size = opt.batchSize, max_size=opt.maxSize,
      context_size = opt.rho}
  valid = dp.Evaluator{
    feedback = dp.Perplexity(),
    sampler = valid_sampler,
    progress = opt.progress,
    callback = function(model, report) -- called every batch
      if valid_sampler._end_of_batch then model:forget() end
    end,
  }
end

--[[Experiment]]--
xp = dp.Experiment{
  id=dp.ObjectID(opt.id),
  model = lm,
  optimizer = train,
  validator = valid,
  --tester = tester,
  observer = {
    ad,
    dp.FileLogger(opt.log),
    dp.EarlyStopper{
      start_epoch=0,
      max_epochs = opt.maxTries,
      error_report={opt.trainOnly and 'optimizer' or 'validator','feedback','perplexity','ppl'}
    }
  },
  random_seed = 123,
  max_epoch = opt.maxEpoch,
  target_module = nn.SplitTable(1,1):type('torch.IntTensor')
}

--[[GPU or CPU]]--
if opt.cuda > 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.cuda)
  xp:cuda()
end

params, grad_params = lm:getParameters()

xp:verbose(not opt.silent)
if not opt.silent then
  print"Language Model :"
  print(lm)
end

xp:run(ds)

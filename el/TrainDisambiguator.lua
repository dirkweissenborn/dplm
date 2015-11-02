require 'optim'
require 'dplm'

--[[command line arguments]]--
cmd = torch.CmdLine()

--[[ training ]]--
cmd:option('--learningRate', 1e-3, 'learning rate at t=0')
cmd:option('--minLR', 1e-4, 'minimal learning rate')
cmd:option('--maxWait', 0, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.5, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--gradClip', 5, 'max magnitude of individual grad params')
cmd:option('--batchSize', 1, 'number of examples per batch')
cmd:option('--cuda', -1, '> 0 means use CUDA with specified device id')
cmd:option('--maxEpoch', 10, 'maximum number of epochs to run')
cmd:option('--maxTries', 4, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--id',dp.uniqueID(),'name of experiment, defaults to dp.uniqueID() generator')
cmd:option('--log',dp.SAVE_DIR,'path of log directory')

--[[ adaption ]]--
cmd:option('--adaptLR', 1e-3, 'learning rate at t=0')

--[[ recurrent layer ]]--
cmd:option('--rho', 100, 'back-propagate through time (BPTT) for rho time-steps')

--[[ data ]]--
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--dataDir', '', 'directory containing corpus data')
cmd:option('--maxSize', 1e10,  'when using sentence level training, this is where sentences are cut off.')

--[[ Model file ]]--
cmd:option('--modelFile','','path to write final model to')
cmd:option('--modelFileIn','','path to load model from')
cmd:option('--candmap','','path to candidate map (Lexicon)')
cmd:option('--facts','','path to facts (Lexicon)')


cmd:text()
opt = cmd:parse(arg or {})

verbose = not opt.silent
print("Starting experiment: " .. opt.id)

-- load model
local disambiguator = dplm.Disambiguator(opt.candmap, opt.facts, opt.modelFileIn)

local candmap = disambiguator._candmap
local charLM = disambiguator._lm
lm = charLM.model

-- we should always remember last state until manually calling forget, even for sentence sampler
lm:remember('both')


-- load data ---------------------------------------
local corpus = dplm.AIDALoader(opt.dataDir)

corpus.train = tablex.sub(corpus.train,1,100)
--corpus.valid   = tablex.sub(corpus.valid,1,10)

-- extract candidates ------------------------------
function extract_candidates(corpus,t)
  if corpus then
    for _,doc in ipairs(corpus) do
      for _,anno in ipairs(doc.annos) do
        t[anno.gs] = anno.gs
      end
    end
  end
end
local concepts = {}
extract_candidates(corpus.train,concepts)
extract_candidates(corpus.valid,concepts)
extract_candidates(corpus.test,concepts)

-- sample negative corpus
function sample_negative_corpus()
  local neg_concepts = {}
  local function sample(corpus)
    if corpus then
      local neg_corpus = tablex.deepcopy(corpus)
      for _,doc in ipairs(neg_corpus) do
        for _,anno in ipairs(doc.annos) do
          local mention = doc.document:sub(anno.c_start, anno.c_end-1)
          local candidates = candmap:get(mention)
          if candidates and #candidates > 1 then
            local neg_cand = candidates[torch.random(#candidates)]
            while neg_cand == anno.gs do neg_cand = candidates[torch.random(#candidates)] end
            anno.gs = neg_cand
            neg_concepts[neg_cand] = neg_cand
          end
        end
      end
      return neg_corpus
    end
  end
  local neg_corpus = { train = sample(corpus.train),
    valid = sample(corpus.valid),
    test = sample(corpus.test)
  }
  return neg_corpus, neg_concepts
end


-- prepare datasets ----
function corpus2string(corpus)
  if corpus then
    local str_parts = {}
    for _,doc in ipairs(corpus) do
      local off = 1
      local part = ""
      local str = doc.document
      for _,anno in ipairs(doc.annos) do
        part = part .. str:sub(off, anno.c_end-1) .. "[" .. anno.gs .. "]"
        off = anno.c_end
      end
      part = stringx.replace(part .. str:sub(off, string.len(str)),'\n',' ')
      table.insert(str_parts,part)
    end
    return table.concat(str_parts,'\n')
  end
end

local pos_ds = dplm.CharSource{
  recurrent=true, bidirectional=opt.bidirectional, string=true,
  train=corpus2string(corpus.train), valid=corpus2string(corpus.valid),
  sentence=true, context_size=opt.bidirectional and opt.rho+1 or opt.rho,
  vocab = charLM.vocab
}
local neg_corpus, neg_concepts = sample_negative_corpus()
local neg_ds = dplm.CharSource{
  string=true, recurrent=true, bidirectional=opt.bidirectional,
  train=corpus2string(neg_corpus.train), valid=corpus2string(neg_corpus.valid),
  sentence=true, vocab = charLM.vocab
}

local PNSampler, parent = torch.class("PosNegSampler","dp.Sampler")

function PNSampler:__init(config)
  parent.__init(self,config)
  self._pos_sampler = dplm.LargeSentenceSampler{epoch_size = -1, batch_size = opt.batchSize,
    context_size=opt.bidirectional and opt.rho+1 or opt.rho, max_size = opt.maxSize, shuffle=false}
  self._neg_sampler = dplm.LargeSentenceSampler{epoch_size = -1, batch_size = opt.batchSize,
    context_size=opt.bidirectional and opt.rho+1 or opt.rho, max_size = opt.maxSize, shuffle=false}
end

function PNSampler:sampleEpoch(dataset)
  dataset = dp.Sampler.toDataset(dataset)
  local neg_dataset
  if dataset:whichSet() == 'test' then
    neg_dataset = neg_ds:testSet()
  elseif dataset:whichSet() == 'valid' then
    neg_dataset = neg_ds:validSet()
  else neg_dataset = neg_ds:trainSet() end

  local pos_it = self._pos_sampler:sampleEpoch(dataset)
  local neg_it = self._neg_sampler:sampleEpoch(neg_dataset)
  self._next_neg = false
  self._one_done = false


  return function(batch)
    local ret
    self._end_of_batch = false
    if self._next_neg then
      ret = {neg_it(batch) }
      self._end_of_batch = self._neg_sampler._end_of_batch
      if not self._one_done and self._end_of_batch then self._next_neg = false end
    else
      ret = {pos_it(batch) }
      self._end_of_batch = self._pos_sampler._end_of_batch
      if not self._one_done and self._end_of_batch then self._next_neg = true end
    end

    if #ret > 0 then
      return unpack(ret)
    elseif self._one_done then
      return
    else
      self._one_done = true
      if self._next_neg then return neg_it(batch)
      else return pos_it(batch) end
    end
  end
end

train_sampler = PosNegSampler{epoch_size = -1, batch_size = opt.batchSize }

-- train --------------------------------------------
local optim_state = {learningRate = opt.learningRate, beta1 = 0 }
local params, grad_params -- initialized later, after it is clear which device is used (cpu, cuda, ...)

local criterion = nn.SwitchCriterion(nn.ClassNLLCriterion())
local seqCriterion = nn.ModuleCriterion(
  nn.SequencerCriterion(criterion),
  nn.Identity(),
  opt.cuda and nn.Sequencer(nn.Convert()) or nn.Identity()
)
local ad = dp.ThresholdedAdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}


train = dp.Optimizer{
  loss = seqCriterion,
  feedback = dplm.LossFeedback{criterion = seqCriterion, target_module = nn.SplitTable(1,1):type('torch.IntTensor')},
  epoch_callback = function(model, report) -- called every epoch
    if report.epoch > 0 then
      if ad.decay ~= 1 then
        optim_state.learningRate = optim_state.learningRate*ad.decay
        optim_state.learningRate = math.max(opt.minLR, optim_state.learningRate)
      end
      if not opt.silent then
        print("learningRate", optim_state.learningRate)
        if opt.meanNorm then
          print("mean gradParam/param norm", opt.meanNorm)
        end
      end
      if opt.modelFile ~= '' then
        torch.save(opt.modelFile, charlm)
        print("CharLM saved to " .. opt.modelFile)
      end
      print("=== Sampling new negative corpus. ===")
      local neg_corpus, neg_concepts = sample_negative_corpus()
      neg_ds = dplm.CharSource{
        string=true, recurrent=true, bidirectional=opt.bidirectional,
        train=corpus2string(neg_corpus.train), valid=corpus2string(neg_corpus.valid),
        sentence=true, context_size=opt.bidirectional and opt.rho+1 or opt.rho,
        vocab = charLM.vocab
      }
    end
    criterion.negate = false
    print("=== Adapting to concepts of corpus. ===")
    disambiguator:adapt_to(tablex.merge(concepts, neg_concepts),
      {learningRate=opt.adaptLR, batch_size = opt.batchSize, rho=opt.rho}, verbose)
    print("=== Continue Training. ===")
    collectgarbage()
  end,
  callback = function(model, report) -- called every batch
    -- flip weights of loss for next example (switching between positive and negative)
    if train_sampler._next_neg then criterion.negate = true
    else criterion.negate = false end

    if train_sampler._end_of_batch then
      if not train_sampler._next_neg then --update only after one batch of positive and negative documents
        grad_params:clamp(-opt.gradClip,opt.gradClip)
        optim.sgd(function(x) return report.loss, grad_params end, params, optim_state)
        local norm = grad_params:norm() / params:norm()
        opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
        --model:maxParamNorm(opt.maxOutNorm) -- affects params
        grad_params:zero() -- affects gradParams
      end
      model:forget()
    end
  end,
  sampler = train_sampler,
  verbose = false,
  progress = verbose
}


if not opt.trainOnly then
  local valid_sampler = PosNegSampler{epoch_size = -1, batch_size = opt.batchSize }
  valid = dp.Evaluator {
    feedback = dplm.LossFeedback{criterion = seqCriterion, target_module = nn.SplitTable(1,1):type('torch.IntTensor')},
    sampler = valid_sampler,
    progress = opt.progress,
    callback = function(model, report) -- called every batch
      if valid_sampler._next_neg then criterion.negate = true
      else criterion.negate = false end
      if valid_sampler._end_of_batch then model:forget() end
    end,
    epoch_callback = function(model, report) -- called every epoch
    criterion.negate = false
    end
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
      max_epochs = opt.maxTries,
      error_report={opt.trainOnly and 'optimizer' or 'validator','feedback','criterion'}
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

xp:run(pos_ds)

if opt.modelFile ~= '' then
  torch.save(opt.modelFile, disambiguator._lm)
  print("CharLM saved to " .. opt.modelFile)
end

disambiguator:close()

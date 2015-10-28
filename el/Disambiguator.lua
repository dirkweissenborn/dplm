
local Disambiguator = torch.class("dplm.Disambiguator")

function Disambiguator:__init(candmap, facts, model, args)
  self._candmap = dplm.DBMap(candmap)
  self._facts = dplm.DBMap(facts)
  self._lm = torch.load(model)

  -- adaption parameters
  args = args or {}
  self._learningRate  = args.learningRate or 1e-2
  self._rho = args.rho or 100
  self._batch_size = args.batch_size or 1
  self._max_size = args.max_size or 1000
  self._max_epochs = args.max_epochs or 2
end


--[[
-- Mentions of form: { 1: {start:..., end:...[, goldstandard:...]} }
--]]
function Disambiguator:disambiguate(str, mentions)

end

function Disambiguator:close()
  self._candmap:close()
  self._facts:close()
end

function Disambiguator:adapt_to(concepts, verbose, opts)
  opts = opts or {}
  local lr  = opts.learningRate or self._learningRate
  local rho = opts.rho or self._rho
  local batch_size = opts.batch_size or self._batch_size
  local max_size = opts.max_size or self._max_size
  local max_epochs = opts.max_epochs or self._max_epochs

  verbose = verbose or false
  require("optim")
  -- retrieve facts for concepts
  local facts = self._facts:getAll(concepts)
  local text = {}
  for k,v in pairs(facts) do
    for _,f in pairs(v) do
      if string.len(f) > 0 then
        table.insert(text, k .. " " .. f)
      end
    end
  end
  text = table.concat(text, '\n')
  --[[create data tensor that can be used by sentence sampler
  local length = 0
  _.each(facts, function(k,v) _.each(v, function (k2,v2) length = length + string.len(k) + string.len(v2) + 2 end) end)
  local d = torch.IntTensor(length,2)
  local start = 1
  _.each(facts, function(concept,v) _.each(v, function (k2,fact)
    local str = concept .. " " .. fact .. "\n"
    local len = string.len(str)
    d:sub(start,start+len-1, 1, 1):fill(start)
    self._lm:to_tensor(str, d:select(2,2):sub(start,start+len-1))
    start = start + len
  end) end)

  local trainSet = dp.SentenceSet{
    data=d, which_set="train", end_id=self._lm:endId(), start_id=self._lm:startId(),
    words=self._lm:vocab_size(), recurrent=self._recurrent
  } --]]

  local ds = dplm.CharSource{ train = text, string = true, sentence = true, vocab=self._lm.vocab }

  local sampler = dplm.LargeSentenceSampler{batch_size = batch_size, max_size=max_size, context_size = rho }

  local optim_state = { learningRate = lr, beta1 = 0}
  local params, grad_params = self._lm:getParameters()

  local optimizer = dp.Optimizer{
    loss = nn.ModuleCriterion(
      nn.SequencerCriterion(nn.ClassNLLCriterion()),
      nn.Identity(),
      nn.Sequencer(nn.Convert())
    ),
    callback = function(model, report) -- called every batch
      grad_params:clamp(-5,5)
      optim.adam(function(x) return report.loss, grad_params end, params, optim_state)
      grad_params:zero() -- affects gradParams
      if sampler._end_of_batch then model:forget() end
    end,
    feedback = dp.Perplexity{verbose = verbose},
    sampler = sampler,
    progress = verbose,
    verbose = verbose
  }

  --[[Experiment]]--
  local xp = dp.Experiment{
    model = self._lm.model,
    optimizer = optimizer,
    random_seed = 123,
    max_epoch = max_epochs,
    target_module = nn.SplitTable(1,1):type('torch.IntTensor')
  }
  xp:cuda()

  xp:verbose(verbose)
  xp:run(ds)
  collectgarbage()
end


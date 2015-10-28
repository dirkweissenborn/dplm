-- convert for use with dp.Loss instread of nn.Criteria.
-- make non-composite
------------------------------------------------------------------------
--[[ LossFeedback ]]-- COPY
-- Feedback
-- Adapter that feeds back and accumulates the error of one or many
-- nn.Criterion. Each supplied nn.Criterion requires a name for 
-- reporting purposes. Default name is typename minus module name(s)
------------------------------------------------------------------------
local LossFeedback, parent = torch.class("dplm.LossFeedback", "dp.Feedback")
LossFeedback.isLossFeedback = true

function LossFeedback:__init(config)
  assert(type(config) == 'table', "Constructor requires key-value arguments")
  local args, criterion, target_module, name = xlua.unpack(
    {config},
    'Criterion', nil,
    {arg='criterion', type='nn.Criterion', req=true,
      help='list of criteria to monitor'},
    {arg='target_module', type='nn.Module'},
    {arg='name', type='string', default='criterion'}
  )
  config.name = name
  parent.__init(self, config)

  self._criterion = criterion
  self._target_module = target_module or nn.Identity()
  self:reset()
end

function LossFeedback:setup(config)
  parent.setup(self, config)
  self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function LossFeedback:_reset()
  -- reset error sums to zero
  self._error = 0
  self._n_sample = 0
end

function LossFeedback:doneEpoch(report)
  if self._n_sample > 0 and self._verbose then
    print(self:name() .. " loss = ".. (self._error / self._n_sample))
  end
end

function LossFeedback:add(batch, output, report)
  local targets = self._target_module:forward(batch:targets():input())
  local current_error = self._criterion:forward(output, targets)
  self._error =  self._error + current_error
  self._n_sample = self._n_sample + #targets
  --TODO gather statistics on backward outputGradients?
end

function LossFeedback:report()
  return {
    [self:name()] = self._error / self._n_sample,
    n_sample = self._n_sample
  }
end
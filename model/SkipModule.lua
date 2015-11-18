--[[
--This module only sends the input through its module every skip-th input
 ]]

local _ = require("moses")

local SkipModule, parent = torch.class('dplm.SkipModule','nn.AbstractRecurrent')

-- zeroOutput is used for the first (skip-1) outputs
function SkipModule:__init(skip, module, zeroOutput, dim)
  parent.__init(self)
  -- we can decorate the module with a Recursor to make it AbstractRecurrent
  self.module = (not torch.isTypeOf(module, 'nn.AbstractRecurrent')) and nn.Recursor(module) or module
  self.modules[1] = module
  self.skip = skip
  self.zeroOutput = zeroOutput
  self.zeroGradInput = torch.Tensor()
  self._zeroOutput = zeroOutput
  self.dim = dim or 1
  self.copyGradOutputs = false
  self.copyInputs = false
  self:forget()
end

-- ajust _zeroOutput to current batchSize (no allocation of memory due to use of :expand())
function SkipModule:matchBatchSize(input)
  while type(input) == "table" do input = input[1] end
  if input:nDimension() > self.dim then --has batches
    local out = self._zeroOutput
    local is_table = type(out) == "table"
    if is_table then out = out[1] end
    local batchSize = input:size(1)
    if out:nDimension() == self.dim or out:size(1) ~= batchSize then
      if is_table then
        self._zeroOutput = _.map(self.zeroOutput, function(_,t)
          local sizes = t:size():totable()
          return t:view(1,unpack(sizes)):expand(batchSize,unpack(sizes))
          end)
      else
        local sizes = self.zeroOutput:size():totable()
        self._zeroOutput = self.zeroOutput:view(1,unpack(sizes)):expand(batchSize,unpack(sizes))
      end
    end
  else
    self._zeroOutput = self.zeroOutput
  end
end

local function recursive_reusable_zero(reuse, to_match)
  local function max_size(input)
    if type(input) == "table" then
      return _.max(input, function(t) return max_size(t) end)
    else
      return input:nElement()
    end
  end
  local max = max_size(to_match)
  if reuse:nElement() < max then reuse:resize(max):zero() end
  -- reuse zeroGradInput with views and sub
  local function recursive_view(input)
    if type(input) == "table" then
      return _.map(input, function(_,t) return recursive_view(t) end)
    else
      return reuse:sub(1,input:nElement()):viewAs(input)
    end
  end
  return recursive_view(to_match)
end

function SkipModule:_zeroGradInput(input)
  return recursive_reusable_zero(self.zeroGradInput, input)
end

function SkipModule:forget()
  self.inputs = {}
  self.outputs = {}
  self.output = self._zeroOutput
  self.step = 1
  nn.Module.forget(self)
end

function SkipModule:updateOutput(input)
  if self.train ~= false then
    local input_ = self.inputs[self.step]
    self.inputs[self.step] = self.copyInputs
        and nn.rnn.recursiveCopy(input_, input)
        or nn.rnn.recursiveSet(input_, input)
  end
  if self.step % self.skip == 0 then
    self.output = self.module:updateOutput(input)
  else
    self:matchBatchSize(input)
    self.output = self._zeroOutput
  end
  self.updateGradInputStep = nil
  self.outputs[self.step] = self.output
  self.step = self.step + 1
  return self.output
end

function SkipModule:backwardThroughTime(timeStep, timeRho)
  timeStep = timeStep or self.step
  local gradInput = self:updateGradInputThroughTime(timeStep, timeRho)
  self:accGradParametersThroughTime(timeStep, timeRho)
  return gradInput
end

function SkipModule:updateGradInputThroughTime(timeStep, rho)
  assert(self.step > 1, "expecting at least one updateOutput")
  self.gradInputs = {}
  timeStep = timeStep or self.step
  local rho = math.min(rho or self.rho, timeStep-1)
  local stop = timeStep - rho
  local gradInput

  for step=timeStep-1,math.max(stop,1),-1 do
    if step % self.skip == 0 then
      --local input = _.slice(self.inputs,self._step-skip+1, self._step)
      gradInput = self.module:updateGradInput(self.inputs[step], self.gradOutputs[step])
      table.insert(self.gradInputs, 1, gradInput)
    else
      gradInput = self:_zeroGradInput(self.inputs[1])
      table.insert(self.gradInputs, 1, gradInput)
    end
  end

  return gradInput
end

function SkipModule:accGradParametersThroughTime(timeStep, rho)
  timeStep = timeStep or self.step
  local rho = math.min(rho or self.rho, timeStep-1)
  local stop = timeStep - rho
  for step=timeStep-1,math.max(stop,1),-1 do
    -- backward propagate through this step
    if step % self.skip == 0 then
      --local input = _.slice(self.inputs,self._step-skip+1, self._step)
      self.module:accGradParameters(self.inputs[step], self.gradOutputs[step], self.scales[step])
    end
  end

  self.gradParametersAccumulated = true
end

function SkipModule:accUpdateGradParametersThroughTime(lr, timeStep, rho)
  timeStep = timeStep or self.step
  local rho = math.min(rho or self.rho, timeStep-1)
  local stop = timeStep - rho
  for step=timeStep-1,math.max(stop,1),-1 do
    -- backward propagate through this step
    if step % self.skip == 0 then
      --local input = _.slice(self.inputs,self._step-skip+1, self._step)
      self.module:accUpdateGradParameters(self.inputs[step], self.gradOutputs[step], lr*self.scales[step])
    end
  end
end

function SkipModule:backwardOnline(online)
  assert(online ~= false, "SkipModule only supports online backwards")
  parent.backwardOnline(self)
  nn.Module.backwardOnline(self,online)
end

function SkipModule:maxBPTTstep(rho)
  self.rho = rho
  nn.Module.maxBPTTstep(self, math.floor(rho/self.skip))
end




-- skips input but propagates outputs at every timestep
local SkipInputModule, parent = torch.class('dplm.SkipInputModule','dplm.SkipModule')

function SkipInputModule:updateOutput(input)
  --self.inputs[self._step] = input
  if self.train ~= false then
    local input_ = self.inputs[self.step]
    self.inputs[self.step] = self.copyInputs
        and nn.rnn.recursiveCopy(input_, input)
        or nn.rnn.recursiveSet(input_, input)
  end
  if self.step == 1 then
    self:matchBatchSize(input)
    self.output = self._zeroOutput
  end
  if self.step % self.skip == 0 then
    --local input = _.slice(self.inputs,self._step-skip+1, self._step)
    self.output = self.module:updateOutput(input)
    if self.train ~= false then
      self._gradOutputs[self.step] = rnn.recursiveResizeAs(self._gradOutputs[self.step], self.output)
      rnn.recursiveFill(self._gradOutputs[self.step], 0)
    end
  end
  self.updateGradInputStep = nil
  self.outputs[self.step] = self.output
  self.step = self.step + 1
  return self.output
end

function SkipInputModule:updateGradInputThroughTime(timeStep, rho)
  assert(self.step > 1, "expecting at least one updateOutput")
  self.gradInputs = {}
  timeStep = timeStep or self.step
  local rho = math.min(rho or self.rho, timeStep-1)
  local stop = timeStep - rho
  local gradInput

  for step=timeStep-1,math.max(stop,1),-1 do
    if step % self.skip == 0 then
      --local input = _.slice(self.inputs,self._step-skip+1, self._step)
      gradInput = self.module:updateGradInput(self.inputs[step], self._gradOutputs[step])
      table.insert(self.gradInputs, 1, gradInput)
    else
      local latestSkipStep = math.floor(step/self.skip) * self.skip
      if latestSkipStep > 0 then
        rnn.recursiveAdd(self._gradOutputs[latestSkipStep], self.gradOutputs[step])
      end
      gradInput = self:_zeroGradInput(self.inputs[1])
      table.insert(self.gradInputs, 1, gradInput)
    end
  end

  return gradInput
end

function SkipModule:accGradParametersThroughTime(timeStep, rho)
  timeStep = timeStep or self.step
  local rho = math.min(rho or self.rho, timeStep-1)
  local stop = timeStep - rho
  for step=timeStep-1,math.max(stop,1),-1 do
    -- backward propagate through this step
    if step % self.skip == 0 then
      --local input = _.slice(self.inputs,self._step-skip+1, self._step)
      self.module:accGradParameters(self.inputs[step], self._gradOutputs[step], self.scales[step])
    end
  end

  self.gradParametersAccumulated = true
end

function SkipModule:accUpdateGradParametersThroughTime(lr, timeStep, rho)
  timeStep = timeStep or self.step
  local rho = math.min(rho or self.rho, timeStep-1)
  local stop = timeStep - rho
  for step=timeStep-1,math.max(stop,1),-1 do
    -- backward propagate through this step
    if step % self.skip == 0 then
      --local input = _.slice(self.inputs,self._step-skip+1, self._step)
      self.module:accUpdateGradParameters(self.inputs[step], self._gradOutputs[step], lr*self.scales[step])
    end
  end
end



-- accumulates input over skip steps into a table and propagates outputs at every timestep
local SkipAccumulateInputModule, parent = torch.class('dplm.SkipAccumulateInputModule','dplm.SkipInputModule')

function SkipAccumulateInputModule:forget()
  self.last_gradInput = {}
  parent.forget(self)
end

function SkipAccumulateInputModule:updateOutput(input)
  if self.train ~= false then
    local input_ = self.inputs[self.step]
    self.inputs[self.step] = self.copyInputs
        and nn.rnn.recursiveCopy(input_, input)
        or nn.rnn.recursiveSet(input_, input)
  end
  if self.step == 1 then
    self:matchBatchSize(input)
    self.output = self._zeroOutput
  end
  if self.step % self.skip == 0 then
    local input = _.slice(self.inputs,self.step-self.skip+1, self.step)
    self.output = self.module:updateOutput(input)
    if self.train ~= false then
      self._gradOutputs[self.step] = rnn.recursiveResizeAs(self._gradOutputs[self.step], self.output)
      rnn.recursiveFill(self._gradOutputs[self.step], 0)
    end
  end

  self.outputs[self.step] = self.output
  self.step = self.step + 1
  -- fix due to a bug in AbstractRecurrent
  self.updateGradInputStep = nil
  return self.output
end

function SkipAccumulateInputModule:updateGradInputThroughTime(timeStep, rho)
  assert(self.step > 1, "expecting at least one updateOutput")
  self.gradInputs = {}
  timeStep = timeStep or self.step
  local rho = math.min(rho or self.rho, timeStep-1)
  local stop = timeStep - rho
  local gradInput

  for step=timeStep-1,math.max(stop,1),-1 do
    local latestSkipStep = math.floor(step/self.skip) * self.skip
    if latestSkipStep > 0 then
      rnn.recursiveAdd(self._gradOutputs[latestSkipStep], self.gradOutputs[step])
    end

    if step % self.skip == 0 then
      local input = _.slice(self.inputs,self.step-self.skip+1, self.step)
      self.last_gradInput = self.module:updateGradInput(input, self._gradOutputs[step])
    end
    gradInput = self.last_gradInput[#self.last_gradInput]
    if not gradInput then
      gradInput = self:_zeroGradInput(self.inputs[1])
    end
    table.insert(self.gradInputs, 1, gradInput)
    self.last_gradInput[#self.last_gradInput] = nil
  end

  return gradInput
end

local SwitchCriterion, parent = torch.class("nn.SwitchCriterion", "nn.Criterion")

function SwitchCriterion:__init(criterion, negate)
  self.criterion = criterion
  self.negate = negate
end

function SwitchCriterion:updateOutput(input, target)
  self.output = self.criterion:forward(input, target)
  if self.negate then self.output = -self.output end
  return self.output
end

function SwitchCriterion:updateGradInput(input, target)
  self.gradInput = self.criterion:backward(input, target)
  if self.negate then self.gradInput = -self.gradInput end
  return self.gradInput
end


local MaskedNLLCriterion, ClassNLLCriterion = torch.class("nn.MaskedNLLCriterion", "nn.ClassNLLCriterion")

function MaskedNLLCriterion:updateOutput(input, target)
  local t = target[1]
  local m = target[2]
  if m:sum() == 0 then
    return 0
  else
    self._maskedInput = self._maskedInput or input:clone()
    self._maskedInput:resizeAs(input):copy(input)
    -- mask 0 means faking
    if input:nDimension() == 2 then
      --batch
      self._maskedInput:cmul(m:view(-1,1):expandAs(input))
    else
      self._maskedInput:cmul(m:expandAs(input))
    end
    return ClassNLLCriterion.updateOutput(self,self._maskedInput,t)
  end
end

function MaskedNLLCriterion:updateGradInput(input, target)
  local t = target[1]
  local m = target[2]
  if m:sum() == 0 then
    self.gradInput = self.gradInput or input:clone()
    self.gradInput:resizeAs(input):zero()
  else
    ClassNLLCriterion.updateGradInput(self,self._maskedInput,t)
    if input:nDimension() == 2 then
      --batch
      self.gradInput:cmul(m:view(-1,1):expandAs(input))
    else
      self.gradInput:cmul(m:expandAs(input))
    end
  end
  return self.gradInput
end

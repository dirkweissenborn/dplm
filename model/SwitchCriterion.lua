
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

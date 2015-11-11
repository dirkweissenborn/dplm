--- AttentionLSTM ---

local AttentionLSTM, parent = torch.class("nn.AttentionLSTM","nn.FastLSTM")

--input is table of tensors: batch x inputSize
function AttentionLSTM:buildModel()
  -- input : {input, prevOutput, prevCell}
  -- output : {output, cell}

  local input = nn.Identity()()
  local prev_h = nn.Identity()()
  local prev_c = nn.Identity()()

  --local narrowTable = nn.NarrowTable(2,10000)(input)
  local attention = controlledAttention(self.inputSize, self.outputSize)({input, prev_h})

  -- join current input with attention
  --local joined_input = nn.JoinTable(2)({input,attention})

  -- Calculate all four gates in one go : input, hidden, forget, output
  -- evaluate the input sums at once for efficiency
  local i2h = nn.Linear(self.inputSize, 4 * self.outputSize)(attention)
  local h2h = nn.LinearNoBias(self.outputSize, 4 * self.outputSize)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, self.outputSize)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
  })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return nn.gModule({input, prev_h, prev_c}, {next_h, next_c})
end


function AttentionLSTM:updateOutput(input)
  local prevOutput, prevCell
  if self.step == 1 then
    prevOutput = self.userPrevOutput or self.zeroTensor
    prevCell = self.userPrevCell or self.zeroTensor
    ---< Only change to parent method
    if input[1]:dim() == 2 then
      self.zeroTensor:resize(input[1]:size(1), self.outputSize):zero()
    else
      self.zeroTensor:resize(self.outputSize):zero()
    --- Only change to parent method >
    end
  else
    -- previous output and cell of this module
    prevOutput = self.output
    prevCell = self.cell
  end

  -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
  local output, cell
  if self.train ~= false then
    self:recycle()
    local recurrentModule = self:getStepModule(self.step)
    -- the actual forward propagation
    output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
  else
    output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
  end

  if self.train ~= false then
    self.inputs[self.step] = {}
    nn.rnn.recursiveSet(self.inputs[self.step], input)
  end

  self.outputs[self.step] = output
  self.cells[self.step] = cell

  self.output = output
  self.cell = cell

  self.step = self.step + 1
  self.gradPrevOutput = nil
  self.updateGradInputStep = nil
  self.accGradParametersStep = nil
  self.gradParametersAccumulated = false
  -- note that we don't return the cell, just the output
  return self.output
end
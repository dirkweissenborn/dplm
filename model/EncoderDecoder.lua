require 'nngraph'

local LSTM2LSTM, parent = torch.class("nn.LSTM2LSTM","nn.Container")

--this should be used as wrapper for Sequencer of outLSTM
-- input should be { 1=output of last step of inLSTM, 2=input to outLSTM }
function LSTM2LSTM:__init(inLSTM,outLSTM)
  parent.__init(self)
  self.inLSTM = inLSTM
  self.outLSTM = outLSTM
  self.module = nn.Sequencer(outLSTM)
  self.modules[1] = self.module
end

function LSTM2LSTM:updateOutput(input)
  if self.inLSTM.step > 1 then
    self.outLSTM.userPrevOutput = self.inLSTM.outputs[self.inLSTM.step-1]
    self.outLSTM.userPrevCell = self.inLSTM.cells[self.inLSTM.step-1]
  else
    self.outLSTM.userPrevOutput, self.outLSTM.userPrevCell = nil, nil
  end
  return self.module:updateOutput(input[1])
end

function LSTM2LSTM:updateGradInput(input, gradOutput)
  local grad_input = self.module:updateGradInput(input[1], gradOutput)
  self.inLSTM.userNextGradCell = self.outLSTM.userGradPrevCell
  return {grad_input, self.outLSTM.userGradPrevOutput}
end

function LSTM2LSTM:accGradParameters(input, gradOutput, scale)
  return self.module:accGradParameters(input[1], gradOutput, scale)
end

-------------Adapted FastLSTM to provide also gradient on prevCell -----

local AFastLSTM, parent = torch.class("nn.AFastLSTM","nn.FastLSTM")

function AFastLSTM:backwardThroughTime()
  assert(self.step > 1, "expecting at least one updateOutput")
  self.gradInputs = {} -- used by Sequencer, Repeater
  local rho = math.min(self.rho, self.step-1)
  local stop = self.step - rho
  if self.fastBackward then
    local gradPrevOutput, gradInput, gradCell
    for step=self.step-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)

      -- backward propagate through this step
      local gradOutput = self.gradOutputs[step]
      if gradPrevOutput then
        self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], gradPrevOutput)
        nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
        gradOutput = self._gradOutputs[step]
      end

      local scale = self.scales[step]
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
      local inputTable = {self.inputs[step], output, cell}
      gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
      local gradInputTable = recurrentModule:backward(inputTable, {gradOutput, gradCell}, scale)
      gradInput, gradPrevOutput, gradCell = unpack(gradInputTable)
      if step > math.max(stop,1) then
        self.gradCells[step-1] = gradCell
      end
      table.insert(self.gradInputs, 1, gradInput)
    end
    if self.userPrevOutput then self.userGradPrevOutput = gradPrevOutput end
    -------------- THIS is the only change -----------------------
    if self.userPrevCell then self.userGradPrevCell = gradCell end
    -------------- THIS is the only change -----------------------
    return gradInput
  else
    local gradInput = self:updateGradInputThroughTime()
    self:accGradParametersThroughTime()
    return gradInput
  end
end

function AFastLSTM:updateGradInputThroughTime()
  assert(self.step > 1, "expecting at least one updateOutput")
  self.gradInputs = {}
  local gradInput, gradPrevOutput, gradCell
  local rho = math.min(self.rho, self.step-1)
  local stop = self.step - rho
  for step=self.step-1,math.max(stop,1),-1 do
    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- backward propagate through this step
    local gradOutput = self.gradOutputs[step]
    if gradPrevOutput then
      self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], gradPrevOutput)
      nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
      gradOutput = self._gradOutputs[step]
    end

    local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
    local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
    local inputTable = {self.inputs[step], output, cell}
    gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
    local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})
    gradInput, gradPrevOutput, gradCell = unpack(gradInputTable)
    if step > math.max(stop,1) then
      self.gradCells[step-1] = gradCell
    end
    table.insert(self.gradInputs, 1, gradInput)
  end
  if self.userPrevOutput then self.userGradPrevOutput = gradPrevOutput end
  -------------- THIS is the only change -----------------------
  if self.userPrevCell then self.userGradPrevCell = gradCell end
  -------------- THIS is the only change -----------------------

  return gradInput
end


-------------Encoder Decoder as gModule ------------------

local EncoderDecoder, parent = torch.class('dplm.EncoderDecoder','nn.Container')

function EncoderDecoder:__init(hiddenSize,vocabularySize,dropout)
  parent.__init(self)
  ------ Encoder -------
  local input  = nn.Identity()()
  local lookup = nn.LookupTable(vocabularySize, hiddenSize[1])(input)
  if dropout then
    lookup = nn.Dropout(dropout)(lookup)
  end
  local split  = nn.SplitTable(1,2)(lookup)

  local encLSTM = {}
  local encLSTMOuts = {}

  local inputSize = hiddenSize[1]
  local lastInput = split
  for i,hiddenSize in ipairs(hiddenSize) do
    -- recurrent layer
    local lstm = nn.AFastLSTM(inputSize, hiddenSize)
    lastInput = nn.Sequencer(lstm)(lastInput)
    table.insert(encLSTM,lstm)
    table.insert(encLSTMOuts,nn.SelectTable(-1)(lastInput))
    inputSize = hiddenSize
  end

  self.encoder = nn.gModule({input}, encLSTMOuts)

  ------ Decoder -------
  local input_d  = nn.Identity()()
  local enc_outs = nn.Identity()()

  local lookup_d = nn.LookupTable(vocabularySize, hiddenSize[1])(input_d)
  if dropout then
    lookup_d = nn.Dropout(dropout)(lookup_d)
  end
  local split_d  = nn.SplitTable(1,2)(lookup_d)
  lastInput = split_d
  local inputSize = hiddenSize[1]
  for i,hiddenSize in ipairs(hiddenSize) do
    local decLSTM = nn.AFastLSTM(inputSize, hiddenSize)
    local last_out = nn.SelectTable(i)(enc_outs) --encoder output
    lastInput = nn.LSTM2LSTM(encLSTM[i], decLSTM)({lastInput,last_out})
    inputSize = hiddenSize
  end

  local softmax = nn.Sequential()
  if dropout then
    softmax:add(nn.Dropout(dropout)(lookup_d))
  end
  softmax:add(nn.Linear(inputSize, vocabularySize))
  softmax:add(nn.LogSoftMax())

  local out = nn.Sequencer(softmax)(lastInput)
  self.decoder = nn.gModule({input_d,enc_outs}, {out})

  ------- Encoder-Decoder ----------
  local in_e, in_d = nn.Identity()(), nn.Identity()()
  local out_e = self.encoder(in_e)
  local out_d = self.decoder({in_d, out_e})

  self.module = nn.gModule({in_e, in_d}, {out_d})
  self.module:remember('both')

  self.modules[1] = self.module
  -- we create a zero-dummy encoder output in case we just want to use the decoder
  self.dummy = {}
  for i=1, #hiddenSize do self.dummy[i] = torch.Tensor() end
end

function EncoderDecoder:updateOutput(input)
  if type(input) == "table" then
    self.decoder:forget()
    return self.module:updateOutput(input)
  else
    self.encoder:forget()
    -- use dummy output of encoder
    return self.decoder:updateOutput({input,self.dummy})
  end
end

function EncoderDecoder:updateGradInput(input, gradOutput)
  if type(input) == "table" then
    return self.module:updateGradInput(input, gradOutput)
  else
    return self.decoder:updateGradInput({input,self.dummy}, gradOutput)
  end
end

function EncoderDecoder:accGradParameters(input, gradOutput, scale)
  if type(input) == "table" then
    return self.module:accGradParameters(input, gradOutput, scale)
  else
    return self.decoder:accGradParameters({input,self.dummy}, gradOutput, scale)
  end
end
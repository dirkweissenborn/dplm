-------------Attentional Encoder as Module ------------------

local SkipStackedLSTM, parent = torch.class('dplm.SkipStackedLSTM','nn.Container')


function SkipStackedLSTM:__init(hiddenSize,vocabularySize,dropout,skip)
  parent.__init(self)

  local input  = nn.Identity()()
  local decodingMask = nn.Identity()() -- batch x seq: 1 (predicting) or 0 (reading/encoding)
  local lookup = nn.LookupTable(vocabularySize, hiddenSize[1]-1)(input) -- batch x seq x (dim-1 [one  missing is flag])
  if dropout then
    lookup = nn.Dropout(dropout)(lookup)
  end
  local decodingMaskView = nn.Replicate(1,2,1)(decodingMask) -- make it 3D
  local inputJoin = nn.JoinTable(2,2)({decodingMaskView,lookup})

  local lastInput  = nn.SplitTable(1,2)(inputJoin)
  local enc_outs = {}
  local inputSize = hiddenSize[1]
  for i,hiddenSize in ipairs(hiddenSize) do
    -- recurrent layer
    local lstm
    if i == 1 then
      lstm = nn.FastLSTM(inputSize, hiddenSize)
      lastInput = nn.Sequencer(lstm)(lastInput)
    else
      lstm = nn.AttentionLSTM(inputSize, hiddenSize)
      local segmentedInput = nn.SegmentTable(skip,true)(lastInput)
      lastInput = nn.Sequencer(lstm)(segmentedInput)
    end
    table.insert(enc_outs,lastInput)
    inputSize = hiddenSize
  end
  self.module =  nn.gModule({input,decodingMask}, enc_outs)

  self.module:remember('both')
  self.modules[1] = self.module
  self._hiddenSize = hiddenSize
end

function SkipStackedLSTM:create_decoder(hiddenSize,vocabularySize,dropout,skip)
  local dec_input = nn.Identity()()
  local enc_outs = {dec_input:split(#hiddenSize)}
  local inputSize = hiddenSize[#hiddenSize]
  local lastInput = enc_outs[#hiddenSize]
  for i=#hiddenSize-1,1,-1 do
    local hiddenSize = hiddenSize[i]
    local zip = nn.ZipWithSkipTable(skip)({enc_outs[i],lastInput})
    local join = nn.Sequencer(nn.JoinTable(2))(zip)
    lastInput = nn.Sequencer(nn.FastLSTM(inputSize+hiddenSize, hiddenSize))(join)
    inputSize = hiddenSize
  end

  local softmax = nn.Sequential()
  if dropout then
    softmax:add(nn.Dropout(dropout))
  end
  softmax:add(nn.Linear(inputSize, vocabularySize))
  softmax:add(nn.LogSoftMax())

  local out = nn.Sequencer(softmax)(lastInput)
  return nn.gModule({dec_input}, {out})
end

function SkipStackedLSTM:updateOutput(input)
  if type(input) == "table" then
    return self.module:updateOutput(input)
  else
    self.dummyMask = self.dummyMask or input:clone()
    self.dummyMask:resizeAs(input):fill(1)
    return self.module:updateOutput({input,self.dummyMask})
  end
end

function SkipStackedLSTM:updateGradInput(input, gradOutput)
  if type(input) == "table" then
    return self.module:updateGradInput(input, gradOutput)
  else
    return self.module:updateGradInput({input,self.dummyMask}, gradOutput)[1]
  end
end

function SkipStackedLSTM:accGradParameters(input, gradOutput, scale)
  if type(input) == "table" then
    return self.module:accGradParameters(input, gradOutput, scale)
  else
    return self.module:accGradParameters({input,self.dummyMask}, gradOutput, scale)
  end
end

-------------Attentional Encoder->Decoder as Module ------------------

local SkipStackedLSTMDecoder, parent =
  torch.class('dplm.SkipStackedLSTMDecoder','dplm.SkipStackedLSTM')


function SkipStackedLSTMDecoder:__init(hiddenSize,vocabularySize,dropout,skip)
  parent.__init(self,hiddenSize,vocabularySize,dropout,skip)

  self._encoder = self.module
  self._decoder = self:create_decoder(hiddenSize,vocabularySize,dropout,skip)

  -- real decoder is composed of encoder and decoder
  local enc_input = nn.Identity()()
  local decodingMask = nn.Identity()()
  local e = self._encoder({enc_input,decodingMask})
  local d = self._decoder(e)
  self.module = nn.gModule({enc_input, decodingMask},{d})

  self.module:remember('both')
  self.modules[1] = self.module
end

function SkipStackedLSTMDecoder:create_decoder(hiddenSize,vocabularySize,dropout,skip)
  local dec_input = nn.Identity()()
  local enc_outs = {dec_input:split(#hiddenSize)}
  local inputSize = hiddenSize[#hiddenSize]
  local lastInput = enc_outs[#hiddenSize]
  for i=#hiddenSize-1,1,-1 do
    local hiddenSize = hiddenSize[i]
    local zip = nn.ZipWithSkipTable(skip)({enc_outs[i],lastInput})
    local join = nn.Sequencer(nn.JoinTable(2))(zip)
    lastInput = nn.Sequencer(nn.FastLSTM(inputSize+hiddenSize, hiddenSize))(join)
    inputSize = hiddenSize
  end

  local softmax = nn.Sequential()
  if dropout then
    softmax:add(nn.Dropout(dropout))
  end
  softmax:add(nn.Linear(inputSize, vocabularySize))
  softmax:add(nn.LogSoftMax())

  local out = nn.Sequencer(softmax)(lastInput)
  return nn.gModule({dec_input}, {out})
end

function SkipStackedLSTMDecoder:encoder()
  -- we clone the encoder, because the original encoder is part of the decoder
  if not self._sharedEncoder then
    self._sharedEncoder = self._encoder:sharedClone()
  end
  return self._sharedEncoder
end

-------------Attention SkipEncoder->SkipDecoder as Module ------------------
--[[
-- Attention is based on the last layer of the encoder (which is used for encoding, but also during decoding)
--]]

local AttentionSkipStackedLSTMEncDec, parent =
  torch.class('dplm.AttentionSkipStackedLSTMEncDec','dplm.SkipStackedLSTM')


function AttentionSkipStackedLSTMEncDec:__init(hiddenSize,vocabularySize,dropout,skip)
  parent.__init(self,hiddenSize,vocabularySize,dropout,skip)

  self._encoder = self.module
  self._decoder = self:create_decoder(hiddenSize,vocabularySize,dropout,skip)

  -- real decoder is composed of encoder and decoder
  local enc_input = nn.Identity()()
  local attention_input = nn.Identity()()
  local decodingMask = nn.Identity()()
  local e = self._encoder({enc_input,decodingMask})
  local d = self._decoder({e,attention_input})
  self.module = nn.gModule({enc_input, attention_input, decodingMask},{d})

  self.module:remember('both')
  self.modules[1] = self.module

  -- self._encoder:sharedClone() -> results in a bug
  local encoder = dplm.SkipStackedLSTM(hiddenSize,vocabularySize,dropout,skip)
  encoder:remember('both')
  local input  = nn.Identity()()
  local mask = nn.Identity()()
  local select = nn.SelectTable(#self._hiddenSize)(encoder({input,mask}))
  self._sharedEncoder = nn.gModule({input,mask},{select})
  self:tie_parameters()
  self.modules[2] = self._sharedEncoder
end

function AttentionSkipStackedLSTMEncDec:create_decoder(hiddenSize,vocabularySize,dropout, skip)
  local dec_input = nn.Identity()() -- table of tensors
  local to_attend = nn.Identity()() -- table of tensors
  local enc_outs = {dec_input:split(#hiddenSize) }
  local inputSize = hiddenSize[#hiddenSize]
  local lastInput = enc_outs[#hiddenSize]

  --for each element in lastInput calculate an attention
  local attention = controlledMultiAttention(inputSize,inputSize,math.ceil(inputSize/10))({to_attend, lastInput})
  lastInput = nn.Sequencer(nn.JoinTable(2,2))(nn.ZipTable()({lastInput, attention}))
  inputSize = 2*inputSize
  for i=#hiddenSize-1,1,-1 do
    local hiddenSize = hiddenSize[i]
    local zip = nn.ZipWithSkipTable(skip)({enc_outs[i],lastInput})
    local join = nn.Sequencer(nn.JoinTable(2))(zip)
    lastInput = nn.Sequencer(nn.FastLSTM(inputSize+hiddenSize, hiddenSize))(join)
    inputSize = hiddenSize
  end

  local softmax = nn.Sequential()
  if dropout then
    softmax:add(nn.Dropout(dropout))
  end
  softmax:add(nn.Linear(inputSize, vocabularySize))
  softmax:add(nn.LogSoftMax())

  local out = nn.Sequencer(softmax)(lastInput)
  return nn.gModule({dec_input, to_attend}, {out})
end

function AttentionSkipStackedLSTMEncDec:updateOutput(input)
  local enc_input = input[1]
  local dec_input = input[2]
  local encoder = self:encoder()

  self.dummyMask = self.dummyMask or enc_input:clone()
  self.dummyMask:resizeAs(enc_input):fill(0)

  if #input >= 3 then
    local mask = input[3]
    self._to_attend = encoder:updateOutput({enc_input, self.dummyMask})
    return self.module:updateOutput({dec_input, self._to_attend, mask})
  else
    self._to_attend = encoder:updateOutput({enc_input,self.dummyMask})
    self.dummyMask:resizeAs(dec_input):fill(1)
    return self.module:updateOutput({dec_input,self._to_attend,self.dummyMask})
  end
end

function AttentionSkipStackedLSTMEncDec:updateGradInput(input, gradOutput)
  local enc_input = input[1]
  local dec_input = input[2]
  local encoder = self:encoder()

  local mask = self.dummyMask
  if #input >= 3 then mask = input[3] end
  mask:resizeAs(dec_input):fill(1)
  local gi = self.module:updateGradInput({dec_input, self._to_attend, mask}, gradOutput)
  self._to_attend_d = gi[2]
  self.dummyMask:resizeAs(enc_input):fill(0)
  local gi2 = encoder:updateGradInput({enc_input, self.dummyMask}, self._to_attend_d)
  return {gi2[1],gi[1],gi[3]}
end

function AttentionSkipStackedLSTMEncDec:accGradParameters(input, gradOutput, scale)
  local enc_input = input[1]
  local dec_input = input[2]
  local encoder = self:encoder()

  local mask = self.dummyMask
  if #input >= 3 then mask = input[3] end

  mask:resizeAs(dec_input):fill(1)
  self.module:accGradParameters({dec_input, to_attend, mask}, gradOutput, scale)
  self.dummyMask:resizeAs(enc_input):fill(0)
  encoder:accGradParameters({enc_input, self.dummyMask}, self._to_attend_d, scale)
end

function AttentionSkipStackedLSTMEncDec:tie_parameters()
  local params,grads = self._encoder:parameters()
  local s_params,s_grads = self._sharedEncoder:parameters()
  for i,v in pairs(params) do
    s_params[i]:set(v)
    s_grads[i]:set(grads[i])
  end
end

function AttentionSkipStackedLSTMEncDec:encoder()
  -- we clone the encoder, because the original encoder is part of the decoder
  if not self._sharedEncoder then
    local encoder  =  self._encoder:sharedClone()
    encoder:remember('both')
    local input  = nn.Identity()()
    local mask = nn.Identity()()
    local select = nn.SelectTable(#self._hiddenSize)(encoder({input,mask}))
    self._sharedEncoder = nn.gModule({input,mask},{select})
    self.modules[2] = self._sharedEncoder
  end

  return self._sharedEncoder
end
-------------Attentional Encoder as Module ------------------

local SkipStackedLSTM, parent = torch.class('dplm.SkipStackedLSTM','nn.Container')

function SkipStackedLSTM:__init(hiddenSize, vocabularySize, skip, dropout, rho)
  parent.__init(self)
  self._hiddenSize = hiddenSize
  self._vocabularySize = vocabularySize
  self._skip = skip
  self._dropout = dropout
  self.module = self:create_input()
  self._encoder, self._lstms = self:create_encoder(rho)
  self.module:add(self._encoder)
  self.module:remember('both')
  self.modules[1] = self.module
end

--static
function SkipStackedLSTM:create_encoder(rho, i)
  rho = rho or 1
  i = i or 1
  local hiddenSize = self._hiddenSize
  local skip = self._skip
  --actual encoder
  local encoder = nn.Sequential()
  local inputSize = hiddenSize[i-1] or hiddenSize[i]

  -- recurrent layer
  local lstm
  if i == 1 then
    lstm = nn.FastLSTM(inputSize, hiddenSize[i], rho)
  else
    lstm = nn.AttentionLSTM(inputSize, hiddenSize[i], rho)
  end
  encoder:add(lstm)
  local lstms
  if i < #hiddenSize then
    local zeros = torch.zeros(hiddenSize[#hiddenSize])
    local inner
    inner, lstms = self:create_encoder(rho, i+1)
    table.insert(lstms,1,lstm)
    encoder:add(dplm.SkipAccumulateInputModule(skip, inner, zeros))
  else
    lstms = {lstm}
  end

  return encoder, lstms
end

--create input module - returns sequential
function SkipStackedLSTM:create_input()
  --local input_module = nn.Sequential()
  --input_module:add(nn.LookupTable(self._vocabularySize, self._hiddenSize[1]-1))
  --if self._dropout then
  --  input_module:add(nn.Dropout(dropout))
  --end
  --local mask_module = nn.Replicate(1,2,1)
  --local module = nn.Sequential()
  --module:add(nn.ParallelTable():add(input_module):add(mask_module))
  --module:add(nn.JoinTable(2,2))
  local input_module = nn.Sequential()
  input_module:add(nn.LookupTable(self._vocabularySize, self._hiddenSize[1]))
  if self._dropout then
    input_module:add(nn.Dropout(dropout))
  end
  return input_module
end

function SkipStackedLSTM:updateOutput(input)
  --if type(input) == "table" then
  return self.module:updateOutput(input)
  --else
  --  self.dummyMask = self.dummyMask or input:clone()
  --  self.dummyMask:resizeAs(input):fill(1)
  --  return self.module:updateOutput({input,self.dummyMask})
  --end
end

function SkipStackedLSTM:updateGradInput(input, gradOutput)
  --if type(input) == "table" then
  return self.module:updateGradInput(input, gradOutput)
  --else
  --  return self.module:updateGradInput({input,self.dummyMask}, gradOutput)[1]
  --end
end

function SkipStackedLSTM:accGradParameters(input, gradOutput, scale)
  --if type(input) == "table" then
  return self.module:accGradParameters(input, gradOutput, scale)
  --else
   -- return self.module:accGradParameters({input,self.dummyMask}, gradOutput, scale)
  --end
end

------------- Skip Decoder as Module ------------------

local SkipStackedLSTMDecoder, parent =
  torch.class('dplm.SkipStackedLSTMDecoder','dplm.SkipStackedLSTM')


function SkipStackedLSTMDecoder:__init(...)
  parent.__init(self, ...)
  if self._dropout then
    self.module:add(nn.Dropout(dropout))
  end
  self.module:add(nn.Linear(self._hiddenSize[1], self._vocabularySize))
  self.module:add(nn.LogSoftMax())
  self.module:remember('both')
  self.modules[1] = self.module
end

function SkipStackedLSTMDecoder:create_encoder(rho, i)
  rho = rho or 1
  i = i or 1
  local hiddenSize = self._hiddenSize
  local skip = self._skip
  --actual encoder
  local encoder = nn.Sequential()
  local inputSize = hiddenSize[i-1] or hiddenSize[i]

  -- recurrent layer
  local lstm
  if i == 1 then
    lstm = nn.FastLSTM(inputSize, hiddenSize[i], rho)
  else
    lstm = nn.AttentionLSTM(inputSize, hiddenSize[i], rho)
  end
  encoder:add(lstm)
  local lstms
  if i < #hiddenSize then
    local zeros = torch.zeros(hiddenSize[i+1])

    local inner
    inner, lstms = self:create_encoder(rho, i+1)
    table.insert(lstms,1,lstm)
    local concat = nn.ConcatTable()
    concat:add(nn.Identity())
    concat:add(dplm.SkipAccumulateInputModule(skip, inner, zeros))
    encoder:add(concat)
    encoder:add(nn.JoinTable(2))
    local dec_lstm = nn.FastLSTM(hiddenSize[i+1]+hiddenSize[i], hiddenSize[i], rho)
    encoder:add(dec_lstm)
    table.insert(lstms,dec_lstm)
  else
    lstms = {lstm}
  end

  return encoder, lstms
end

------------- Attentional Skip Decoder as Module ------------------

local AttentionSkipStackedLSTMDecoder, parent =
torch.class('dplm.AttentionSkipStackedLSTMDecoder','dplm.SkipStackedLSTM')

function AttentionSkipStackedLSTMDecoder:__init(hiddenSize, vocabularySize, skip, dropout, interactionSize, rho)
  nn.Container.__init(self)
  self._hiddenSize = hiddenSize
  self._vocabularySize = vocabularySize
  self._interactionSize = interactionSize or math.ceil(hiddenSize[#hiddenSize]/10)
  self._skip = skip
  self._dropout = dropout
  self.module = self:create_input()
  -- input is table: {input indices, {to attend,...}}
  self.module = nn.Sequential():add(
    nn.ParallelTable():add(self.module):add(
      nn.ConcatTable():add(
        nn.Identity()
      ):add( --already calculateprojections beforehand, to not calculate it in every timestep
        nn.Sequencer(nn.LinearNoBias(hiddenSize[#hiddenSize], self._interactionSize))
      )
    )
  )
  self._encoder, self._lstms = self:create_encoder(rho)
  self.module:add(self._encoder)
  if self._dropout then
    self.module:add(nn.Dropout(dropout))
  end
  self.module:add(nn.Linear(self._hiddenSize[1], self._vocabularySize))
  self.module:add(nn.LogSoftMax())
  self.module:remember('both')
  self.modules[1] = self.module
end

-- input is table: {input, to_attend}, where input is either tensor or table depending on i and to_attend is table
-- of tensor we should put attention on
function AttentionSkipStackedLSTMDecoder:create_encoder(rho, i)
  rho = rho or 1
  i = i or 1
  local hiddenSize = self._hiddenSize
  local skip = self._skip
  --actual encoder
  local encoder = nn.Sequential()
  if i > 1 then --input correction
    -- after SkipAccumulateInputModule we end up with a repitition of real input and to_attend
    -- e.g., { {input1, to_attend}, {input2, to_attend}, {input3, to_attend}, ...}
    -- we need { {input1, input2, input3}, to_attend}
    local input_correct = nn.Sequencer(nn.SelectTable(1))
    local to_attend_correct = nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(2))
    encoder:add(nn.ConcatTable():add(input_correct):add(to_attend_correct))
  end
  local inputSize = hiddenSize[i-1] or hiddenSize[i]

  -- recurrent layer
  local lstms
  local lstm

  if i < #hiddenSize then
    if i == 1 then
      lstm = nn.FastLSTM(inputSize, hiddenSize[i], rho)
    else
      lstm = nn.AttentionLSTM(inputSize, hiddenSize[i], rho)
    end
    local zeros = torch.zeros(hiddenSize[i+1])
    if i == #hiddenSize-1 then zeros = { zeros, zeros} end --also account for attention
    local inner
    inner, lstms = self:create_encoder(rho, i+1)
    table.insert(lstms,1,lstm)
    --attention is only used in middle layer --> pipe it through this layer with Identity
    encoder:add(nn.ParallelTable():add(lstm):add(nn.Identity()))
    local concat = nn.ConcatTable()
    concat:add(nn.SelectTable(1))
    concat:add(dplm.SkipAccumulateInputModule(skip, inner, zeros))
    encoder:add(concat)
    encoder:add(nn.FlattenTable()) --so we do not have to join attention
    encoder:add(nn.JoinTable(2))
    inputSize = hiddenSize[i+1]+hiddenSize[i]
    if i == #hiddenSize-1 then inputSize = inputSize + hiddenSize[i+1] end --also add attention
    local dec_lstm = nn.FastLSTM(inputSize, hiddenSize[i], rho)
    encoder:add(dec_lstm)
    table.insert(lstms,dec_lstm)
  else
    -- input is lower layer input and to_attend table
    if i == 1 then
      lstm = nn.FastLSTM(inputSize, hiddenSize[i], rho)
    else
      lstm = nn.AttentionLSTM(inputSize, hiddenSize[i], rho)
    end
    encoder:add(nn.ParallelTable():add(lstm):add(nn.Identity())) -- apply lstm before attention
    encoder:add(
      nn.ConcatTable():add(
        nn.SelectTable(1) -- output of lstm
      ):add(
        controlledAttentionAlreadyProjected(hiddenSize[i], hiddenSize[i], self._interactionSize)  -- attention
      )
    )

    lstms = {lstm}
  end

  return encoder, lstms
end


-------------Attention SkipEncoder->SkipDecoder as Module ------------------
--[[
-- Attention is based on the last layer of the encoder (which is used for encoding, but also during decoding)
--]]

local AttentionSkipStackedLSTMEncDec, parent =
  torch.class('dplm.AttentionSkipStackedLSTMEncDec','dplm.SkipStackedLSTMDecoder')


function AttentionSkipStackedLSTMEncDec:__init(hiddenSize, vocabularySize, attInteractionSize, skip, dropout, tie)
  parent.__init(self,hiddenSize,vocabularySize,skip,dropout)
  self._interaction_size = attInteractionSize or hiddenSize[#hiddenSize]/10
  self._decoder = self.module
  --create encoder of same shape
  self._encoder = dplm.SkipStackedLSTM(hiddenSize,vocabularySize,dropout,skip)

  -- real decoder is composed of encoder and decoder
  local enc_input = nn.Identity()()
  local decodingMask = nn.Identity()()
  local e = self._encoder({enc_input,decodingMask})
  local attention_input = nn.Identity()()
  local proj_attention = nn.Sequencer(nn.LinearNoBias(hiddenSize[#hiddenSize], self._interaction_size))(attention_input)
  local d = self._decoder({e,proj_attention})
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
  self.modules[2] = self._sharedEncoder
  if tie then self:tie_parameters() end
end

function AttentionSkipStackedLSTMEncDec:create_encoder(rho, i)
  rho = rho or 1
  i = i or 1
  local hiddenSize = self._hiddenSize
  local skip = self._skip
  --actual encoder
  local encoder = nn.Sequential()
  local inputSize = hiddenSize[i-1] or hiddenSize[i]

  -- recurrent layer
  local lstms
  local lstm

  if i < #hiddenSize then
    if i == 1 then
      lstm = nn.FastLSTM(inputSize, hiddenSize[i], rho)
    else
      lstm = nn.AttentionLSTM(inputSize, hiddenSize[i], rho)
    end
    --attention is only used in middle layer
    encoder:add(lstm)
    local zeros = {torch.zeros(hiddenSize[i+1])}
    local inner
    inner, lstms = self:create_encoder(rho, i+1)
    table.insert(lstms,1,lstm)

    encoder:add(nn.ParallelTable():add(lstm):add(nn.Identity()))

    local concat = nn.ConcatTable()
    concat:add(nn.SelectTable(1))
    concat:add(dplm.SkipAccumulateInputModule(skip, inner, zeros))
    encoder:add(concat)
    encoder:add(nn.FlattenTable()) --so we do not have to join attention
    encoder:add(nn.JoinTable(2))
    inputSize = hiddenSize[i+1]+hiddenSize[i]
    if i == #hiddenSize-1 then inputSize = inputSize + hiddenSize[i+1] end --also add attention
    local dec_lstm = nn.FastLSTM(inputSize, hiddenSize[i], rho)
    encoder:add(dec_lstm)
    table.insert(lstms,dec_lstm)
  else
    -- input is lower layer input and to_attend table
    if i == 1 then
      lstm = nn.FastLSTM(inputSize+hiddenSize[i], hiddenSize[i], rho)
    else
      lstm = nn.AttentionLSTM(inputSize+hiddenSize[i], hiddenSize[i], rho)
    end
    encoder:add(nn.ParallelTable():add(lstm):add(nn.Identity())) -- apply lstm before attention
    encoder:add(
      nn.ConcatTable():add(
        nn.SelectTable(1) -- output of lstm
      ):add(
        controlledAttention(hiddenSize[i],hiddenSize[i],self._interaction_size)  -- attention
      )
    )

    lstms = {lstm}
  end

  return encoder, lstms
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
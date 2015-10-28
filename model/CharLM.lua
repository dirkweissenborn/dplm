require "cunn"

local CharLM = torch.class("dplm.CharLM")

function CharLM:__init(model,vocab)
  self.model = model
  self.vocab = vocab
end

function CharLM:forward(text, forget)
  self.model:evaluate()
  if forget then self.model:forget() end
  local t = torch.zeros(string.len(text))
  for i=1, string.len(text) do
    t[i] = self.vocab[text:sub(i,i)]
  end  
  local out = self.model:forward(t:view(1,-1))
  return out[#out]
end

function CharLM:rev_vocab()
  if not self._rev_vocab then
    self._rev_vocab = {}
    for k,v in pairs(self.vocab) do
      self._rev_vocab[v] = k
    end  
  end
  return self._rev_vocab
end

function CharLM:to_tensor(text, d)
  local i = 0
  d = d or torch.IntTensor(string.len(text))

  for char in text:gmatch '.' do
    i = i + 1
    if char == '\n' then
      d[i] = self:endId()
    else
      d[i] = self.vocab[char] or self:unkId()
    end
  end

  return d
end

function CharLM:endId() return self.vocab["</S>"] end
function CharLM:startId() return self.vocab["<S>"] end
function CharLM:unkId() return self.vocab["<U>"] end

function CharLM:getParameters()
  if not self._params then
    self._params, self._gradParams = self.model:getParameters()
  end
  return self._params, self._gradParams
end

function CharLM:vocab_size()
  self._vocab_size = self._vocab_size or tablex.size(self.vocab)
  return self._vocab_size
end

function CharLM:save(f)
  --make model small -> batch size 1 and no shared clones

  self.model:forget()
  local function remove_clones(module)
    if module.sharedClones then
      module.sharedClones = {module.sharedClones[1]}
    end
    if module.gradInputs then
      module.gradInputs = {module.gradInputs[1]}
    end
    if module.gradOutputs then
      module.gradOutputs = {module.gradOutputs[1]}
      module._gradOutputs = {module._gradOutputs[1]}
    end
    if module.outputs then
      module.outputs = {module.outputs[1]}
    end
    if module.inputs then
      module.inputs = {module.inputs[1]}
    end
    -- LSTM
    if module.cells then
      module.cells = {module.cells[1]}
      module.gradCells = {module.gradCells[1]}
    end
    if module.modules then for _,m in pairs(module.modules) do remove_clones(m) end end
  end
  remove_clones(self.model,t)
  local params = self:getParameters()
  self.model:forward(torch.ones(1,1):typeAs(params))
  self.model:backward(torch.ones(1,1):typeAs(params),{torch.zeros(1,self:vocab_size()):typeAs(params)})
  torch.save(f,self)
end

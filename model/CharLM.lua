require "cunn"

local CharLM = torch.class("dplm.CharLM")

function CharLM:__init(model,vocab)
  self.model = model
  self.vocab = vocab
end

function CharLM:forward(text, forget)
  self.model:evaluate()
  if forget then self.model:forget() end
  local l = string.len(text)
  if forget then l = l + 1 end
  local t = torch.zeros(l):fill(self:startId())
  for i=1, string.len(text) do
    local idx = i
    if forget then idx = idx+1 end
    t[idx] = self.vocab[text:sub(i,i)]
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

function CharLM:clean()
  self.model:forget()
  local function clean(module)
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
    if module.modules then for _,m in pairs(module.modules) do clean(m) end end
  end
  clean(self.model)
  local params = self:getParameters()
  self.model:training()
  self.model:forward(torch.ones(1,1):typeAs(params))
  self.model:backward(torch.ones(1,1):typeAs(params),{torch.zeros(1,self:vocab_size()):typeAs(params)})
  self.model:forget()
  collectgarbage()
end

function CharLM:save(f)
  self:clean()
  torch.save(f,self)
end

function CharLM:beam_search(str, n, l, temp)
  local do_sample = temp
  temp = temp or 1
  --sampling doesn't work well, yet
  local function sample(p)
    if do_sample then
      local sum = p:clone():div(temp):exp():sum()
      local ix = torch.Tensor(math.min(n,p:size(1)))
      local y = torch.Tensor(math.min(n,p:size(1)))
      for i=1,math.min(n,p:size(1)) do
        local u = torch.uniform(0,sum)
        local k = 0
        while u > 0 do
          k = k + 1
          u = u - math.exp(p[k]/temp)
        end
        ix[i] = k
        y[i] = p[k]
      end
      return y, ix
    else
      return torch.sort(p,1,true)
    end
  end

  self:clean()
  local model = self.model
  local p = self:forward(str,true)
  local rev_vocab = {}
  for k,v in pairs(self.vocab) do rev_vocab[v] = k end

  local beam = {}
  local y, ix = sample(p[1])

  local function get_state(m,out,cell)
    out =  out or {}
    cell = cell or {}
    if torch.isTypeOf(m, "nn.AbstractRecurrent") then
      table.insert(out,m.output:clone())
      if m.cell then
        table.insert(cell,m.cell:clone())
      end
    end
    if m.modules then
      for _,m2 in ipairs(m.modules) do
        get_state(m2, out, cell)
      end
    end
    return {out,cell}
  end

  local function set_state(m,out,cell,io,ic)
    io = io or 1
    ic = ic or 1
    out =  out or {}
    cell = cell or {}
    if torch.isTypeOf(m, "nn.AbstractRecurrent") then
      m.output = out[io]
      io = io + 1
      if m.cell then
        m.cell = cell[ic]
        ic = ic + 1
      end
    end
    if m.modules then
      for _,m2 in ipairs(m.modules) do
        io, ic = set_state(m2, out, cell, io, ic)
      end
    end
    return io, ic
  end

  local s = get_state(model)
  for i = 1, math.min(n, p[1]:size(1)) do
    if #beam < n then
      local b = {}
      b.is = {ix[i]}
      b.chars = {rev_vocab[ix[i]]}
      b.p = y[i]
      b.s = s
      table.insert(beam, b)
    end
  end

  for j = 2, l do
    local new_beam = {}
    for i= 1,  #beam do
      local b = beam[i]
      set_state(model, unpack(b.s))
      local p = self:forward(b.chars[#b.chars])
      local y,ix = sample(p[1])
      local s = get_state(model)
      for ii = 1, math.min(n, p[1]:size(1)) do
        local b2 = tablex.deepcopy(b)
        b2.s = s
        table.insert(b2.is, ix[ii])
        table.insert(b2.chars, rev_vocab[ix[ii]])
        b2.p = b.p + y[ii]
        table.insert(new_beam, b2)
      end
    end
    local i = 1
    for k,v in tablex.sortv(new_beam, function(x,y) return x.p > y.p end) do
      if i < n+1 then
        beam[i] = v
        i= i+1
      end
    end
    collectgarbage()
  end
  self.model = model
  -- cleanup
  for i= 1, #beam do beam[i].m = nil end
  collectgarbage()
  return beam
end

function next_word(decoder, str, n, max_l)
  local beam = beam_search(decoder, str, n or 10, max_l or 20)
  local next = table.concat(beam[1].chars)
  local split = stringx.split(next," ")
  return str .. split[1] .. " "
end
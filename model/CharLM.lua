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
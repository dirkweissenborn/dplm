require 'lmdb'

local Lexicon = torch.class("dplm.Lexicon")

function Lexicon:__init(path,silent)
  self.db= lmdb.env{
    Path = path,
    Name = 'Lexicon'
  }
  self.db:open()
  self._silent = silent
  if not silent then
    print(self.db:stat())
  end
end

function Lexicon:get(key)
  local reader = self.db:txn(true)
  local ret = reader:get(key)
  reader:abort()
  return ret
end

function Lexicon:getAll(keys)
  local ret = {}
  local reader = self.db:txn(true)
  for _,k in pairs(keys) do
    local res = reader:get(k)
    ret[k] = res
  end
  reader:abort()
  return ret
end

function Lexicon:putAll(t)
  local writer = self.db:txn()
  for k,v in pairs(t) do
    writer:put(k,v)
  end
  writer:commit()
end

function Lexicon:put(k,v)
  local writer = self.db:txn()
  writer:put(k,v)
  writer:commit()
end

function Lexicon:close() self.db:close() end
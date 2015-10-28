require 'dplm'
require 'pl'
local tds = require 'tds'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:option('-s','\t','Seperator used for indexing. Assuming txt file with key, value per line separated by s.')
cmd:option('-f','','File to index')
cmd:option('-o','lexicon.lmdb','output path')

local opt = cmd:parse(arg)
local l = dplm.Lexicon(opt.o)

print("Loading data into memory...")
local facts = tds.Hash()
local size = 0
for line in io.lines(opt.f) do
  local split = stringx.split(line,opt.s,2)
  if not stringx.startswith(line, '#') and split[1] and split[2] then
    facts[split[1]] = facts[split[1]] or tds.Hash()
    local hs = facts[split[1]]
    hs[#hs] = split[2]
    size = size + 1
    if size % 1e6 == 0 then print(string.format("Loaded %d facts into memory",size)) end
  end
end
print("Writing to database...")
l:putAll(facts)

l:close()
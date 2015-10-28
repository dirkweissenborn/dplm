require 'pl'

--[[
-- { [1]={document="..",annos:{ [1]={c_start=x,c_end=y[,gs=".."],... },...}
]]--

local AIDALoader = torch.class("dplm.AIDALoader")

function AIDALoader:__init(dir)
  local offset = 0
  local annos = self:loadAnnotations(path.join(dir,'AIDA-YAGO2-annotations.tsv'))
  self.train, offset = self:loadDataset(path.join(dir,"eng.train"),annos,offset)
  self.valid, offset = self:loadDataset(path.join(dir,"eng.testa"),annos,offset)
  self.test = self:loadDataset(path.join(dir,"eng.testb"),annos,offset)
end

function AIDALoader:loadDataset(file, annos, offset)
  offset = offset or 0
  local current
  local docs = {}
  local i = 0
  local tag = ""
  local is_NE = false
  local id = offset
  local doc_annos
  local current_anno
  local str_offset = 0
  for l in io.lines(file) do
    local split = stringx.split(l,' ')

    -- add annotation, if has ended here
    if is_NE and (tag ~= split[4] or stringx.startswith(split[4],"B-")) then
      current_anno.c_end = str_offset
      is_NE = false
      tag = ""
    end

    if stringx.startswith(l,"-DOCSTART-") then
      current = { document = "", annos = {} }
      table.insert(docs,current)
      i = 0
      id = id + 1
      doc_annos = annos[id]
      str_offset = 0
    elseif split[1] then
      i = i+1
      -- create new anno
      if doc_annos[i] then
        tag = split[4]
        is_NE = true
        current_anno = {}
        current_anno.gs = doc_annos[i]
        current_anno.c_start = str_offset + 1
        table.insert(current.annos, current_anno)
      end
      current.document = current.document .. split[1] .. " "
      str_offset = str_offset + string.len(split[1]) + 1
    else
      str_offset = str_offset + 1
      current.document = current.document .. "\n"
    end
  end
  return docs, id
end

function AIDALoader:loadAnnotations(file)
  local annotations = {}
  local current
  local pref_len = string.len("http://en.wikipedia.org/wiki/")
  for l in io.lines(file) do
    if not stringx.startswith(l,'#') then
      local split = stringx.split(l,'\t')
      if split[2] then
        if split[2] ~= "--NME--" then
          current[tonumber(split[1])+1] = split[3]:sub(pref_len+1)
        end
      elseif stringx.startswith(l,"-DOCSTART-") then
        current = {}
        table.insert(annotations,current)
      end
    end
  end
  return annotations
end
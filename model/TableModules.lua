
-- partitions a table into a table of tables containing a max number of tensors
-- kind of the opposite of FlattenTable
-- used just before AttentionLSTM
local SegmentTable,parent = torch.class("nn.SegmentTable","nn.Module")

function SegmentTable:__init(partition_size,prepend)
  self.partition_size = partition_size
  self._prepend = prepend
  self._left_over = {}
  self.step = 1
end

function SegmentTable:updateOutput(input)
  assert(self.train ~= false or #input == 1, "During evalutation SegmentTable can only be used 1 step at a time")
  self.output = {}
  for _,v in pairs(input) do
    table.insert(self._left_over, v)
  end

  if self._prepend then
    table.insert(self.output, self._prepend)
  end

  for p in _.partition(self._left_over,self.partition_size) do
    if #p == self.partition_size then table.insert(self.output, p)
    elseif self.train ~= false then
      self._left_over = p
    else
      _.slice(self._left_over,1,#p)
      rnn.recursiveCopy(self._left_over, p)
    end
  end

  self.step = self.step + #input
  return self.output
end

function SegmentTable:forget()
  self._left_over = {}
end


function SegmentTable:updateGradInput(input, gradOutput)
  self.gradInput = _.flatten(gradOutput)
  return self.gradInput
end

-- partitions a table into a table of max tables with min members each
-- kind of the opposite of FlattenTable

local PartitionTable,parent = torch.class("nn.PartitionTable","nn.Module")

function PartitionTable:__init(max_partitions, min_members,first_extra)
  self.max_partitions = max_partitions
  self.min_members = min_members or 1
  self.first_extra = first_extra
end

function PartitionTable:updateOutput(input)
  self.output = {}
  local t = input

  if self.first_extra then
    table.insert(self.output, {input[1]})
    t = _.tail(t,2)
  end
  local num_partitions = math.min(self.max_partitions, #t/self.min_members)
  for p in _.partition(t, num_partitions) do
    table.insert(self.output, p)
  end
  return self.output
end

function PartitionTable:updateGradInput(input, gradOutput)
  self.gradInput = _.flatten(gradOutput)
  return self.gradInput
end

-- SlidingWindowTable

local SlidingWindowTable,parent = torch.class("nn.SlidingWindowTable","nn.Module")

function SlidingWindowTable:__init(size)
  self.size = size
end

function SlidingWindowTable:updateOutput(input)
  self.output = self.output or {}
  for i=1, #input do
    table.insert(self.output, _.slice(input,math.max(i-self.size,1), i))
  end
  return self.output
end

function SlidingWindowTable:updateGradInput(input, gradOutput)
  self.gradInput = _.flatten(gradOutput)
  for i=1, #input do
    table.insert(self.output, _.slice(input,math.max(i-self.size,1), i))
  end
  return self.gradInput
end

-- zip two table where second's size is a fraction of the first size --> reuse entries in second skip times and zip

local ZipWithSkipTable,parent = torch.class("nn.ZipWithSkipTable","nn.Module")

function ZipWithSkipTable:__init(skip)
  self.skip = skip
end

function ZipWithSkipTable:updateOutput(input)
  local t1 = input[1]
  local t2 = input[2]
  local skip = self.skip > 0 and self.skip or math.floor((#t1-1)/(#t2-1))
  self.output = {}
  for i,v in ipairs(t1) do
    local i2 = math.ceil(i/skip)
    table.insert(self.output, {v,t2[i2]})
  end
  return self.output
end

function ZipWithSkipTable:updateGradInput(input, gradOutput)
  local t1 = input[1]
  local t2 = input[2]
  local skip = self.skip > 0 and self.skip or math.floor((#t1-1)/(#t2-1))
  self.gradInput = {}
  self.gradInput[1] =  {}
  self.gradInput[2] = self.gradInput[2] and _.slice(self.gradInput[2],1,#t2) or {}
  for i,v in ipairs(gradOutput) do
    table.insert(self.gradInput[1],v[1])
    local i2 = math.ceil(i/skip)
    self.gradInput[2][i2] = self.gradInput[2][i2] or t2[i2]:clone():zero()
    self.gradInput[2][i2]:resizeAs(input[2][i2])
    self.gradInput[2][i2]:add(v[2])
  end
  return self.gradInput
end


-- Zip Table with Table

local ZipTableWithTable, parent = torch.class("nn.ZipTableWithTable","nn.Module")

function ZipTableWithTable:__init()
end

function ZipTableWithTable:updateOutput(input)
  local tab = input[1]
  local tab2 = input[2]
  self.output = {}
  for _,v in ipairs(tab) do table.insert(self.output, {v, tab2 }) end
  return self.output
end

function ZipTableWithTable:updateGradInput(input, gradOutput)
  local tab2 = input[2]
  self.gradInput = self.gradInput or {}
  self.gradInput[1] = self.gradInput[1] and _.slice(self.gradInput[1],1,#gradOutput) or {}
  self.gradInput[2] = self.gradInput[2] and _.slice(self.gradInput[2],1,#tab2) or rnn.recursiveResizeAs({}, tab2)
  rnn.recursiveResizeAs(self.gradInput[2], tab2)
  rnn.recursiveFill(self.gradInput[2], 0)
  for i,v in ipairs(gradOutput) do
    local table_t = self.gradInput[1][i]
    if not table_t then
      table_t = v[1]:clone()
      self.gradInput[1][i] = table_t
    else
      table_t:resizeAs(v[1]):copy(v[1])
    end
    rnn.recursiveAdd(self.gradInput[2], v[2])
  end
  return self.gradInput
end


-- narrow table as second table

local NarrowTableAt,parent = torch.class("nn.NarrowTableAt","nn.Module")

function NarrowTableAt:updateOutput(input)
  local t1 = input[1]
  local at = input[2]
  self.output = _.tail(t1, at[1], #t1)
  return self.output
end

function NarrowTableAt:updateGradInput(input, gradOutput)
  local t1 = input[1]
  local at = input[2]
  self.dummy = self.dummy or t1[1]:clone():zero()
  self.dummy:resizeAs(t1[1])
  self.gradInput = {}
  self.gradInput[1] = tablex.copy(gradOutput)
  self.gradInput[2] = at:clone():zero()
  for i=1, at[1] do
    table.insert(self.gradInput[1], 1, self.dummy)
  end
  return self.gradInput
end
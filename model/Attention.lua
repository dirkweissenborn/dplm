
-- creates gModule with Input: {Control Tensor, {to_attend table, to_attend_projections}} ; Output: Attention
function controlledAttentionAlreadyProjected(size, control_size, interaction_size)
  control_size = control_size or size
  interaction_size = interaction_size or size

  local module = nn.Sequential()
  -- calculate projection to interaction layer only for controller because already done for to_attend

  module:add(
    nn.ConcatTable():add(
      nn.Sequential():add(
        nn.ParallelTable():add(
          nn.Linear(control_size,interaction_size)
        ):add(
          nn.SelectTable(2)
        ) --{controller_projections,to_attend_projections}
      ):add(
        nn.ZipTableWithTensor()
      ):add(
        nn.Sequencer(nn.Sequential():add(nn.CAddTable()):add(nn.Tanh()):add(nn.LinearNoBias(interaction_size,1)))
      ):add(
        nn.JoinTable(2,2)
      ):add(
        nn.SoftMax()
      )
    ):add(
      nn.Sequential():add(nn.SelectTable(2)):add(nn.SelectTable(1)) --to attend
    )
  ):add(nn.FMixtureTable())

  return module
end

-- creates gModule with Input: {Table of attendees, Control Tensor} ; Output: Attention
function controlledAttention(size, control_size, interaction_size)
  control_size = control_size or size
  interaction_size = interaction_size or size

  local module = nn.Sequential()

  -- calculate projection to interaction layer
  module:add(
    nn.ConcatTable():add(
      nn.Sequential():add(
        nn.ParallelTable():add(
          nn.Linear(control_size,interaction_size)
        ):add(
          nn.Sequencer(nn.LinearNoBias(size,interaction_size))
        ) --{controller,to_attend_projections}
      ):add(
        nn.ZipTableWithTensor()
      ):add(
        nn.Sequencer(nn.Sequential():add(nn.CAddTable()):add(nn.Tanh()):add(nn.LinearNoBias(interaction_size,1)))
      ):add(
        nn.JoinTable(2,2)
      ):add(
        nn.SoftMax()
      )
    ):add(
      nn.SelectTable(2) --to attend
    )
  ):add(nn.FMixtureTable())

  return module
end

-- creates gModule with Input: {Table of attendees, Table of control Tensors} ;
-- Output: Table of Attentions for each controller
function controlledMultiAttention(size, control_size, interaction_size)
  control_size = control_size or size
  interaction_size = interaction_size or size

  --Input
  local to_attend = nn.Identity()() --table of tensors
  local control   = nn.Identity()() --table of controllers

  -- Create inner module that is applied to each attendee
  local control_proj   = nn.Identity()()
  local part_projs     = nn.Identity()()
  local seq             = nn.Sequential()
  seq:add(nn.CAddTable())
  seq:add(nn.Tanh())
  seq:add(nn.LinearNoBias(interaction_size,1))
  local t_zip          = nn.ZipTableWithTensor()({part_projs, control_proj})
  local scores         = nn.JoinTable(2,2)(nn.Sequencer(seq)(t_zip))
  local proj_ms        = nn.gModule({control_proj,part_projs},{scores})

  -- calc all projs that are combined later
  local part_ps     = nn.Sequencer(nn.LinearNoBias(size, interaction_size))(to_attend)
  local control_ps  = nn.Sequencer(nn.Linear(control_size,interaction_size))(control)

  -- Use sequencer to apply proj to all attendees and then join scores
  local zip            = nn.ZipTableWithTable()({control_ps, part_ps})
  local scoress        = nn.Sequencer(proj_ms)(zip)
  local softmax        = nn.Sequencer(nn.SoftMax())(scoress)

  local scoresAndAttend= nn.ZipTableWithTable()({softmax,to_attend})
  local attentions     = nn.Sequencer(nn.FMixtureTable())(scoresAndAttend)

  return nn.gModule({control, to_attend},{attentions})
end

function simpleAttention(size)
  --Input
  local to_attend = nn.Identity()()
  -- Use sequencer to apply proj to all attendees and then join scores
  local scores         = nn.JoinTable(2,2)(nn.Sequencer(nn.LinearNoBias(size, 1))(to_attend))
  local softmax        = nn.SoftMax()(scores)
  local attention      = nn.MixtureTable()({softmax, to_attend})
  return nn.gModule({to_attend},{attention})
end

-- Fixed MixtureTable to make it compatible with attention

local FMixtureTable,parent = torch.class("nn.FMixtureTable","nn.MixtureTable")

function FMixtureTable:updateOutput(input)
  local gaterInput = input[1]
  if self.dimG and self.size:size() >= self.dimG then
    self.size[self.dim] = gaterInput:size(self.dimG)
  end
  return parent.updateOutput(self,input)
end


local ZipTableWithTensor, parent = torch.class("nn.ZipTableWithTensor","nn.Module")

function ZipTableWithTensor:__init()
end

function ZipTableWithTensor:updateOutput(input)
  local tab = input[2]
  local ten = input[1]
  self.output = {}
  for _,v in ipairs(tab) do table.insert(self.output, {v,ten}) end
  return self.output
end

function ZipTableWithTensor:updateGradInput(input, gradOutput)
  local ten = input[1]
  self.gradInput = self.gradInput or {}
  self.gradInput[2] = self.gradInput[2] and _.slice(self.gradInput[2],1,#gradOutput) or {}
  self.gradInput[1] = self.gradInput[1] or ten:clone()
  self.gradInput[1]:resizeAs(ten):zero()
  for i,v in ipairs(gradOutput) do
    local table_t = self.gradInput[2][i]
    if not table_t then
      table_t = v[2]:clone()
      self.gradInput[2][i] = table_t
    else
      table_t:resizeAs(v[2]):copy(v[2])
    end
    self.gradInput[1]:add(v[1])
  end
  return self.gradInput
end
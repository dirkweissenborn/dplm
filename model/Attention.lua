
-- creates gModule with Input: {Table of attendees, Control Tensor} ; Output: Attention
function controlledAttention(size, control_size, interaction_size)
  control_size = control_size or size
  interaction_size = interaction_size or size

  --Input
  local to_attend = nn.Identity()() --table of tensors
  local control   = nn.Identity()() --single controller

  -- Create inner module that is applied to each attendee
  local input_part     = nn.Identity()()
  local control_proj   = nn.Identity()()

  local part_proj      = nn.LinearNoBias(size, interaction_size)(input_part)
  local proj_out       = nn.Tanh()(nn.CAddTable()({part_proj,control_proj}))
  local score          = nn.LinearNoBias(size,1)(proj_out)
  local proj_m         = nn.gModule({input_part,control_proj},{score})

  -- Use sequencer to apply proj to all attendees and then join scores
  local control_p      = nn.Linear(control_size,interaction_size)(control)
  local zip            = nn.ZipTableWithTensor()({to_attend, control_p})
  local scores         = nn.JoinTable(2,2)(nn.Sequencer(proj_m)(zip))

  local softmax        = nn.SoftMax()(scores)
  local attention      = nn.FMixtureTable()({softmax, to_attend})

  return nn.gModule({to_attend,control},{attention})
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

  return nn.gModule({to_attend,control},{attentions})
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
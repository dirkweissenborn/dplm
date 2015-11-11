require "dp"
require "pl"
require "rnn"
require 'tds'

dplm = {}
torch.include('dplm', 'data/CharSource.lua')
torch.include('dplm', 'data/SplitCharSource.lua')
torch.include('dplm', 'model/CharLM.lua')
torch.include('dplm', 'sampler/LargeSentenceSampler.lua')
torch.include('dplm', 'train/ThresholdedAdaptiveDecay.lua')
torch.include('dplm', 'feedback/LossFeedback.lua')
torch.include('dplm', 'model/Criterion.lua')
torch.include('dplm', 'model/EncoderDecoder.lua')
torch.include('dplm', 'model/AttentionLSTM.lua')
torch.include('dplm', 'model/Attention.lua')
torch.include('dplm', 'model/AttentionalSkipStackedLSTM.lua')
torch.include('dplm', 'model/TableModules.lua')

-- Disambiguation
torch.include('dplm', 'el/DBMap.lua')
torch.include('dplm', 'el/Disambiguator.lua')
torch.include('dplm', 'data/AIDALoader.lua')


return dplm
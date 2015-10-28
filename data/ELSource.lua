

local ELSource = torch.class("dplm.ELSource")

function ELSource:__init(loader, disambiguator)
  self._disambiguator = disambiguator
  self._loader = loader

  _.each(loader.train, function() end)
end


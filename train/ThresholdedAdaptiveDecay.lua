
------------------------------------------------------------------------
--[[ ThresholdedAdaptiveDecay ]]--
-- Observer used by optimizer callbacks.
-- Decays decay attribute when error on validation didn't improve by
-- threshold for max_wait epochs.
-- Should observe in conjuction with a dp.ErrorMinima instance (such as 
-- EarlyStopper)
------------------------------------------------------------------------
local ThresholdedAdaptiveDecay, parent = torch.class("dp.ThresholdedAdaptiveDecay", "dp.Observer")

function ThresholdedAdaptiveDecay:__init(config)
  assert(type(config) == 'table', "Constructor requires key-value arguments")
  local args
  args, self._max_wait, self._threshold, self._decay_factor
    = xlua.unpack(
      {config},
      'AdaptiveDecay',
      'Decays learning rate when validation set does not reach '..
          'a new minima for max_wait epochs',
      {arg='max_wait', type='number', default=0,
        help='maximum number of epochs to wait for a new minima ' ..
            'to be found. After that, the learning rate is decayed.'},
      {arg='threshold', type='number', default=1e-2,
        help='threshold for expected minimal improvement'},
      {arg='decay_factor', type='number', default=0.5,
        help='Learning rate is decayed by lr = lr*decay_factor every '..
            'time a new minima has not been reached for max_wait epochs'}
    )
  parent.__init(self, "errorMinima")
  self._wait = 0
  -- public attribute
  self.decay = 1
  self._last_error = 1e10
end

function ThresholdedAdaptiveDecay:errorMinima(found_minima, error_minima)
  if found_minima then
    -- check if improvement is above threshold
    found_minima = self._last_error - self._threshold > error_minima._minima
  end
  self._last_error = error_minima._minima
  self._wait = found_minima and 0 or self._wait + 1
  if self._max_wait < self._wait then
    self._wait = 0
    self.decay = self._decay_factor
  else
    self.decay = 1
  end
end
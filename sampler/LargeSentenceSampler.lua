------------------------------------------------------------------------
--[[ SentenceSampler ]]--
-- Iterates over parallel sentences of equal size one word at a time.
-- The sentences size are iterated through randomly.
-- Used for Recurrent Neural Network Language Models.
-- Note that epoch_size is the minimum number of samples per epoch.
------------------------------------------------------------------------
local LargeSentenceSampler, parent = torch.class("dplm.LargeSentenceSampler","dp.SentenceSampler")

function LargeSentenceSampler:__init(config)
  parent.__init(self, config)
  -- the billion words validation set has a sentence of 820 words???
  -- but the test set is alright
  self._max_size = config.max_size or 999999999999
  self._context_size = config.context_size or 999999999999
  self._shuffle = true
  if config.shuffle ~= nil then
    self._shuffle = config.shuffle
  end

end

function LargeSentenceSampler:_sampleEpoch(dataset)
  dataset = dp.Sampler.toDataset(dataset)

  local sentenceStartId = dataset:startId()
  local sentenceEndId = dataset._end_id
  local sentenceTable_, corpus = dataset:groupBySize()
  local text = corpus:select(2,2) -- second column is the text

  -- remove sentences of size > self._max_size --> we process them also batch wise
  --[[sentenceTable_ = _.map(sentenceTable_,
    function(k,v)
      if k <= self._max_size then
        return v
      else
        return nil
      end
    end) ]]--

  local max = 0
  local sentenceTable = {}
  for size, s in pairs(sentenceTable_) do
    max = math.max(max,size)
    sentenceTable[size] = {indices=s.indices:resize(s.count), sampleIdx=1 }
  end

  -- put left overs of smaller sentences together with next bigger sentences to always have batch_size batches
  local left_overs = {}
  if max > 1 then
    for l=1,max-1 do
      local s = sentenceTable[l]
      if s then
        local count = s.indices:size(1)
        local left_over_count = count % self._batch_size
        if left_over_count > 0 then
          local nLeftOvers = #left_overs
          if (left_over_count+nLeftOvers) >= self._batch_size then
            local additional = self._batch_size - left_over_count
            local old_indices = s.indices
            s.indices = torch.zeros(count + additional)
            s.indices:sub(1,count):copy(old_indices)
            --fill with left overs
            for i=1, additional do
              s.indices[count+i] = left_overs[i]
            end
            --remove extracted left overs
            for i= 1, additional do left_overs[i] = nil end
            for i= additional +1, nLeftOvers do
              left_overs[i-additional] = left_overs[i]
              left_overs[i] = nil
            end
          else --cannot fill this up with leftovers -> move leftovers of this sentence size to global leftovers
            for i=1, left_over_count do
              table.insert(left_overs, s.indices[-i])
            end
            if count > left_over_count then --more than one batch
              s.indices = s.indices:resize(count-left_over_count)
            else
              sentenceTable[l] = nil
            end
          end
        end
      end
    end
    -- the rest goes into the max bucket
    local s = sentenceTable[max]
    local count = s.indices:size(1)
    local old_indices = s.indices
    s.indices = torch.zeros(count + #left_overs)
    s.indices:sub(1,count):copy(old_indices)
    --fill with left overs
    for i=1, #left_overs do s.indices[count+i] = left_overs[i] end
  end

  local nSample = 0
  for size, s in pairs(sentenceTable) do
    nSample = nSample + s.indices:size(1)
  end

  local epochSize = self._epoch_size or nSample
  local nSampled = 0
  local batch
  local function newBatch()
    return batch or dp.Batch{
      which_set=dataset:whichSet(), epoch_size=epochSize,
      inputs=dp.ClassView('bt', torch.IntTensor{{1}}),
      targets=dp.ClassView('bt', torch.IntTensor{{1}})
    }
  end
  
  -- Flag that can be queried to now whether the sample reached end of a batch
  self._end_of_batch = true

  while true do
    -- reset sampleIdx
    for _, s in pairs(sentenceTable) do
      s.sampleIdx=1
    end
    
    local sentenceSizes = _.sort(_.keys(sentenceTable))
    if self._shuffle then sentenceSizes = _.shuffle(sentenceSizes) end
    local nSizes = #sentenceSizes

    while nSizes > 0 do

      for i,_sentenceSize in pairs(sentenceSizes) do
        local s = sentenceTable[_sentenceSize]
        local sentenceSize = math.min(_sentenceSize,self._max_size)
        local start = s.sampleIdx
        local stop = math.min(start + self._batch_size - 1, s.indices:size(1))

        -- batch of word indices, each at same position in different sentence
        local textIndices = s.indices:narrow(1, start, stop - start + 1)
        self._text_indices = self._text_indices or torch.LongTensor()
        self._text_indices:resize(textIndices:size(1))
        self._text_indices:copy(textIndices)
        textIndices = self._text_indices

        batch = batch or newBatch()
        local input_v = batch:inputs()
        assert(torch.isTypeOf(input_v, 'dp.ClassView'))
        local inputs = input_v:input() or torch.IntTensor()
        inputs:resize(textIndices:size(1), sentenceSize+1)
        local target_v = batch:targets()
        assert(torch.isTypeOf(target_v, 'dp.ClassView'))
        local targets = target_v:input() or torch.IntTensor()
        targets:set(inputs:narrow(2,2,inputs:size(2)-1))
        -- metadata
        batch:setup{
          batch_iter=(nSampled + textIndices:size(1) - 1),
          batch_size=self._batch_size,
          n_sample=textIndices:size(1)
        }

        for wordOffset=1,sentenceSize do
          if wordOffset == 1 then
            inputs:select(2,1):fill(sentenceStartId)
          end

          local target = inputs:select(2,wordOffset+1)
          target:index(text, 1, textIndices)

          -- move to next word in each sentence
          for i=1,textIndices:size(1) do
            -- only add 1 if this is not end of sentence
            if text[textIndices[i]] ~= sentenceEndId then
              textIndices[i] = textIndices[i] + 1
            end
          end  
        end

        inputs = inputs:narrow(2, 1, sentenceSize)
        nSampled = nSampled + textIndices:size(1)

        input_v:setClasses(dataset:vocabulary())
        target_v:setClasses(dataset:vocabulary())
        for i=1, math.ceil(sentenceSize/self._context_size) do
          local offset = (i-1)*self._context_size
          local end_i = math.min(sentenceSize, offset + self._context_size)
          -- re-encapsulate in dp.Views
          input_v:forward('bt', inputs:sub(1,-1,offset+1,end_i))
          target_v:forward('bt', targets:sub(1,-1,offset+1,end_i))
          self._end_of_batch = end_i == sentenceSize
          coroutine.yield(batch, math.min(nSampled, epochSize), epochSize)
        end


        if nSampled >= epochSize then
          batch = coroutine.yield(false)
          nSampled = 0
        end

        s.sampleIdx = s.sampleIdx + textIndices:size(1)
        if s.sampleIdx > s.indices:size(1) then
          sentenceSizes[i] = nil
          nSizes = nSizes - 1
        end
      end

      self:collectgarbage()

    end
  end
end
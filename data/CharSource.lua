local CharSource, parent = torch.class("dplm.CharSource", "dp.DataSource")
CharSource.isCharSource = true

function CharSource:__init(config)
  config = config or {}
  assert(torch.type(config) == 'table' and not config[1],
    "Constructor requires key-value arguments")
  local args, train, valid, test, split_fracs
  args, self._name, self._context_size, self._recurrent,
  self._bidirectional, self._data_path, self._string, split_fracs, self._sentence = xlua.unpack(
    {config},
    'CharSource',
    'Creates a DataSource out of 3 strings or text files',
    {arg='name', type='string', req=true,
      help='name of datasource. Also name of directory in data_path'},
    {arg='context_size', type='number', default=1,
      help='number of previous words to be used to predict the next one.'},
    {arg='recurrent', type='number', default=false,
      help='For RNN training, set this to true. In which case, '..
          'outputs a target word for each input word'},
    {arg='bidirectional', type='boolean', default=false,
      help='For BiDirectionalLM, i.e. Bidirectional RNN/LSTMs, '..
          'set this to true. In which case, target = input'},
    {arg='data_path', type='string', req=true,
      help='path to train, valid, test files'},
    {arg='string', type='boolean', default=false,
      help='set this to true when the *file args are the text itself'},
    {arg='split_fractions', type='table', default={1,0,0},
      help='split fractions for training/valid/test, which should sum to a maximum of one'},
    {arg='sentence', type='boolean', default=true,
      help='split fractions for training/valid/test, which should sum to a maximum of one'}
  )

  -- perform safety checks on split_fractions
  -- perform safety checks on split_fractions
  assert(split_fracs[1] >= 0 and split_fracs[1] <= 1, 'bad split fraction ' .. split_fracs[1] .. ' for train, not between 0 and 1')
  assert(split_fracs[2] >= 0 and split_fracs[2] <= 1, 'bad split fraction ' .. split_fracs[2] .. ' for val, not between 0 and 1')
  assert(split_fracs[3] >= 0 and split_fracs[3] <= 1, 'bad split fraction ' .. split_fracs[3] .. ' for test, not between 0 and 1')
  assert(split_fracs[1] + split_fracs[2] + split_fracs[3] <= 1, "split fraction sum > 1 not allowed!")

  self._classes = {}

  -- everything is loaded in the same order
  -- to keep the mapping of word to word_id consistent
  local train, valid, test = self:createDataSets(split_fracs)
  self:trainSet(train)
  self:validSet(valid)
  self:testSet(test)

  parent.__init(self, {
    train_set=self:trainSet(),
    valid_set=self:validSet(),
    test_set=self:testSet()
  })
end

function CharSource:createDataSets(split_fracs)
  require 'pl'

  local data_dir = self._data_path
  if data_dir == nil or data_dir == '' then
    return
  end

  local input_file = path.join(data_dir, 'input.txt')
  assert(path.exists(input_file),"CharSource requires file: " .. input_file)
  local vocab_file = path.join(data_dir, 'vocab_lw.t7')
  local tensor_file = path.join(data_dir, 'data_lw.t7')

  -- fetch file attributes to determine if we need to rerun preprocessing
  local run_prepro = false
  if not (path.exists(vocab_file) or path.exists(tensor_file)) then
    -- prepro files do not exist, generate them
    print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
    run_prepro = true
  else
    -- check if the input file was modified since last time we
    -- ran the prepro. if so, we have to rerun the preprocessing
    local input_attr = lfs.attributes(input_file)
    local vocab_attr = lfs.attributes(vocab_file)
    local tensor_attr = lfs.attributes(tensor_file)
    if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
      print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
      run_prepro = true
    end
  end

  local data
  if run_prepro then
    -- construct a tensor with all the data, and vocab file
    print('one-time setup: preprocessing input text file ' .. input_file .. '...')
    data, self.vocab = self:text_to_tensor(input_file, vocab_file, tensor_file)
  else
    self.vocab = torch.load(vocab_file)
    data = torch.load(tensor_file)
  end

  for word, word_id in pairs(self.vocab) do
    self._classes[word_id] = word
  end

  local len = data:size(1)
  self._sentence_start = self.vocab["<S>"]
  self._sentence_end = self.vocab["</S>"]

  local train_l = math.floor(len * split_fracs[1])
  while data[train_l][2] ~= self._sentence_end do
    train_l = train_l - 1
  end
  local train = self:create_dataset(data:sub(1, train_l, 1, -1),"train")
  data:sub(1, train_l, 1, -1)
  local valid_l = math.floor(len * split_fracs[2])
  local valid
  if valid_l > 0 then
    while data[train_l+valid_l][2] ~= self._sentence_end do
      valid_l = valid_l - 1
    end
    local valid_data = data:sub(train_l+1, train_l+valid_l, 1, -1)
    valid_data:select(2,1):add(-train_l)
    valid = self:create_dataset(valid_data,"valid")
  end
  local test_l = math.floor(len * split_fracs[3])
  local test
  if test_l > 0 then
    while data[train_l+valid_l+test_l][2] ~= self._sentence_end do
      test_l = test_l + 1
    end
    local test_data = data:sub(train_l+valid_l+1, train_l+valid_l+test_l, 1, -1)
    test_data:select(2,1):add(-train_l-valid_l)
    test = self:create_dataset(test_data,"test")
  end

  return train, valid, test
end

function CharSource:create_dataset(data, whichset)
  if self._sentence then
    return dp.SentenceSet{
      data=data, which_set=whichset,
      context_size=self._context_size, end_id=self._sentence_end, start_id=self._sentence_start,
      words=self._classes, recurrent=self._recurrent
    }
  else
    return dp.TextSet{
      data=data:select(2,2), which_set=whichset, context_size=self._context_size,
      recurrent=self._recurrent, bidirectional=self._bidirectional,
      words=self._classes
    }
  end
end

--static
function CharSource:text_to_tensor(in_textfile, out_vocabfile, out_tensorfile)
  local cache_len = 10000
  local tot_len = 0
  local f = io.open(in_textfile, "r")
  -- create vocabulary if it doesn't exist yet
  -- record all characters to a set
  local unordered = { ["</S>"] = true, ["<S>"] = true, ["<U>"] = true }
  local rawdata = f:read(cache_len)
  local word_split
  repeat
    for char in rawdata:gmatch '.' do
      if char ~= "\n" and not unordered[char] then unordered[char] = true end
    end
    tot_len = tot_len + #rawdata
    rawdata = f:read(cache_len)
  until not rawdata

  f:close()
  -- sort into a table (i.e. keys become 1..N)
  local ordered = {}
  for char in pairs(unordered) do ordered[#ordered + 1] = char end
  table.sort(ordered)
  -- invert `ordered` to create the char->int mapping
  local vocab = {}
  for i, char in ipairs(ordered) do
    vocab[char] = i
  end
  -- construct a tensor with all the data
  local d
  if #ordered < 256 then d = torch.IntTensor(tot_len,2)
  elseif #ordered < 32767 then d = torch.ShortTensor(tot_len,2) end
  f = io.open(in_textfile, "r")
  local i = 0
  local end_i = vocab["</S>"]
  for rawdata in f:lines() do
    local start_index = i+1
    for char in rawdata:gmatch '.' do
      i = i + 1
      d[i][2] = vocab[char]
      d[i][1] = start_index
    end
    if self._sentence then
      i = i+1
      d[i][2] = end_i
      d[i][1] = start_index
    end
  end
  f:close()

  -- save output preprocessed files
  print('saving ' .. out_vocabfile)
  torch.save(out_vocabfile, vocab)
  print('saving ' .. out_tensorfile)
  torch.save(out_tensorfile, d)
  return d, vocab
end

function CharSource:vocabulary()
  return self._classes
end

function CharSource:vocabularySize()
  return table.length(self._classes)
end
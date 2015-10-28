local CharSource, parent = torch.class("dplm.CharSource", "dp.DataSource")
CharSource.isCharSource = true

function CharSource:__init(config)
  config = config or {}
  assert(torch.type(config) == 'table' and not config[1],
    "Constructor requires key-value arguments")
  local args, train, valid, test
  args, self._context_size, self._recurrent,
  self._bidirectional, train, valid, test,
  self._string, self.vocab, self._sentence
    = xlua.unpack(
    {config},
    'TextSource',
    'Creates a DataSource out of 3 strings or text files',
    {arg='context_size', type='number', default=1,
      help='number of previous words to be used to predict the next one.'},
    {arg='recurrent', type='number', default=false,
      help='For RNN training, set this to true. In which case, '..
          'outputs a target word for each input word'},
    {arg='bidirectional', type='boolean', default=false,
      help='For BiDirectionalLM, i.e. Bidirectional RNN/LSTMs, '..
          'set this to true. In which case, target = input'},
    {arg='train', type='string',
      help='training text data or name of training file'},
    {arg='valid', type='string',
      help='validation text data or name of validation file'},
    {arg='test', type='string',
      help='test text data or name of test file'},
    {arg='string', type='boolean', default=false,
      help='set this to true when the *file args are the text itself'},
    {arg='vocab', type='table', 'vocabulary to be used'},
    {arg='sentence', type='boolean', default=true,
      help='split fractions for training/valid/test, which should sum to a maximum of one'}
  )

  local function load(textOrFile)
    local text = textOrFile
    if text and not self._string then
      text = file.read(textOrFile)
    end
    return text
  end

  train, valid, test = load(train), load(valid), load(test)

  if not self.vocab then
    print("Creating new vocabulary for datasource, because none was provided.")
    self.vocab = CharSource.create_vocab(table.concat({train, valid, test}))
  end
  self._sentence_start = self.vocab["<S>"]
  self._sentence_end = self.vocab["</S>"]

  self._classes = {}
  for word, word_id in pairs(self.vocab) do
    self._classes[word_id] = word
  end

  -- everything is loaded in the same order
  -- to keep the mapping of word to word_id consistent
  self:trainSet(self:createDataSet(train, 'train'))
  if valid then self:validSet(self:createDataSet(valid, 'valid')) end
  if test then self:testSet(self:createDataSet(test, 'test')) end

  parent.__init(self, {
    train_set=self:trainSet(),
    valid_set=self:validSet(),
    test_set=self:testSet()
  })
end


function CharSource:createDataSet(text, whichSet)
  local data = CharSource.text_to_tensor(text, self.vocab, self._sentence)
  local ds = self:create_dataset(data,whichSet)
  return ds
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


function CharSource:vocabulary()
  return self._classes
end

function CharSource:vocabularySize()
  return table.length(self._classes)
end

function CharSource.create_vocab(text)
  -- create vocabulary if it doesn't exist yet
  -- record all characters to a set
  local unordered = { ["</S>"] = true, ["<S>"] = true, ["<U>"] = true }
  for char in text:gmatch '.' do
    if char ~= "\n" and not unordered[char] then unordered[char] = true end
  end
  -- sort into a table (i.e. keys become 1..N)
  local ordered = {}
  for char in pairs(unordered) do ordered[#ordered + 1] = char end
  table.sort(ordered)
  -- invert `ordered` to create the char->int mapping
  local vocab = {}
  for i, char in ipairs(ordered) do
    vocab[char] = i
  end
  return vocab
end

  --static
function CharSource.text_to_tensor(text, vocab, sentence)
  -- construct a tensor with all the data
  local tot_len = string.len(text)
  if not sentence then tot_len = tot_len end
  local d
  local vocab_size = tablex.size(vocab)
  if vocab_size < 256 then d = torch.IntTensor(tot_len+1,2)
  elseif vocab_size < 32767 then d = torch.ShortTensor(tot_len+1,2) end
  local i = 0
  local end_i = vocab["</S>"]
  for line in stringx.lines(text) do
    local start_index = i+1
    for char in line:gmatch '.' do
      i = i + 1
      d[i][2] = vocab[char] or vocab["<U>"]
      d[i][1] = start_index
    end
    i = i+1
    d[i][2] = end_i
    d[i][1] = start_index
  end

  return d, vocab
end

function CharSource:vocabulary()
  return self._classes
end

function CharSource:vocabularySize()
  return table.length(self._classes)
end
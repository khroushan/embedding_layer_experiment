import numpy as np

### method to extract the Markov transition matrix
def shift(in_tup, element):
    return in_tup[1:] + (element,)

def markov_dict_extractor(events: list, mk_len = 2):
  """ Count number of transition from state `w_i` to `w_i+1`. Calculate the 
  transition probabilities for available states.

  input:
  ------
  events: list, sequence of states
  mk_len: int, number of states gathered

  output:
  -------
  transition_dict: dict of tuples: count transition from one state to another.
  transition_norm: dict of tuples: transition rate (prob.)

  TODO:
  - the norm is not correct, the sum of prob. in each columns and rows
  must be one.
  - a function that convert the transition dictionary to a sparse matrix 
  representation with dictionary of indx and jndx to corresponding words.
  """
  events.append('EMPTY') # to count the last element as well
  transition_dic = {}
  pre = ()
  for event in events:
      if len(pre) < mk_len:
          pre +=  (event,)
      else:
          try:
              transition_dic[pre] += 1
          except: 
              transition_dic[pre] = 1
          pre = shift(pre, event)
          
          
  return transition_dic

def get_vocabs_dict(text: str, lowercase=True):
  """ Get vocaburaly of the text

    - convert it to lower case
  """
  vocaburaly = set(text.lower().split())
  # reserve 0 for unknown vocab in the text
  vocab_dict = {v:i for v,i in zip(vocaburaly, range(1, len(vocaburaly)+1))}
  return vocab_dict


def transition_mtx_to_sparse(trans_dict: dict, mk_len=2):
  """ Sparse matrix representaion of markov transition dictionary.
  Extract i-index, j-index and value transition 
  from transition dict 
  """
  if mk_len == 2:
    indx = [c[0] for c in trans_dict.keys()]
    jndx = [c[1] for c in trans_dict.keys()]
    cnt_vect = [v for v in trans_dict.values()] 
  elif mk_len == 1:
    indx = [c[0] for c in trans_dict.keys()]
    jndx = []
    cnt_vect = [v for v in trans_dict.values()]
  else:
    print("order of markov unit must be <= 2 !")

  return np.array(indx), np.array(jndx), np.array(cnt_vect)


def renormalize_mtx(indx, jndx, cnt_vect):
  """ Renormalize given sparse transition matrix
  """
  indx_unique_freq = {}
  for w in np.unique(indx):
    indx_unique_freq[w] = cnt_vect[indx==w].sum()

  cnt_renorm = np.zeros(cnt_vect.shape, float)
  for i,w in enumerate(indx):
    cnt_renorm[i] = cnt_vect[i]/indx_unique_freq[w]

  return indx, jndx, cnt_renorm


def extract_markov_transition(events: list, mk_len = 2):
  """ Evaluate transition probability for a list of events
  """
  assert mk_len == 2
  # Evaluate transition counts
  dict_text = markov_dict_extractor(events, mk_len=2)
  word_counts = markov_dict_extractor(events, mk_len=1)
  
  trans_prob = {}
  for k in dict_text.keys():
    # Divide by total occurance of each keywords
    trans_prob[k] = dict_text[k]/word_counts[(k[0],)]

  return trans_prob   

def get_overlap_sparse_mtx(indx, jndx, cnt_vect, 
                          sub_indx, sub_jndx):
    """ To get the subset of original matrix for given indx and jndx
    """
    ov_indx = []
    ov_jndx = []
    ov_cnt = []
    for i,j in zip(k,l):
        if np.sum((a == i) & (b == j)) == 1:
            ov_indx.append(a[a == i][0])
            ov_jndx.append(b[b == j][0])
            ov_cnt.append(c[(a == i) & (b == j)][0])
            
    return np.array(ov_indx), np.array(ov_jndx), np.array(ov_cnt)
    
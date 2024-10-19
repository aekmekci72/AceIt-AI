from typing import OrderedDict
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, BertModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import spacy
from transformers import BertTokenizer as BTokenizer, BertModel as BModel
from warnings import filterwarnings as ignore_warnings
from transformers import pipeline

ignore_warnings('ignore')

qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def generate_question_answer(sentence, answer):
    t5_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    t5_tokenizer = AutoTokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

    text = "context: {} answer: {}".format(sentence, answer)
    max_len = 256
    encoding = t5_tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = t5_model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 early_stopping=True,
                                 num_beams=5,
                                 num_return_sequences=1,
                                 no_repeat_ngram_size=2,
                                 max_length=300)

    decoded_outputs = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]

    question = decoded_outputs[0].replace("question:", "")
    question = question.strip()

    answer_output = qa_pipeline(question=question, context=txt)

    return question, answer_output['answer']

bert_tokenizer = BTokenizer.from_pretrained('bert-base-uncased')
bert_model = BModel.from_pretrained("bert-base-uncased")
sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
nlp = spacy.load("en_core_web_sm")

def generate_question(sentence, answer):
  t5_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
  t5_tokenizer = AutoTokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

  text = "context: {} answer: {}".format(sentence,answer)
  max_len = 256
  encoding = t5_tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outputs = t5_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=300)

  decoded_outputs = [t5_tokenizer.decode(ids,skip_special_tokens=True) for ids in outputs]

  question = decoded_outputs[0].replace("question:","")
  question = question.strip()
  return question

def calculate_embedding(doc):
  bert_tokenizer = BTokenizer.from_pretrained('bert-base-uncased')
  bert_model = BModel.from_pretrained("bert-base-uncased")
  
  tokens = bert_tokenizer.tokenize(doc)
  token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
  segment_ids = [1] * len(tokens)

  torch_tokens = torch.tensor([token_ids])
  torch_segments = torch.tensor([segment_ids])

  return bert_model(torch_tokens, torch_segments)[-1].detach().numpy()

def get_parts_of_speech(context):
  doc = nlp(context)
  pos_tags = [token.pos_ for token in doc]
  return pos_tags, context.split()

def get_sentences(context):
  doc = nlp(context)
  return list(doc.sents)

def get_vectorizer(doc):
  stop_words = "english"
  n_gram_range = (1,1)
  vectorizer = CountVectorizer(ngram_range = n_gram_range, stop_words = stop_words).fit([doc])
  return vectorizer.get_feature_names_out()

def get_keywords(context, module_type = 't'):
  keywords = []
  top_n = 5
  for sentence in get_sentences(context):
    key_words = get_vectorizer(str(sentence))
    print(f'Vectors : {key_words}')
    if module_type == 't':
      sentence_embedding = calculate_embedding(str(sentence))
      keyword_embedding = calculate_embedding(' '.join(key_words))
    else:
      sentence_embedding = sentence_model.encode([str(sentence)])
      keyword_embedding = sentence_model.encode(key_words)
    
    distances = cosine_similarity(sentence_embedding, keyword_embedding)
    print(distances)
    keywords += [(key_words[index], str(sentence)) for index in distances.argsort()[0][-top_n:]]

  return keywords

txt = """
Deviations from the five conditions for Hardy-Weinberg equilibrium can alter allele frequencies in a population, microevolution The three main causes of evolutionary change are natural selection, genetic drift, and gene flow Genetic drift - a change in the gene pool of a population due to chance, the effects of this are most pronounced in small populations Two situations in which genetic drift can have a significant impact on a population are those that produce the bottleneck effect and the founder effect Catastrophes that drastically reduce population size is called bottleneck effect; and the surviving population is likely not genetically representative of the original population, changing the gene pool Founder effect - when a few individuals become isolated from a larger population and form a new population whose gene pool is not reflective of the original population Gene flow - the transfer of alleles from one population to another as a result of movement of individuals or their gametes"""
qa_pairs = []
answer_dict = OrderedDict()

for answer, context in get_keywords(txt, 'st'):
    question, generated_answer = generate_question_answer(context, answer)
    if generated_answer not in answer_dict:
        answer_dict[generated_answer] = question
        qa_pairs.append((question, generated_answer))

print("Unique Question-Answer Pairs:")
for qa_pair in qa_pairs:
    print(f"Question: {qa_pair[0]}")
    print(f"Answer: {qa_pair[1]}")
    print()
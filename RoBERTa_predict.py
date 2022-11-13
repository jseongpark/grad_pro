from RoBERTa_functions import *
    
def load_model():
  current_path = get_current_path() + save_model_path
  return tf.keras.models.load_model(current_path, custom_objects={'TFRobertaModel':TFRobertaModel})
   
def get_result(pre_list):
  score = pre_list[0] * 100
  return score

def detect_curse(score):
  if score < 50:
    print(">> 욕이 아님 ( %.2f%% )" % score)
  else:
    print(">> !!욕이다( %.2f%% )" % score)

def get_predict_by_model(model, tokenizer, str):
  test_text = [str]
  test_tok = roberta_encode(test_text, tokenizer)
  test_input = test_tok 
  pre = model.predict(test_input)
  return get_result(pre.reshape(-1))

def test_model(model, tokenizer):
  while(True):
    input_text = input("문장을 입력하세요. (종료는 -1) :")
    if input_text == "-1":
      break
    else:
      score = get_predict_by_model(model, tokenizer, input_text)
      detect_curse(score)

def get_predict(str):
  model = load_model()
  tokenizer = get_tokenizer()
  return get_predict_by_model(model, tokenizer, str)

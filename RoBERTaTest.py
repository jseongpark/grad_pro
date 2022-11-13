from RoBERTa_predict import *

input_str = "나는 오늘 아침밥을 먹었다"

# HAVE TO LOAD
model = load_model()
tokenizer = get_tokenizer()

# encode_test(tokenizer, input_str)
# GET PREDICT SCORE
predict = get_predict_by_model(model, tokenizer, input_str)

# DETECT INPUT IS CURSE
detect_curse(predict)

# IF YOU WANT LOOP TEST, REMOVE # AND TRY BELOW LINE
test_model(model, tokenizer)
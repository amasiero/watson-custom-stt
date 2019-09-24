from ibm_watson import SpeechToTextV1
import time
import json
import io

speech_to_text = SpeechToTextV1(
	iam_apikey = "api_key",
  	url = "https://stream.watsonplatform.net/speech-to-text/api"
)

language_models = speech_to_text.list_language_models().get_result()

customizations = language_models["customizations"]

for customization in customizations:
	id = customization["customization_id"]
	print("deleting customization:", id)
	speech_to_text.delete_language_model(id)

language_model = speech_to_text.create_language_model(
		'Education language model',
		'pt-BR_BroadbandModel',
		description = 'An education language model to use on distance learning'
	).get_result()


customization_id = language_model["customization_id"]
print("creating customization:", customization_id)

language_model = speech_to_text.get_language_model(customization_id).get_result()
status = language_model["status"]

with io.open('corpus.txt', mode = 'r', encoding = 'utf-8') as corpus_file:
	speech_to_text.add_corpus(
		customization_id,
		'CORPUS1',
		corpus_file
		)

corpus = speech_to_text.get_corpus(
		customization_id,
		'CORPUS1'
	).get_result()

while corpus['status'] != 'analyzed':
	time.sleep(2)
	corpus = speech_to_text.get_corpus(customization_id, 'CORPUS1').get_result()

words = speech_to_text.list_words(customization_id).get_result()

with open('words.json', 'w+') as words_file:
	words_file.write(json.dumps(words))

speech_to_text.train_language_model(customization_id)

while language_model['status'] != 'available':
	time.sleep(2)
	language_model = speech_to_text.get_language_model(customization_id).get_result()

print('Language model trainning completed!')
print('Save customization_id:', customization_id)

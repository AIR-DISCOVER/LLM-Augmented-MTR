import base64
import json
import requests
import os
import ast
import argparse
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='MTR Interested Agents Visualization.')
parser.add_argument('--batch_tag', '--bt', default=1, type=int, help='batch visulazation tag.')
parser.add_argument('--batch_size', '--bz', default=500, type=int, help='batch visulazation tag.')
parser.add_argument('--key_list', '--key', default=1, type=int, help='batch visulazation tag.')
parser.add_argument('--output_name', '--on', default='test', type=str, help='batch visulazation tag.')

# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def main(args):
	# key from openai
	api_key_list = ["sk-*****"]
	api_key = api_key_list[key_list - 1]

	batch_tag, batch_size, key_list, output_name = args.batch_tag, args.batch_size, args.key_list, args.output_name
	if batch_tag == 0:
		print('batch_tag must start from 1.')
		return 

	file_path = 'LLM-Augmented-MTR/llm_augmented_prompt/data/samples'
	response_dir = 'LLM-Augmented-MTR/llm_augmented_prompt/prompt/output/' + output_name
	prompt_path = "LLM-Augmented-MTR/llm_augmented_prompt/data/text_prompt.txt"
	error_dir = 'LLM-Augmented-MTR/llm_augmented_prompt/prompt/output' + + output_name + '/info'
	error_path = error_dir + '/error_image_' + output_name + '.txt'
	os.makedirs(response_dir, exist_ok=True)
	os.makedirs(error_dir, exist_ok=True)

	image_files = []
	gt_directions = []
	filenames_all = sorted(os.listdir(file_path), key=lambda x: int(x.split('_')[0], 16) + int(x.split('_')[1]))
	for filename in filenames_all:
		if filename.endswith(".png"):
			parts = filename.split('_')
			last_part = parts[-1]
			gt_directions.append(last_part[:-4])
			image_files.append(os.path.join(file_path, filename))

	# get batch filenames
	image_files = image_files[(batch_tag - 1) * batch_size: batch_tag * batch_size]
	file_count = len(image_files)
	for idx in tqdm(range(file_count)):
		response_file_name = os.path.basename(image_files[idx]).split('.')[0]
		response_output_path = os.path.join(response_dir, response_file_name + '.json')
		if os.path.exists(response_output_path): 
			continue
		# print(image_files[idx])
  
		# Get tc-map and prompt(text-prompt + caption)
		txt_idx_path = os.path.join(file_path, response_file_name.split('.')[0] + '.txt')
		with open(txt_idx_path, 'r', encoding='UTF-8') as f:
			image_caption = f.read()
		with open(prompt_path, "r", encoding='UTF-8') as f2:
			prompt_content = f2.read()
		prompt_content = prompt_content + "\n\n" +  "Q: Please read the above documentation carefully and remember what problems need to be solved." + image_caption + "\n" + "A: "

		# Use gpt4-v
		base64_image = encode_image(image_files[idx])
		headers = {
			"Content-Type": "application/json",
			"Authorization": f"Bearer {api_key}"
		}
		system_prompt2 = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2021-09\nCurrent date: 2024-01-17"
		payload = {
			"model": "gpt-4-vision-preview",
			"messages": [
				{	
					"role": "system", 
					"content": system_prompt2
				},
				{
					"role": "user",
					"content": [
						{
							"type": "text",
							"text": prompt_content
						},
						{
							"type": "image_url",
							"image_url": {
								"url": f"data:image/png;base64,{base64_image}",
								
							}
						}
					]
				}
			],
			"temperature": 0.7,
			"top_p": 0.9,
			"max_tokens": 4000
		}

		error_times = 0
		while True:
			try:
				response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
				response_new = response.json()['choices'][0]['message']['content']
				response_new = ast.literal_eval(response_new)

				with open(response_output_path, 'w') as f:
					json.dump(response.json(), f)	
				# time.sleep(3.0)	
				break
			except KeyboardInterrupt:
				return
			except Exception as e:
				error_times += 1
				time.sleep(10.0)
				if error_times > 1:
					with open(error_path, 'a') as f:
						f.write(response_file_name + '\n')
					break
				print(e)
				print(image_files[idx])


if __name__ == "__main__":
	args = parser.parse_args()
	main(args)

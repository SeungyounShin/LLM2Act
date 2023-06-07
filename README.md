# LLM2Act

# Dataset Split Information

## Mbpp

curl -X GET "https://datasets-server.huggingface.co/splits?dataset=mbpp"

|                    |                   |
| ------------------ | ----------------- |
| InstructGPT (fewshot) | DEMO: 50<br>SR: 0.4 |
| LLama              | DEMO: 50<br>SR: 0.04 |
|                    |                   |

## Fever

[Fact checking thinking](https://www.notion.so/Fact-checking-thinking-0074e0b3c2574a66a9a26633dac70d91?pvs=21)

|                    | em    |
| ------------------ | ----- |
| InstructGPT (fewshot) | 0.54  |
| gpt2-large (fewshot) | 1 500 0.002  |
| llama (fewshot)    | 188 500 0.376  |
| llama (fewshot-chat) | 181 500 0.362 |
|                    |       |
| llama (webshop-fever) |       |

## Webshop

500-550 eval

|                    | SR                    |
| ------------------ | --------------------- |
| InstructGPT (fewshot) | 50 0.617 0.32 0.0       |
| llama7B (fewshot)  | 0                     |
| alpaca (fewshot)   | 0                     |
| llama7B-SFT (1)    | 50 0.676 0.4 0.0        |
| GPT2-large (1)     | 50 0.420 0.12 0.0       |
| llama7B-SFT (2)    | 50 0.706 0.44 0.0       |
| llama7B-SFT Chat   | 50 0.741 0.46 0.0       |
| llama7B-SFT Chat Full | 50 0.676 0.36 0.0       |
|                    |                       |
| webshop-fever-mbpp | 50 0.6726 0.38       |

(1): base-model(llama7B) InstructGPT 로 0-500까지의 webshop train demo 를 react 방식 prompt로 모으고 여기서 1.0 의 reward 를 받은 173개의 데이터를 instruction 부분만 나머지 모든 demo(action, obs) 를 모두 SFT 훈련을함. temperature 0.2 top_p 0.8 에서 sampling.

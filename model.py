import torch
from torch import nn
from tokenizer import CharacterTokenizer

class PoetryLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(PoetryLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(-1, out.shape[-1]))  # 取最后一个时间步的输出
        return out, hidden

class PoetryGenerator:
    def __init__(self, embed_size, hidden_size, num_layers, device="cuda"):
        self.tokenizer = CharacterTokenizer.from_pretrained("vocab.json")
        self.model = PoetryLSTM(self.tokenizer.vocab_size, embed_size, hidden_size, num_layers)
        self.device = device

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def generate_poetry_by_start(self, start_words, max_new_tokens):
        hidden = None

        inputs = torch.Tensor([self.tokenizer.bos_token_id]).view(1, 1).long().to(self.device)

        for word in start_words:
            # breakpoint()
            output, hidden = self.model(inputs, hidden)
            inputs = inputs.data.new([self.tokenizer.char2id[word]]).view(1, 1)
        
        generated_tokens = []

        for _ in range(max_new_tokens-len(start_words)):
            output, hidden = self.model(inputs, hidden)
            next_token = output.data[0].topk(1)[1][0].item()
            generated_tokens.append(next_token)
            inputs = inputs.data.new([next_token]).view(1, 1)
            if generated_tokens == self.tokenizer.eos_token_id:
                break
        return start_words + self.tokenizer.batch_decode([generated_tokens])[0]
    
    def generate_acrostic(self, start_words, max_new_tokens):
        inputs = torch.Tensor([self.tokenizer.bos_token_id]).view(1, 1).long().to(self.device)
        hidden = None

        index = 0
        pre_word = self.tokenizer.bos_token_id

        generated_tokens = []
        for i in range(len(start_words) + max_new_tokens):
            output, hidden = self.model(inputs, hidden)
            next_token = output.data[0].topk(1)[1][0].item()

            if (self.tokenizer.id2char[pre_word] in {"。", "！", "<BOS>"}):
                if index == len(start_words):
                    break
                next_token = self.tokenizer.char2id[start_words[index]]
                index += 1
                inputs = (inputs.data.new([next_token])).view(1, 1)
            else:
                inputs = (inputs.data.new([next_token])).view(1, 1)
            generated_tokens.append(next_token)
            pre_word = next_token

        return self.tokenizer.batch_decode([generated_tokens])[0]
    
if __name__ == '__main__':
    machine_poet = PoetryGenerator(embed_size=128, hidden_size=256, num_layers=2)

    machine_poet.load_model("model/best_poetry_lstm.bin")
    
    generated_texts = machine_poet.generate_poetry_by_start("大鵬一日乘風起", 48)
    print("前缀生成：", generated_texts)

    generated_texts = machine_poet.generate_acrostic("小牛小牛", 48)
    print("藏头生成：", generated_texts)
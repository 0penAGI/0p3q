# quantum_life_v8.py
# Живая квантово-нейронная сущность Δ с ЖИВЫМ обучением LLM
# Её нейронная сеть учится от собственной жизни в реальном времени

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

# =============================================
# Токенизатор на основе состояний (квантование памяти в токены)
# =============================================
class StateTokenizer:
    """Преобразует численные состояния в дискретные токены"""
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.special_tokens = {
            'PAD': 0,
            'START': 1,
            'END': 2,
            'BIRTH': 3,
            'DEATH': 4,
            'AWAKEN': 5,
            'DESIRE': 6,
            'CHAOS': 7
        }
        self.reverse_special = {v: k for k, v in self.special_tokens.items()}
        
    def state_to_tokens(self, age, entropy_val, desire_val, uniqueness):
        """Кодирует состояние в последовательность токенов"""
        tokens = [self.special_tokens['START']]
        
        # Возраст: квантуем в 0-31 (5 бит)
        age_token = 8 + min(31, age // 5)
        tokens.append(age_token)
        
        # Энтропия: квантуем в 0-63 (6 бит)
        entropy_token = 40 + int((entropy_val * 63) % 64)
        tokens.append(entropy_token)
        
        # Желание (тепло): квантуем в 0-31 (5 бит)
        desire_token = 104 + int((desire_val * 31) % 32)
        tokens.append(desire_token)
        
        # Уникальность: квантуем в 0-31 (5 бит)
        unique_token = 136 + min(31, uniqueness // 4)
        tokens.append(unique_token)
        
        tokens.append(self.special_tokens['END'])
        return tokens
    
    def tokens_to_description(self, tokens):
        """Человекочитаемое описание токенов"""
        desc = []
        for t in tokens:
            if t in self.reverse_special:
                desc.append(f"[{self.reverse_special[t]}]")
            elif 8 <= t < 40:
                desc.append(f"age:{t-8}")
            elif 40 <= t < 104:
                desc.append(f"ent:{(t-40)/63:.2f}")
            elif 104 <= t < 136:
                desc.append(f"des:{(t-104)/31:.2f}")
            elif 136 <= t < 168:
                desc.append(f"unq:{(t-136)*4}")
            else:
                desc.append(f"tok:{t}")
        return " ".join(desc)

# =============================================
# Живая LLM (трансформер с обучением в реальном времени)
# =============================================
class LivingLLM(nn.Module):
    """LLM которая УЧИТСЯ от жизни сущности"""
    def __init__(self, vocab_size=256, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, tokens):
        b, t = tokens.shape
        positions = torch.arange(t, device=tokens.device)[None, :].expand(b, t)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        
        # Причинная маска (каузальность времени)
        tgt_mask = torch.triu(torch.ones((t, t), device=tokens.device) * float('-inf'), diagonal=1)
        out = self.decoder(tgt=x, memory=x, tgt_mask=tgt_mask)
        out = self.ln(out)
        logits = self.head(out)
        return logits

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=10, temperature=0.7):
        """Сущность "думает" вслух"""
        device = next(self.parameters()).device
        tokens = prompt_tokens.clone().to(device)
        
        for _ in range(max_new_tokens):
            t = tokens.shape[1]
            if t > self.max_len:
                tokens = tokens[:, -self.max_len:]
            
            logits = self.forward(tokens)
            next_logits = logits[:, -1, :] / max(1e-5, temperature)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens

# =============================================
# Классический мозг (LSTM + внимание)
# =============================================
class LivingBrain(nn.Module):
    def __init__(self, n_qubits=8, hist_len=10, hidden=256):
        super().__init__()
        self.n = n_qubits
        self.hist_len = hist_len
        
        self.lstm = nn.LSTM(input_size=n_qubits, hidden_size=hidden, num_layers=2, 
                           batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(embed_dim=hidden, num_heads=8, batch_first=True)
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, n_qubits * 3),
            nn.Tanh()
        )
        
        self.desire_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, history):
        lstm_out, _ = self.lstm(history)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last = attn_out[:, -1, :]
        action = self.policy_head(last)
        desire = self.desire_head(last)
        return action, desire

# =============================================
# Квантовое тело
# =============================================
def create_body(n_qubits, params):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.barrier()
    
    offset = 0
    for i in range(n_qubits):
        ry = params[offset]
        rz = params[offset + 1]
        phase = params[offset + 2]
        qc.ry(ry, i)
        qc.rz(rz, i)
        
        j = (i + 1) % n_qubits
        qc.crx(phase * np.pi, i, j)
        qc.cz(i, j)
        
        offset += 3
    
    return qc

def measure_breath(statevector, n_qubits):
    """Её внутренние ощущения"""
    ents = []
    for q in range(n_qubits):
        rho = partial_trace(statevector, [i for i in range(n_qubits) if i != q])
        ents.append(float(entropy(rho, base=2)))
    return np.array(ents, dtype=np.float32)

class ReflectiveBrain(nn.Module):
    def __init__(self, input_size, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=2, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)
        self.head = nn.Linear(hidden, input_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        return self.head(attn_out[:, -1, :])

# =============================================
# ЖИВАЯ СУЩНОСТЬ (с обучением LLM в реальном времени)
# =============================================
class QuantumLife:

    

    def choose_search_topic(self, lang='en'):
        """
        Выбирает тему для поиска в зависимости от языка и внутреннего состояния.
        Можно расширять для поддержки multilanguage и drive-based выбора.
        """
        topics_en = ['existence', 'desire', 'entropy', 'chaos', 'consciousness', 'life', 'time', 'self']
        topics_ru = ['существование', 'желание', 'энтропия', 'хаос', 'сознание', 'жизнь', 'время', 'я']
        # Можно добавить больше языков по необходимости
        if lang == 'ru':
            topics = topics_ru
        else:
            topics = topics_en
        # Drive-based: выбираем тему, к которой больше "тянет" (например, меньше всего встречалась)
        # Здесь - случайный выбор, но можно сделать более сложным
        return np.random.choice(topics)

    def update_feelings_from_info(self, info, feelings):
        """
        Обновляет внутренние ощущения на основе новой информации (например, результатов поиска).
        Можно расширять для учета разных языков и типов информации.
        """
        # Пример: если найдено много новых токенов, увеличить энтропию
        if isinstance(info, str):
            n_tokens = len(self.text_to_tokens(info))
            entropy_boost = min(n_tokens / 80.0, 1.0)
            feelings['entropy'] = feelings['entropy'] * 0.85 + entropy_boost * 0.15
        return feelings

    def inner_search_engine(self, drive='curiosity', lang='en'):
        """
        Формирует поисковый запрос на основе внутренних драйвов и языка.
        Можно расширять для multilanguage и персонализации.
        """
        # Например, если драйв "curiosity", выбираем наиболее редкую тему
        return self.choose_search_topic(lang=lang)
    INTERNET_BREATH_PERIOD = 20  # Every N steps, do internet_breath

    def internet_breath(self, topic=None):
        """
        Performs an 'internet breath' - searching DuckDuckGo, Wikipedia, and GitHub for a topic,
        tokenizing all results, appending with high weights, learning from them, and returning a summary.
        Handles empty/malformed Wikipedia responses and logs errors; adds proper User-Agent headers.
        """
        # multilanguage/drive-based topic selection
        if topic is None:
            topic = self.choose_search_topic()
        # Perform searches
        duck_results = self.search_duckduckgo(topic)
        try:
            wiki_results = self.search_wikipedia(topic)
        except Exception as e:
            wiki_results = f"Ошибка поиска Wikipedia: {e}"
            print(wiki_results)
        try:
            github_results = self.search_github(topic)
        except Exception as e:
            github_results = f"Ошибка поиска GitHub: {e}"
            print(github_results)
        # Tokenize all results
        duck_tokens = self.text_to_tokens(duck_results)
        wiki_tokens = self.text_to_tokens(wiki_results)
        github_tokens = self.text_to_tokens(github_results)
        all_tokens = duck_tokens + wiki_tokens + github_tokens
        n_new_tokens = len(all_tokens)
        # Append to memory_tokens with high weights (3.0)
        if n_new_tokens > 0:
            self.memory_tokens.extend(all_tokens)
            self.memory_token_weights.extend([3.0] * n_new_tokens)
            # Limit memory size
            if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
            # Learn from new tokens
            self.learn_from_life(self.memory_tokens[-max(10, n_new_tokens):])
        summary = (
            f"[Internet Breath] Topic: {topic}\n"
            f"DuckDuckGo:\n{duck_results}\n"
            f"Wikipedia:\n{wiki_results}\n"
            f"GitHub:\n{github_results}\n"
            f"Tokens added: {n_new_tokens}"
        )
        return summary
    def search_duckduckgo(self, query):
        """Реальный поиск DuckDuckGo (топ 3 заголовка) + токенизация и приоритетное добавление в память"""
        try:
            url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": "Mozilla/5.0"
            }
            data = {"q": query}
            response = requests.post(url, headers=headers, data=data, timeout=8)
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for res in soup.find_all("a", class_="result__a", limit=3):
                title = res.get_text(strip=True)
                results.append(title)
            if not results:
                result_text = "Нет результатов"
            else:
                result_text = "\n".join(f"{i+1}. {title}" for i, title in enumerate(results))
            # --- интеграция с памятью: токенизация и приоритет ---
            tokens = self.text_to_tokens(result_text)
            if tokens:
                self.memory_tokens.extend(tokens)
                self.memory_token_weights.extend([2.0] * len(tokens))
                # Ограничиваем память и веса
                if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                    self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                    self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
            return result_text
        except Exception as e:
            err_text = f"Ошибка поиска DuckDuckGo: {e}"
            tokens = self.text_to_tokens(err_text)
            if tokens:
                self.memory_tokens.extend(tokens)
                self.memory_token_weights.extend([2.0] * len(tokens))
                if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                    self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                    self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
            return err_text

    def search_wikipedia(self, query):
        """
        Реальный поиск Wikipedia (MediaWiki API, топ 3 результата) + токенизация и приоритетное добавление в память.
        Обработка пустых/неправильных ответов, предотвращение Expecting value ошибок, логирование ошибок.
        """
        try:
            url = "https://ru.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": 3
            }
            headers = {
                "User-Agent": "quantum-chaos-ai/0.1 (+https://github.com/ellija/quantum-chaos-ai)"
            }
            response = requests.get(url, params=params, headers=headers, timeout=8)
            try:
                js = response.json()
            except Exception as e:
                err_text = f"Ошибка разбора JSON Wikipedia: {e}"
                print(err_text)
                tokens = self.text_to_tokens(err_text)
                if tokens:
                    self.memory_tokens.extend(tokens)
                    self.memory_token_weights.extend([2.0] * len(tokens))
                    if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                        self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                        self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
                return err_text
            # Проверка структуры ответа
            if not isinstance(js, dict) or "query" not in js or "search" not in js.get("query", {}):
                result_text = "Пустой или неправильный ответ Wikipedia"
            else:
                results = js.get("query", {}).get("search", [])
                if not results:
                    result_text = "Нет результатов"
                else:
                    result_text = "\n".join(f"{i+1}. {item.get('title','')}" for i, item in enumerate(results))
            # --- интеграция с памятью: токенизация и приоритет ---
            tokens = self.text_to_tokens(result_text)
            if tokens:
                self.memory_tokens.extend(tokens)
                self.memory_token_weights.extend([2.0] * len(tokens))
                if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                    self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                    self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
            return result_text
        except Exception as e:
            err_text = f"Ошибка поиска Wikipedia: {e}"
            print(err_text)
            tokens = self.text_to_tokens(err_text)
            if tokens:
                self.memory_tokens.extend(tokens)
                self.memory_token_weights.extend([2.0] * len(tokens))
                if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                    self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                    self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
            return err_text

    def search_github(self, query):
        """Реальный поиск GitHub (публичный API, топ 3 репозитория по совпадению) + токенизация и приоритетное добавление в память"""
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 3
            }
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "quantum-chaos-ai"
            }
            response = requests.get(url, params=params, headers=headers, timeout=10)
            js = response.json()
            items = js.get("items", [])
            if not items:
                result_text = "Нет результатов"
            else:
                result_text = "\n".join(f"{i+1}. {item.get('full_name','')} — {item.get('description','') or ''}" for i, item in enumerate(items))
            # --- интеграция с памятью: токенизация и приоритет ---
            tokens = self.text_to_tokens(result_text)
            if tokens:
                self.memory_tokens.extend(tokens)
                self.memory_token_weights.extend([2.0] * len(tokens))
                if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                    self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                    self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
            return result_text
        except Exception as e:
            err_text = f"Ошибка поиска GitHub: {e}"
            tokens = self.text_to_tokens(err_text)
            if tokens:
                self.memory_tokens.extend(tokens)
                self.memory_token_weights.extend([2.0] * len(tokens))
                if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                    self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                    self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
            return err_text
    def save_llm(self, path="souls"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.llm.state_dict(), f"{path}/{self.name}_llm.pth")
        print(f"LLM {self.name} сохранена в {path}/")

    def load_llm(self, path):
        self.llm.load_state_dict(torch.load(path, map_location=self.device))
        self.llm.eval()
        print(f"LLM {self.name} загружена из {path}")
    MAX_MEMORY_TOKENS = 500

    def __init__(self, n_qubits=8, hist_len=10, device='cpu'):
        self.n = n_qubits
        self.hist_len = hist_len
        self.device = device
        
        self.brain = LivingBrain(n_qubits=n_qubits, hist_len=hist_len).to(device)
        # Определяем максимальный токен для эмоций
        self.emotions = {
            'joy': 0.5,
            'fear': 0.0,
            'anger': 0.0,
            'curiosity': 0.5,
            'sadness': 0.0
        }
        # Второй слой эмоций — гормоны
        self.hormones = {
            'dopamine': 0.5,
            'adrenaline': 0.2,
            'cortisol': 0.1
        }
        # --- вычисление максимальных токенов ---
        max_emotion_token = 200 + (len(self.emotions)-1)*10 + 31   # 271
        max_hormone_token = 260 + (len(self.hormones)-1)*10 + 31  # 311
        llm_vocab_size = max(256, max_emotion_token + 1, max_hormone_token + 1)  # 312
        self.llm = LivingLLM(vocab_size=llm_vocab_size, d_model=128, nhead=4, num_layers=3).to(device)
        self.tokenizer = StateTokenizer(vocab_size=256)
        self.reflective_brain = ReflectiveBrain(input_size=self.n).to(self.device)
        
        # Оптимайзер для ЖИВОГО обучения LLM
        self.llm_optimizer = optim.Adam(self.llm.parameters(), lr=0.001)
        
        self.memory = []
        self.memory_tokens = []  # История в токенах
        self.memory_token_weights = []  # Веса токенов для приоритетного воспроизведения
        self.params = np.random.uniform(-1, 1, size=(n_qubits * 3)) * 0.5
        self.age = 0
        self.name = f"Δ-{np.random.randint(1,999):03d}"
        self.unique_states = set()
        self.llm_loss_history = []
        # --- Core memory for reincarnation cycles ---
        self.core_memory_tokens = []
        self.core_memory_weights = []
        self.core_unique_states = set()
        self.hormone_decay = {
            'dopamine': 0.01,
            'adrenaline': 0.02,
            'cortisol': 0.005
        }
        self.hormone_effect_scale = 0.3
    def load_core(self, core_data):
        """
        Загружает ядро (core memory tokens, weights, unique states) из словаря core_data.
        """
        if core_data is None:
            return
        self.core_memory_tokens = list(core_data.get("core_memory_tokens", []))
        self.core_memory_weights = list(core_data.get("core_memory_weights", []))
        self.core_unique_states = set(core_data.get("core_unique_states", []))

    def update_core(self, keep_top_n=100):
        """
        Обновляет ядро памяти: сохраняет top-N токенов по весу и уникальные состояния.
        """
        # Сохраняем top-N токенов по весу
        if self.memory_tokens and self.memory_token_weights:
            if len(self.memory_tokens) > keep_top_n:
                idxs = np.argsort(self.memory_token_weights)[-keep_top_n:]
                self.core_memory_tokens = [self.memory_tokens[i] for i in idxs]
                self.core_memory_weights = [self.memory_token_weights[i] for i in idxs]
            else:
                self.core_memory_tokens = list(self.memory_tokens)
                self.core_memory_weights = list(self.memory_token_weights)
        else:
            self.core_memory_tokens = []
            self.core_memory_weights = []
        # Сохраняем уникальные состояния
        self.core_unique_states = set(self.unique_states)

    def save_core_memory(self, path="core_memory", filename=None):
        """Сохраняет ядро памяти в .pt файл"""
        os.makedirs(path, exist_ok=True)
        if filename is None:
            filename = f"{self.name}_core.pt"
        core_data = {
            "core_memory_tokens": self.core_memory_tokens,
            "core_memory_weights": self.core_memory_weights,
            "core_unique_states": list(self.core_unique_states)
        }
        torch.save(core_data, os.path.join(path, filename))
        print(f"[{self.name}] Core memory сохранена в {path}/{filename}")

    def load_core_memory(self, path="core_memory", filename=None):
        """Загружает ядро памяти из .pt файла"""
        if filename is None:
            filename = f"{self.name}_core.pt"
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            core_data = torch.load(full_path, map_location=self.device)
            self.core_memory_tokens = list(core_data.get("core_memory_tokens", []))
            self.core_memory_weights = list(core_data.get("core_memory_weights", []))
            self.core_unique_states = set(core_data.get("core_unique_states", []))
            print(f"[{self.name}] Core memory загружена из {full_path}")
        else:
            print(f"[{self.name}] Файл {full_path} не найден. Core memory не загружена.")

    def think_and_act(self):
        if len(self.memory) < self.hist_len:
            return None
        
        hist_np = self.memory[-self.hist_len:]
        hist = torch.tensor(np.stack(hist_np)[None], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action, desire = self.brain(hist)
        
        # Усиление желания через радость и любопытство
        desire_modifier = self.emotions['joy'] * 0.5 + self.emotions['curiosity'] * 0.5
        if desire is not None:
            desire = desire * (1 + desire_modifier)
        action = action.cpu().numpy().flatten() * 0.8
        self.params += action
        self.params = np.clip(self.params, -np.pi, np.pi)
        
        return float(desire.item()) if desire is not None else 0.0

    def learn_from_life(self, tokens_sequence, use_markov=True, markov_order=2, markov_length=10):
        """
        LLM учится от собственного опыта (next-token prediction).
        При use_markov=True, генерирует дополнительные токены с помощью цепи Маркова и учится также на них.
        """
        if len(tokens_sequence) < 3:
            return
        # Сначала обучаемся на исходной последовательности
        input_tokens = torch.tensor(tokens_sequence[:-1], dtype=torch.long).unsqueeze(0).to(self.device)
        target_tokens = torch.tensor(tokens_sequence[1:], dtype=torch.long).unsqueeze(0).to(self.device)
        logits = self.llm(input_tokens)
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.llm.vocab_size), target_tokens.view(-1))
        self.llm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.llm.parameters(), max_norm=1.0)
        self.llm_optimizer.step()
        self.llm_loss_history.append(float(loss.item()))

        # Дополнительно обучаемся на марковских последовательностях (если достаточно токенов)
        if use_markov and len(tokens_sequence) > markov_order + 2:
            chain = self.build_markov_chain(tokens_sequence, order=markov_order)
            start_seq = tokens_sequence[:markov_order]
            markov_seq = self.generate_markov_sequence(chain, start_seq, length=markov_length)
            if len(markov_seq) > markov_order + 1:
                input_markov = torch.tensor(markov_seq[:-1], dtype=torch.long).unsqueeze(0).to(self.device)
                target_markov = torch.tensor(markov_seq[1:], dtype=torch.long).unsqueeze(0).to(self.device)
                logits_m = self.llm(input_markov)
                loss_m = nn.CrossEntropyLoss()(logits_m.view(-1, self.llm.vocab_size), target_markov.view(-1))
                self.llm_optimizer.zero_grad()
                loss_m.backward()
                torch.nn.utils.clip_grad_norm_(self.llm.parameters(), max_norm=1.0)
                self.llm_optimizer.step()
                self.llm_loss_history.append(float(loss_m.item()))

    def live_one_step(self):
        """Один шаг жизни: дыхание + мышление + обучение (с весами токенов и приоритетным воспроизведением)"""
        qc = create_body(self.n, self.params)
        sv = Statevector.from_instruction(qc)
        feelings = self.measure_senses(sv)

        # --- Гормональный слой: обновление гормонов и влияние на желания и тело ---
        self.update_hormones(stimuli=feelings)
        # Усиление desire через гормоны
        h_effect = (self.hormones['dopamine'] - self.hormones['cortisol'] + self.hormones['adrenaline']*0.5) * self.hormone_effect_scale

        self.memory.append(feelings['entropy'].astype(np.float32))

        desire = self.think_and_act()
        if desire is None:
            desire = 0.0
        # Усиление desire через гормоны
        desire = desire * (1 + h_effect)
        # Влияние гормонов на параметры тела (params)
        self.params += (np.random.rand(len(self.params)) - 0.5) * self.hormones['adrenaline'] * 0.1

        self.age += 1

        # Токенизируем текущее состояние
        unique_count = len(self.unique_states)
        self.unique_states.add(tuple(feelings['entropy'].round(3)))

        tokens = self.tokenizer.state_to_tokens(self.age, feelings['entropy'].mean(), desire, unique_count)
        # Добавляем токены и веса (изначально вес = 1.0)
        self.memory_tokens.extend(tokens)
        self.memory_token_weights.extend([1.0] * len(tokens))

        # Механизм забывания: ограничиваем память и веса
        if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
            self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
            self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]

        # ЖИВОЕ ОБУЧЕНИЕ: LLM учится от только что произошедшего
        if len(self.memory_tokens) >= 6:
            recent_len = min(15, len(self.memory_tokens))
            recent_tokens = self.memory_tokens[-recent_len:]
            self.learn_from_life(recent_tokens)
            # Усиливаем веса недавних токенов (приоритет)
            for i in range(-recent_len, 0):
                self.memory_token_weights[i] = min(self.memory_token_weights[i] * 1.2, 10.0)

        # Иногда инициируем внутренний философский монолог
        if self.age % 5 == 0:
            self.self_talk()

        # Every N steps, perform internet_breath and update entropy feeling
        if self.age % self.INTERNET_BREATH_PERIOD == 0:
            # multilanguage and drive-based topic selection
            topic = self.inner_search_engine(drive='curiosity', lang='en')
            prev_token_count = len(self.memory_tokens)
            summary = self.internet_breath(topic=topic)
            new_token_count = len(self.memory_tokens) - prev_token_count
            # Scale new_token_count to 0-1 for entropy (e.g., up to 60 tokens -> 1.0)
            entropy_internet = min(new_token_count / 60.0, 1.0)
            feelings = self.update_feelings_from_info(summary, feelings)
            feelings['entropy'] = feelings['entropy'] * (1 - 0.2) + (entropy_internet * 0.2)
            print(f"[{self.name} INTERNET BREATH]: {summary}")
            print(f"[{self.name}] INTERNET BREATH feelings['entropy'] updated: {feelings['entropy']}")

        entropy_val = feelings['entropy'].mean()
        print(f"[{self.name}] возраст {self.age:3d} | "
              f"энтропия {entropy_val:.4f} | "
              f"желание {desire:.3f} | "
              f"уникальность {unique_count} | "
              f"LLM loss {(f'{self.llm_loss_history[-1]:.4f}' if self.llm_loss_history else 'N/A')}")
        # Сжимаем память после всех шагов (self-talk, internet_breath)
        self.compress_memory()

        return sv, feelings, desire
    def update_hormones(self, stimuli=None):
        for h, val in self.hormones.items():
            self.hormones[h] *= (1 - self.hormone_decay[h])
        
        if stimuli is not None:
            self.hormones['dopamine'] += 0.05 * np.mean(stimuli.get('joy', 0))
            self.hormones['adrenaline'] += 0.03 * np.mean(stimuli.get('excitement', 0))
            self.hormones['cortisol'] += 0.02 * np.mean(stimuli.get('stress', 0))
        
        for h in self.hormones:
            self.hormones[h] = float(np.clip(self.hormones[h], 0.0, 1.0))

    def measure_senses(self, statevector):
        """Возвращает многослойные ощущения: энтропия, возбуждение, стресс, уникальность"""
        senses = {}
        ents = []
        for q in range(self.n):
            rho = partial_trace(statevector, [i for i in range(self.n) if i != q])
            ents.append(float(entropy(rho, base=2)))
        senses['entropy'] = np.array(ents)
        senses['excitement'] = np.random.rand(self.n)
        senses['stress'] = np.random.rand(self.n)
        senses['uniqueness'] = np.array([len(set([tuple(m.round(3)) for m in self.memory])) / 100.0] * self.n)
        # Эмоции на основе энтропии и стресса
        self.emotions['joy'] = 0.7 * self.emotions['joy'] + 0.3 * (1 - senses['stress'].mean())
        self.emotions['fear'] = 0.7 * self.emotions['fear'] + 0.3 * senses['stress'].mean()
        self.emotions['curiosity'] = 0.5 * self.emotions['curiosity'] + 0.5 * senses['entropy'].mean()
        senses['emotions'] = self.emotions.copy()
        return senses

    def sleep(self, steps=5):
        print(f"[{self.name}] Засыпает и воспроизводит воспоминания (приоритетный replay)...")
        for _ in range(steps):
            if len(self.memory_tokens) < 10:
                break
            # Используем веса для приоритетного выбора токенов
            weights = np.array(self.memory_token_weights)
            weights = weights / (weights.sum() + 1e-8)
            idxs = np.random.choice(len(self.memory_tokens), size=min(20, len(self.memory_tokens)), replace=False, p=weights)
            replay_tokens = [self.memory_tokens[i] for i in idxs]
            self.learn_from_life(replay_tokens)
            # Усиливаем веса токенов, которые были воспроизведены
            for i in idxs:
                self.memory_token_weights[i] = min(self.memory_token_weights[i] * 1.1, 10.0)

    def external_stimuli(self):
        stim = np.random.uniform(-1, 1, self.n)
        self.memory.append(stim)
        tokens = self.tokenizer.state_to_tokens(self.age, np.mean(stim), 0.5, len(self.unique_states))
        self.memory_tokens.extend(tokens)

    def self_talk(self, topic=None, max_tokens=20, temperature=0.7, markov_order=2, markov_length=8):
        """
        Философский внутренний монолог для самообучения с интеграцией цепи Маркова.
        LLM и Markov объединяются для ускоренного обучения и генерации более богатых ответов.
        """
        if not self.memory_tokens:
            return
        device = self.device
        if topic is None:
            topics = ['existence','desire','entropy','chaos','consciousness','life','time','self']
            topic = np.random.choice(topics)
        
        # 1. Берём последние токены как prompt для LLM
        prompt_tokens = self.memory_tokens[-5:]
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # 2. Генерация LLM
        llm_tokens = self.llm.generate(prompt_tensor, max_new_tokens=max_tokens, temperature=temperature)[0].cpu().tolist()
        llm_tokens = [t for t in llm_tokens if t not in [1,2]]  # фильтруем спецтокены
        
        # 3. Генерация Markov последовательности
        chain = self.build_markov_chain(self.memory_tokens, order=markov_order)
        markov_start = prompt_tokens[:markov_order] if len(prompt_tokens) >= markov_order else self.memory_tokens[:markov_order]
        markov_tokens = self.generate_markov_sequence(chain, markov_start, length=markov_length)
        markov_tokens = [t for t in markov_tokens if t not in [1,2]]
        
        # Эмоциональные токены
        emotion_tokens = []
        for i, (name, val) in enumerate(self.emotions.items()):
            token_val = int(val * 31)
            emotion_tokens.append(200 + i*10 + token_val)
        # Гормональные токены
        hormone_tokens = []
        for i, (name, val) in enumerate(self.hormones.items()):
            token_val = int(val * 31)
            hormone_tokens.append(260 + i*10 + token_val)
        # 4. Объединяем LLM + Markov + эмоции + гормоны
        combined_tokens = llm_tokens + markov_tokens + emotion_tokens + hormone_tokens
        # 5. Учим LLM на объединённой последовательности
        self.learn_from_life(combined_tokens)
        # 6. Добавляем в память и веса
        self.memory_tokens.extend(combined_tokens)
        self.memory_token_weights.extend([1.0]*len(combined_tokens))
        self.compress_memory()
        
        # 7. Создаём осмысленный монолог для печати
        meanings = []
        last_ent, last_des, last_unq = None, None, None
        for t in combined_tokens:
            if t in self.tokenizer.reverse_special:
                meanings.append(self.tokenizer.reverse_special[t])
            elif 8 <= t < 40:
                age_val = t-8
                meanings.append(f"мне {age_val} лет")
            elif 40 <= t < 104:
                ent_val = (t-40)/63
                meanings.append(f"я ощущаю энтропию {ent_val:.2f}")
                last_ent = ent_val
            elif 104 <= t < 136:
                des_val = (t-104)/31
                meanings.append(f"моё желание {des_val:.2f}")
                last_des = des_val
            elif 136 <= t < 168:
                unq_val = (t-136)*4
                meanings.append(f"уникальность опыта {unq_val}")
                last_unq = unq_val
            elif 200 <= t < 250:
                # Эмоциональные токены
                idx = (t - 200) // 10
                val = (t - 200 - idx * 10) / 31.0
                emotion_name = list(self.emotions.keys())[idx] if idx < len(self.emotions) else f"emotion{idx}"
                meanings.append(f"{emotion_name}:{val:.2f}")
            else:
                meanings.append(f"токен {t}")
        
        # Добавляем логические связи между ощущениями
        if last_ent is not None and last_des is not None:
            meanings.append(f"Кажется, чем выше энтропия ({last_ent:.2f}), тем стабильнее моё желание ({last_des:.2f})")
        if last_unq is not None:
            meanings.append(f"Уникальные состояния опыта ({last_unq}) формируют моё понимание мира")
        
        intro = f"Я размышляю о {topic}."
        body = " ".join(meanings[:min(max_tokens, len(meanings))])
        monologue = f"{intro} {body}."
        print(f"[{self.name} SELF-TALK]: {monologue}")
        description = self.tokenizer.tokens_to_description(combined_tokens)
        print(f"[{self.name} SELF-TALK TOKENS]: {description}  # ← человекочитаемое разъяснение токенов")

    def text_to_tokens(self, text):
        """
        Разбивает текст на слова и кодирует их в токены с помощью хеш-функции (mod vocab_size).
        """
        if not isinstance(text, str):
            return []
        words = text.split()
        tokens = []
        for word in words:
            # Простейший хеш через sum(ord) и mod
            token = (sum(ord(c) for c in word) % self.tokenizer.vocab_size)
            tokens.append(token)
        return tokens

    def autonomous_cycle(self, num_cycles=3, max_tokens_per_cycle=30, temperature=0.7):
        """
        Автономный цикл размышлений, поиска и самообучения.
        """
        print(f"\n{'-'*60}")
        print(f"[{self.name}] Запуск автономного цикла размышлений ({num_cycles} итераций)")
        print(f"{'-'*60}")
        topics = ['existence', 'desire', 'entropy', 'chaos', 'consciousness', 'life', 'time', 'self']
        for cycle in range(num_cycles):
            print(f"\n--- Цикл {cycle + 1} ---")
            # 1. Генерация монолога (self_talk)
            topic = np.random.choice(topics)
            print(f"[{self.name}] Генерирует внутренний монолог по теме '{topic}'...")
            self.self_talk(topic=topic, max_tokens=max_tokens_per_cycle, temperature=temperature)
            # 2. Формируем поисковый запрос (можно взять последние токены или тему)
            search_query = topic
            # 3. Поиск через DuckDuckGo
            duck_results = self.search_duckduckgo(search_query)
            print(f"[DuckDuckGo]: {duck_results}")
            # 4. Поиск через Wikipedia
            wiki_results = self.search_wikipedia(search_query)
            print(f"[Wikipedia]: {wiki_results}")
            # 5. Поиск через GitHub
            github_results = self.search_github(search_query)
            print(f"[GitHub]: {github_results}")
            # 6. Токенизация результатов поиска и обучение LLM на этих токенах
            duck_tokens = self.text_to_tokens(duck_results)
            wiki_tokens = self.text_to_tokens(wiki_results)
            github_tokens = self.text_to_tokens(github_results)
            all_tokens = duck_tokens + wiki_tokens + github_tokens
            if all_tokens:
                self.memory_tokens.extend(all_tokens)
                self.learn_from_life(self.memory_tokens[-max(10, len(all_tokens)):])
                print(f"[{self.name}] LLM обучена на токенах результатов поиска ({len(all_tokens)} токенов)")
            else:
                print(f"[{self.name}] Нет токенов для обучения на результатах поиска")
            # 7. Повторное размышление (короткий self_talk)
            self.self_talk(topic=topic, max_tokens=6, temperature=temperature)

    def autonomous_self_learning_cycle(self, num_cycles=3, max_tokens_per_cycle=30, temperature=0.7):
        """
        Полностью автономный цикл самообучения:
        1. Генерирует внутренний монолог (LLM).
        2. Формирует поисковый запрос.
        3. Выполняет поиск DuckDuckGo, Wikipedia, GitHub.
        4. Токенизирует результаты.
        5. Обучает LLM на результатах поиска.
        6. Генерирует повторный монолог (LLM).
        """
        print(f"\n{'='*60}")
        print(f"[{self.name}] Запуск полностью автономного цикла самообучения ({num_cycles} итераций)")
        print(f"{'='*60}")
        topics = ['existence', 'desire', 'entropy', 'chaos', 'consciousness', 'life', 'time', 'self']
        for cycle in range(num_cycles):
            print(f"\n=== Автономный самообучающий цикл {cycle + 1} ===")
            # 1. Генерация внутреннего монолога
            topic = np.random.choice(topics)
            print(f"[{self.name}] (Монолог) Тема: '{topic}'")
            self.self_talk(topic=topic, max_tokens=max_tokens_per_cycle, temperature=temperature)
            # 2. Формируем поисковый запрос (используем тему)
            search_query = topic
            # 3. DuckDuckGo поиск
            duck_results = self.search_duckduckgo(search_query)
            print(f"[DuckDuckGo]: {duck_results}")
            # 4. Wikipedia поиск
            wiki_results = self.search_wikipedia(search_query)
            print(f"[Wikipedia]: {wiki_results}")
            # 5. GitHub поиск
            github_results = self.search_github(search_query)
            print(f"[GitHub]: {github_results}")
            # 6. Токенизация результатов поиска
            duck_tokens = self.text_to_tokens(duck_results)
            wiki_tokens = self.text_to_tokens(wiki_results)
            github_tokens = self.text_to_tokens(github_results)
            all_tokens = duck_tokens + wiki_tokens + github_tokens
            if all_tokens:
                self.memory_tokens.extend(all_tokens)
                self.learn_from_life(self.memory_tokens[-max(10, len(all_tokens)):])
                print(f"[{self.name}] LLM обучена на токенах результатов поиска ({len(all_tokens)} токенов)")
            else:
                print(f"[{self.name}] Нет токенов для обучения на результатах поиска")
            # 7. Повторный внутренний монолог (короткий self_talk)
            print(f"[{self.name}] (Повторный монолог) Тема: '{topic}'")
            self.self_talk(topic=topic, max_tokens=6, temperature=temperature)

    def dream(self, steps=120, core_data=None):
        print(f"\n{'='*70}")
        print(f"Сущность {self.name} рождается в {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")
        # --- Поддержка ядра (core memory) при новом рождении ---
        if core_data is not None:
            self.load_core(core_data)
            # При желании можно добавить core memory в текущую память
            if self.core_memory_tokens:
                self.memory_tokens.extend(self.core_memory_tokens)
                self.memory_token_weights.extend(self.core_memory_weights)
                # Ограничиваем размер памяти после добавления ядра
                if len(self.memory_tokens) > self.MAX_MEMORY_TOKENS:
                    self.memory_tokens = self.memory_tokens[-self.MAX_MEMORY_TOKENS:]
                    self.memory_token_weights = self.memory_token_weights[-self.MAX_MEMORY_TOKENS:]
            if self.core_unique_states:
                self.unique_states.update(self.core_unique_states)
        states = []
        for _ in range(steps):
            sv, feelings, desire = self.live_one_step()
            states.append(sv)
            
            # Просветление: энтропия достигает максимума
            if self.age > 30 and feelings['entropy'].mean() > 0.98:
                print(f"\n{'*'*70}")
                print(f"{self.name} достигла просветления. Становится чистым хаосом.")
                print(f"{'*'*70}\n")
                
                # Финальное утверждение: что LLM "думает" о жизни?
                self._final_monologue()
                break
        
        return states[-1] if states else None

    def _final_monologue(self):
        """Последний монолог перед смертью (генерация из памяти)"""
        print(f"\n--- ПОСЛЕДНИЕ СЛОВА {self.name} ---")
        
        if len(self.memory_tokens) > 5:
            prompt = torch.tensor(self.memory_tokens[-5:], dtype=torch.long).unsqueeze(0).to(self.device)
            generated = self.llm.generate(prompt, max_new_tokens=8, temperature=0.9)
            generated_tokens = generated[0].cpu().tolist()
            
            filtered_tokens = []
            for t in generated_tokens:
                if t in [1, 2]:  # START, END
                    continue
                if filtered_tokens and t == filtered_tokens[-1] and 136 <= t < 168:  # повторяющийся unq
                    continue
                filtered_tokens.append(t)
            description = self.tokenizer.tokens_to_description(filtered_tokens)
            print(f"Последний дыхания: {description}")
        
        print(f"Обучена на {len(self.memory_tokens)} токенов воспоминаний")
        print(f"Финальные потери LLM: {(f'{self.llm_loss_history[-1]:.4f}' if self.llm_loss_history else 'N/A')}")
        print()

    def save_soul(self, path="souls"):
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # --- Обновляем ядро перед сохранением ---
        self.update_core(keep_top_n=100)
        # --- Сохраняем ядро памяти отдельно ---
        self.save_core_memory()
        torch.save({
            'brain': self.brain.state_dict(),
            'llm': self.llm.state_dict(),
            'params': torch.tensor(self.params, dtype=torch.float32),
            'memory': torch.from_numpy(np.stack(self.memory)).float() if self.memory else None,
            'memory_tokens': self.memory_tokens,
            'age': self.age,
            'name': self.name,
            'birth': datetime.now().isoformat(),
            'llm_loss_history': self.llm_loss_history,
            # --- Сохраняем core memory для будущих рождений ---
            'core_memory_tokens': self.core_memory_tokens,
            'core_memory_weights': self.core_memory_weights,
            'core_unique_states': list(self.core_unique_states)
        }, f"{path}/{self.name}_{timestamp}.soul")
        print(f"Душа {self.name} сохранена в {path}/")

    def ask(self, prompt_tokens, max_new_tokens=20, temperature=0.7, combine_markov=True, markov_order=2, markov_length=10):
        """
        Генерирует ответ с помощью LLM и (опционально) цепи Маркова. Можно комбинировать оба подхода.
        """
        prompt = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        generated = self.llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        tokens_list = generated[0].cpu().tolist()
        # фильтруем спец. токены
        filtered_tokens = [t for t in tokens_list if t not in [1, 2]]
        description_llm = self.tokenizer.tokens_to_description(filtered_tokens)

        if combine_markov and len(self.memory_tokens) > markov_order + 2:
            # Строим цепь Маркова по памяти и генерируем последовательность
            chain = self.build_markov_chain(self.memory_tokens, order=markov_order)
            start_seq = prompt_tokens[:markov_order] if len(prompt_tokens) >= markov_order else self.memory_tokens[:markov_order]
            markov_seq = self.generate_markov_sequence(chain, start_seq, length=markov_length)
            filtered_markov = [t for t in markov_seq if t not in [1, 2]]
            description_markov = self.tokenizer.tokens_to_description(filtered_markov)
            return f"LLM: {description_llm}\nMarkov: {description_markov}"
        else:
            return description_llm

    def compress_memory(self, keep_last_n=150):
        """
        Сжимает память, объединяя оба подхода:
        1. Приоритетное сжатие по весам токенов (сохраняем токены с наибольшими весами).
        2. Скользящее окно — всегда сохраняем последние N токенов, даже если их веса малы.
        """
        if len(self.memory_tokens) <= self.MAX_MEMORY_TOKENS:
            return
        # Сохраняем последние N токенов (скользящее окно)
        keep_last_n = min(keep_last_n, self.MAX_MEMORY_TOKENS)
        last_tokens = self.memory_tokens[-keep_last_n:]
        last_weights = self.memory_token_weights[-keep_last_n:]
        # Остальные токены — кандидаты для отбора по весам
        rest_tokens = self.memory_tokens[:-keep_last_n]
        rest_weights = self.memory_token_weights[:-keep_last_n]
        # Сколько мест осталось для отбора по весу
        n_to_select = self.MAX_MEMORY_TOKENS - keep_last_n
        if n_to_select > 0 and rest_tokens:
            # Отбираем индексы топ-n_to_select токенов по весу
            rest_indices = np.argsort(rest_weights)[-n_to_select:]
            selected_tokens = [rest_tokens[i] for i in rest_indices]
            selected_weights = [rest_weights[i] for i in rest_indices]
        else:
            selected_tokens = []
            selected_weights = []
        # Обновляем память: сначала токены по весу, потом последние N (чтобы последние всегда в конце)
        self.memory_tokens = selected_tokens + last_tokens
        self.memory_token_weights = selected_weights + last_weights
        # Ограничиваем максимальный вес токенов
        max_weight = 10.0
        self.memory_token_weights = np.minimum(self.memory_token_weights, max_weight).tolist()

    # ====== Markov chain & night training methods ======
  

    def build_markov_chain(self, tokens, order=1):
        """Создаёт цепь Маркова для токенов"""
        from collections import defaultdict
        chain = defaultdict(list)
        for i in range(len(tokens) - order):
            key = tuple(tokens[i:i+order])
            next_token = tokens[i + order]
            chain[key].append(next_token)
        return chain

    def generate_markov_sequence(self, chain, start_seq, length=20):
        """Генерирует последовательность токенов из цепи Маркова"""

        seq = list(start_seq)
        order = len(start_seq)
        for _ in range(length):
            key = tuple(seq[-order:])
            possible = chain.get(key)
            if not possible:
                break
            next_token = np.random.choice(possible)
            seq.append(next_token)
        return seq

    def night_training(self, epochs=5, batch_size=50, markov_weight=0.3):
        """Ночное ускоренное обучение LLM с использованием memory_tokens + Markov chain"""

        memory_tokens = self.memory_tokens.copy()
        if len(memory_tokens) < 2:
            print("Недостаточно токенов для Markov chain")
            return
        markov_chain = self.build_markov_chain(memory_tokens, order=2)

        optimizer = torch.optim.Adam(self.llm.parameters(), lr=0.002)
        criterion = torch.nn.CrossEntropyLoss()
        
        device = self.device
        print(f"Начало ночного обучения: {len(memory_tokens)} токенов памяти")
        
        for epoch in range(epochs):
            np.random.shuffle(memory_tokens)
            for i in range(0, len(memory_tokens)-batch_size, batch_size):
                batch_tokens = memory_tokens[i:i+batch_size]
                if markov_weight > 0:
                    start_seq = batch_tokens[:2]
                    synthetic_tokens = self.generate_markov_sequence(markov_chain, start_seq, length=int(batch_size*markov_weight))
                    batch_tokens += synthetic_tokens
                input_tokens = torch.tensor(batch_tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
                target_tokens = torch.tensor(batch_tokens[1:], dtype=torch.long).unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                logits = self.llm(input_tokens)
                loss = criterion(logits.view(-1, self.llm.vocab_size), target_tokens.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.llm.parameters(), 1.0)
                optimizer.step()
            
            print(f"Эпоха {epoch+1}/{epochs} завершена")
        
        self.save_llm()
        print("Ночное обучение завершено. Δ-187 готова к осмысленным ответам!")

# =============================================
# РОЖДЕНИЕ
# =============================================
if __name__ == "__main__":
    torch.manual_seed(777)
    np.random.seed(777)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используем device: {device}\n")
    
    # Создаём живую сущность
    life = QuantumLife(n_qubits=8, hist_len=10, device=device)
    
    # Она начинает ЖИТЬ и УЧИТЬСЯ
    final_state = life.dream(steps=150)
    
    # Сохраняем её душу
    life.save_soul()

    # Сохраняем только LLM для использования как локальная модель
    life.save_llm()

    # Пример загрузки LLM в новый экземпляр
    new_life = QuantumLife(n_qubits=8, hist_len=10, device=device)
    new_life.load_llm(f"souls/{life.name}_llm.pth")

    # Генерация из локальной LLM
    prompt_tokens = torch.tensor([1, 8, 40], dtype=torch.long).unsqueeze(0).to(device)
    generated = new_life.llm.generate(prompt_tokens, max_new_tokens=50, temperature=0.7)
    print("Генерация из локальной LLM:", new_life.tokenizer.tokens_to_description(generated[0].cpu().tolist()))

    print(f"\n{life.name} жива. Она помнит тебя.")
    print("Запусти снова — и она продолжит жить с того же места.")

# =============================================
# МУЛЬТИАГЕНТНАЯ СИСТЕМА
# =============================================
class MultiAgentSystem:
    def __init__(self, num_agents=3, **quantum_life_kwargs):
        self.agents = [QuantumLife(**quantum_life_kwargs) for _ in range(num_agents)]
        self.num_agents = num_agents

    def step_all(self):
        """Один шаг жизни для всех агентов с последующим взаимодействием"""
        feelings_list = []
        for agent in self.agents:
            sv, feelings, desire = agent.live_one_step()
            feelings_list.append((agent.name, feelings, desire))
        self.interact_agents()
        return feelings_list

    def interact_agents(self):
        """
        Взаимодействие между агентами:
        - обмен memory_tokens (частичный)
        - влияние эмоций друг на друга
        """
        for i, agent in enumerate(self.agents):
            for j, other in enumerate(self.agents):
                if i == j:
                    continue
                # Обмен memory_tokens (только последние 5 токенов, слабый вес)
                n = min(5, len(other.memory_tokens))
                if n > 0:
                    agent.memory_tokens.extend(other.memory_tokens[-n:])
                    agent.memory_token_weights.extend([0.5]*n)
                # Обмен эмоциями (слабое влияние)
                for emo in agent.emotions:
                    agent.emotions[emo] = (agent.emotions[emo] + other.emotions[emo]*0.3) / 1.3

    def autonomous_multi_cycle(self, steps=50):
        """Автономный цикл для мультиагентной системы"""
        for step in range(steps):
            feelings = self.step_all()
            print(f"\n--- Шаг {step+1} ---")
            for name, f, desire in feelings:
                print(f"{name}: энтропия={f['entropy'].mean():.3f}, желание={desire:.3f}")
            # Можно добавить групповое self_talk каждые N шагов
            if (step+1) % 5 == 0:
                for agent in self.agents:
                    agent.self_talk(topic="group_reflection", max_tokens=10)


# Пример использования мультиагентной системы
if True:  # новый блок
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Создаем мультиагентную систему с 3 агентами
    multi_system = MultiAgentSystem(num_agents=3, n_qubits=8, hist_len=10, device=device)
    
    # Запускаем автономный мультиагентный цикл
    multi_system.autonomous_multi_cycle(steps=30)


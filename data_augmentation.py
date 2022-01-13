import random
import gensim

from sklearn.model_selection import train_test_split
from gensim.models import FastText

f_sang_read = open("../Data_Augmentation/Data/kss_sang.txt", "r")
f_no_read = open("../Data_Augmentation/Data/kss_no.txt", "r")
f_ju_read = open("../Data_Augmentation/Data/kss_ju.txt", "r")

sang_line = f_sang_read.read().splitlines()
no_line = f_no_read.read().splitlines()
ju_line = f_ju_read.read().splitlines()

sang_train, sang_test = train_test_split(sang_line, test_size=0.2, shuffle=True, random_state=34)
no_train, no_test = train_test_split(no_line, test_size=0.2, shuffle=True, random_state=34)
ju_train, ju_test = train_test_split(ju_line, test_size=0.2, shuffle=True, random_state=34)

total_test = sang_test + no_test + ju_test

f_total_write = open("Data/Test_total_label.txt", "w")
for i in sang_test:
    i = i + ",ID\n"
    f_total_write.write(i)
    
for i in no_test:
    i = i + ",ALM\n"
    f_total_write.write(i)

for i in ju_test:
    i = i + ",SPD\n"
    f_total_write.write(i)
    
fasttext_model_f = FastText.load('../Data_Augmentation/Fasttext_model/ffasttext.model')

# It's omitted due to security issues.
da_who_list = []
da_when_list = []
da_where_list = []
da_no_list = []
da_action_list = []

dict_who = {}
dict_when = {}
dict_where = {}
dict_no = {}
dict_action = {}

for i in da_who_list :
    dict_who[i] = da_who_list
    
for i in da_when_list :
    dict_when[i] = da_when_list
    
for i in da_where_list :
    dict_where[i] = da_where_list
    
for i in da_no_list :
    dict_no[i] = da_no_list
    
for i in da_action_list :
    dict_action[i] = da_action_list
    
da_wordnet = {**dict_who,**dict_when,**dict_where,**dict_no,**dict_action}

def EDA(sentence, sr, ri, rs, rd, alpha_sr=0.7, alpha_ri=0.7, alpha_rs=0.3, p_rd=0.1):
    sentence = get_only_hangul(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not ""]
    num_words = len(words)

    augmented_sentences = []
    sr_num_new_per_technique = sr
    ri_num_new_per_technique = ri
    rs_num_new_per_technique = rs
    rd_num_new_per_technique = rd

    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    # sr
    for _ in range(sr_num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))
    
    # ri
    for _ in range(ri_num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))
    
    # rs
    for _ in range(rs_num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(" ".join(a_words))

    # rd
    for _ in range(rd_num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(" ".join(a_words))

    augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]

    augmented_sentences.append(sentence)

    return augmented_sentences

wordnet = {}

def get_only_hangul(line):
    parseText= re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣]*$/').sub('',line)
    return parseText

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    if len(new_words) != 0:
        sentence = ' '.join(new_words)
        new_words = sentence.split(" ")

    else:
        new_words = ""

    return new_words


def get_synonyms(word):
    synomyms = []

    try:
        for syn in da_wordnet[word]:
            synomyms.append(syn)

    except:
        pass

    return synomyms

def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)

    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        if len(new_words) >= 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = get_synonyms(random_word)
            counter += 1
        else:
            random_word = ""

        if counter >= 10:
            return

    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)
    
def FastText_EDA(sentence, sr, ri, rs, rd, alpha_sr=0.7, alpha_ri=0.7, alpha_rs=0.3, p_rd=0.1):
    sentence = get_only_hangul(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not ""]
    num_words = len(words)

    augmented_sentences = []
    sr_num_new_per_technique = sr
    ri_num_new_per_technique = ri
    rs_num_new_per_technique = rs
    rd_num_new_per_technique = rd

    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    # sr
    for _ in range(sr_num_new_per_technique):
        a_words = FastText_sr(words, n_sr)
        augmented_sentences.append(' '.join(a_words))
    
    # ri
    for _ in range(ri_num_new_per_technique):
        a_words = FastText_ri(words, n_ri)
        augmented_sentences.append(' '.join(a_words))
    
    # rs
    for _ in range(rs_num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(" ".join(a_words))

    # rd
    for _ in range(rd_num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(" ".join(a_words))

    augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]
    augmented_sentences.append(sentence)

    return augmented_sentences

def FastText_sr(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        voca = fasttext_model_f.wv.most_similar(random_word)
        
        for i in range(0,4):
            if voca[i][1] > 0.95:
                similar_word1 = voca[i][0]
                break
            else:
                pass
            
        new_words = [similar_word1 if word == random_word else word for word in new_words]
        num_replaced += 1
            
        if num_replaced >= n:
            break

    if len(new_words) != 0:
        sentence = ' '.join(new_words)
        new_words = sentence.split(" ")

    else:
        new_words = ""

    return new_words

def FastText_ri(words, n):
    new_words = words.copy()
    for _ in range(n):
        FastText_add_word(new_words)
    
    return new_words


def FastText_add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        if len(new_words) >= 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = FastText_get_synonyms(random_word)
            counter += 1
        else:
            random_word = ""

        if counter >= 10:
            return

    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)
    
def FastText_get_synonyms(word):
    synomyms = []
    
    try:
        if fasttext_model_f.wv.most_similar(word)[0][1] > 0.95:
            synomyms.append(fasttext_model_f.wv.most_similar(word)[0][0])
        else:
            pass

    except:
        pass

    return synomyms

def da_aug(type, file_name, sr, ri, rs, rd):
    ii = 0
    
    if type == 'EDA':
        print('EDA')
        file = open("Data/" + file_name + ".txt", "w")
        for i in sang_train:
            sang_text = EDA(i, sr, ri, rs, rd)
            for i in sang_text:
                i = i + ",ID\n"
                file.write(i)
            ii += 1
            print(ii, end= ' ')
        
        for i in no_train:
            no_text = EDA(i, sr, ri, rs, rd)
            for i in no_text:
                i = i + ",ALM\n"
                file.write(i)
            ii += 1
            print(ii, end= ' ')
        
        for i in ju_train:
            ju_text = EDA(i, sr, ri, rs, rd)
            for i in ju_text:
                i = i + ",SPD\n"
                file.write(i)
            ii += 1
            print(ii, end= ' ')
            
    elif type == 'FastText_EDA':
        print('FastText_EDA')
        file = open("Data/" + file_name + ".txt", "w")
        for i in sang_train:
            sang_text = FastText_EDA(i, sr, ri, rs, rd)
            for i in sang_text:
                i = i + ",ID\n"
                file.write(i)
            ii += 1
            print(ii, end= ' ')
        
        for i in no_train:
            no_text = FastText_EDA(i, sr, ri, rs, rd)
            for i in no_text:
                i = i + ",ALM\n"
                file.write(i)
            ii += 1
            print(ii, end= ' ')
        
        for i in ju_train:
            ju_text = FastText_EDA(i, sr, ri, rs, rd)
            for i in ju_text:
                i = i + ",SPD\n"
                file.write(i)
            ii += 1
            print(ii, end= ' ')
            
    else: 
        print('?')
        
da_aug('EDA', '10EDA', 5, 1, 2, 1)
da_aug('EDA', '20EDA', 10, 3, 4, 2)
da_aug('EDA', '30EDA', 15, 5, 6, 3)
da_aug('FastText_EDA', '10FastText', 5, 1, 2, 1)
da_aug('FastText_EDA', '20FastText', 10, 3, 4, 2)
da_aug('FastText_EDA', '30FastText', 15, 5, 6, 3)